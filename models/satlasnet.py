# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/allenai/satlas
Modifications: modifications for compatibility with the benchmark
Authors: Yuru Jia, Valerio Marsocci
'''

import torch
import torch.nn
import requests
import collections
from io import BytesIO
import math
import numpy as np
import torch.nn.functional as F
import torchvision
from enum import Enum, auto
import pdb

def adjust_state_dict_prefix(state_dict, needed, prefix=None, prefix_allowed_count=None):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component. 
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, '', 1)

        new_state_dict[key] = value
    return new_state_dict


class Backbone(Enum):
    SWINB = auto()
    SWINT = auto()
    RESNET50 = auto()
    RESNET152 = auto()

class Head(Enum):
    CLASSIFY = auto()
    MULTICLASSIFY = auto()
    DETECT = auto()
    INSTANCE = auto()
    SEGMENT = auto()
    BINSEGMENT = auto()
    REGRESS = auto()

class FPN(torch.nn.Module):
    def __init__(self, backbone_channels):
        super(FPN, self).__init__()

        out_channels = 128
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        output = list(output.values())

        return output


class Upsample(torch.nn.Module):
    # Computes an output feature map at 1x the input resolution.
    # It just applies a series of transpose convolution layers on the
    # highest resolution features from the backbone (FPN should be applied first).

    def __init__(self, backbone_channels):
        super(Upsample, self).__init__()
        self.in_channels = backbone_channels

        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels

        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch//2, out_channels)
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ch, ch, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
            ch = next_ch
            depth /= 2

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x[0])
        return [output] + x




class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch):
        super(SwinBackbone, self).__init__()

        if arch == 'swinb':
            self.backbone = torchvision.models.swin_v2_b()
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
            self.embed_dim = 128

        elif arch == 'swint':
            self.backbone = torchvision.models.swin_v2_t()
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]

            self.embed_dim = 96
        else:
            raise ValueError("Backbone architecture not supported.")

        self.backbone.features[0][0] = torch.nn.Conv2d(num_channels, self.backbone.features[0][0].out_channels, kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class ResnetBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch='resnet50'):
        super(ResnetBackbone, self).__init__()

        if arch == 'resnet50':
            self.resnet = torchvision.models.resnet.resnet50(weights=None)
            ch = [256, 512, 1024, 2048]
        elif arch == 'resnet152':
            self.resnet = torchvision.models.resnet.resnet152(weights=None)
            ch = [256, 512, 1024, 2048]
        else:
            raise ValueError("Backbone architecture not supported.")

        self.resnet.conv1 = torch.nn.Conv2d(num_channels, self.resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.out_channels = [
            [4, ch[0]],
            [8, ch[1]],
            [16, ch[2]],
            [32, ch[3]],
        ]

    def train(self, mode=True):
        super(ResnetBackbone, self).train(mode)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)

        return [layer1, layer2, layer3, layer4]


class AggregationBackbone(torch.nn.Module):
    def __init__(self, num_channels, backbone):
        super(AggregationBackbone, self).__init__()

        # Number of channels to pass to underlying backbone.
        self.image_channels = num_channels

        # Prepare underlying backbone.
        self.backbone = backbone

        # Features from images within each group are aggregated separately.
        # Then the output is the concatenation across groups.
        # e.g. [[0], [1, 2]] to compare first image against the others
        self.groups = [[0, 1, 2, 3, 4, 5, 6, 7]]

        ngroups = len(self.groups)
        self.out_channels = [(depth, ngroups*count) for (depth, count) in self.backbone.out_channels]

        self.aggregation_op = 'max'

    def forward(self, x):
        # First get features of each image.
        all_features = []
        for i in range(0, x.shape[1], self.image_channels):
            features = self.backbone(x[:, i:i+self.image_channels, :, :])
            all_features.append(features)

        # Now compute aggregation over each group.
        # We handle each depth separately.
        l = []
        for feature_idx in range(len(all_features[0])):
            aggregated_features = []
            for group in self.groups:
                group_features = []
                for image_idx in group:
                    # We may input fewer than the maximum number of images.
                    # So here we skip image indices in the group that aren't available.
                    if image_idx >= len(all_features):
                        continue

                    group_features.append(all_features[image_idx][feature_idx])
                # Resulting group features are (depth, batch, C, height, width).
                group_features = torch.stack(group_features, dim=0)

                if self.aggregation_op == 'max':
                    group_features = torch.amax(group_features, dim=0)

                aggregated_features.append(group_features)

            # Finally we concatenate across groups.
            aggregated_features = torch.cat(aggregated_features, dim=1)

            l.append(aggregated_features)

        return l

class NoopTransform(torch.nn.Module):
    def __init__(self):
        super(NoopTransform, self).__init__()

        self.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=800,
            image_mean=[],
            image_std=[],
        )

    def forward(self, images, targets):
        images = self.transform.batch_images(images, size_divisible=32)
        image_sizes = [(image.shape[1], image.shape[2]) for image in images]
        image_list = torchvision.models.detection.image_list.ImageList(images, image_sizes)
        return image_list, targets

    def postprocess(self, detections, image_sizes, orig_sizes):
        return detections


class FRCNNHead(torch.nn.Module):
    def __init__(self, task, backbone_channels, num_categories=2):
        super(FRCNNHead, self).__init__()

        self.task_type = task
        self.use_layers = list(range(len(backbone_channels)))
        num_channels = backbone_channels[self.use_layers[0]][1]
        featmap_names = ['feat{}'.format(i) for i in range(len(self.use_layers))]
        num_classes = num_categories

        self.noop_transform = NoopTransform()

        # RPN
        anchor_sizes = [[32], [64], [128], [256]]
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = torchvision.models.detection.rpn.RPNHead(num_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=2000)
        rpn_post_nms_top_n = dict(training=2000, testing=2000)
        rpn_nms_thresh = 0.7
        self.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

        # ROI
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7, sampling_ratio=2)
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(backbone_channels[0][1] * box_roi_pool.output_size[0] ** 2, 1024)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, num_classes)
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        self.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if self.task_type == 'instance':
            # Use Mask R-CNN stuff.
            self.roi_heads.mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=featmap_names, output_size=14, sampling_ratio=2)

            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            self.roi_heads.mask_head = torchvision.models.detection.mask_rcnn.MaskRCNNHeads(backbone_channels[0][1], mask_layers, mask_dilation)

            mask_predictor_in_channels = 256
            mask_dim_reduced = 256
            self.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

    def forward(self, image_list, raw_features, targets=None):
        device = image_list[0].device
        images, targets = self.noop_transform(image_list, targets)

        features = collections.OrderedDict()
        for i, idx in enumerate(self.use_layers):
            features['feat{}'.format(i)] = raw_features[idx]

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        losses = {'base': torch.tensor(0, device=device, dtype=torch.float32)}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        loss = sum(x for x in losses.values())
        return detections, loss


class SimpleHead(torch.nn.Module):
    def __init__(self, task, backbone_channels, num_categories=2):
        super(SimpleHead, self).__init__()

        self.task_type = task 
        
        use_channels = backbone_channels[0][1]
        num_layers = 2
        self.num_outputs = num_categories
        if self.num_outputs is None:
            if self.task_type == 'regress':
                self.num_outputs = 1
            else:
                self.num_outputs = 2

        layers = []
        for _ in range(num_layers-1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(use_channels, use_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)

        if self.task_type == 'segment':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        elif self.task_type == 'bin_segment':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            def loss_func(logits, targets):
                targets = targets.argmax(dim=1)
                return torch.nn.functional.cross_entropy(logits, targets, reduction='none')[:, None, :, :]
            self.loss_func = loss_func

        elif self.task_type == 'regress':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            self.loss_func = lambda outputs, targets: torch.square(outputs - targets)

        elif self.task_type == 'classification':
            self.extra = torch.nn.Linear(use_channels, self.num_outputs)
            self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        elif self.task_type == 'multi-label-classification':
            self.extra = torch.nn.Linear(use_channels, self.num_outputs)
            self.loss_func = lambda logits, targets: torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, image_list, raw_features, targets=None):
        raw_outputs = self.layers(raw_features[0])
        loss = None

        if self.task_type == 'segment':
            outputs = torch.nn.functional.softmax(raw_outputs, dim=1)
            if targets is not None:
                task_targets = torch.stack([target for target in targets], dim=0).long()
                loss = self.loss_func(raw_outputs, task_targets)
                loss = loss.mean()

        elif self.task_type == 'bin_segment':
            outputs = torch.nn.functional.softmax(raw_outputs, dim=1)

            if targets is not None:
                task_targets = torch.stack([target for target in targets], dim=0).long()
                loss = self.loss_func(raw_outputs, task_targets)
                loss = loss.mean()

        elif self.task_type == 'regress':
            raw_outputs = raw_outputs[:, 0, :, :]
            outputs = 255*raw_outputs

            if targets is not None:
                task_targets = torch.stack([target for target in targets], dim=0).long()
                loss = self.loss_func(raw_outputs, task_targets.float()/255)
                loss = loss.mean()

        elif self.task_type == 'classification':
            features = torch.amax(raw_outputs, dim=(2,3))
            logits = self.extra(features)
            outputs = torch.nn.functional.softmax(logits, dim=1)

            if targets is not None:
                task_targets = torch.stack([target for target in targets], dim=0).long()
                loss = self.loss_func(logits, task_targets)
                loss = loss.mean()

        elif self.task_type == 'multi-label-classification':
            features = torch.amax(raw_outputs, dim=(2,3))
            logits = self.extra(features)
            outputs = torch.sigmoid(logits)

            if targets is not None:
                task_targets = torch.stack([target for target in targets], dim=0).long()
                loss = self.loss_func(logits, task_targets)
                loss = loss.mean()

        return outputs, loss


class Model(torch.nn.Module):
    def __init__(self, in_chans=3, multi_image=False, img_size = 224, backbone=Backbone.SWINB, fpn=False, head=None, num_categories=None, weights=None):
        """
        Initializes a model, based on desired imagery source and model components. This class can be used directly to
        create a randomly initialized model (if weights=None) or can be called from the Weights class to initialize a 
        SatlasPretrain pretrained foundation model.

        Args:
            in_chans (int): Number of input channels that the backbone model should expect.
            multi_image (bool): Whether or not the model should expect single-image or multi-image input.
            backbone (Backbone): The architecture of the pretrained backbone. All image sources support SwinTransformer.
            fpn (bool): Whether or not to feed imagery through the pretrained Feature Pyramid Network after the backbone.
            head (Head): If specified, a randomly initialized head will be included in the model. 
            num_categories (int): If a Head is being returned as part of the model, must specify how many outputs are wanted.
            weights (torch weights): Weights to be loaded into the model. Defaults to None (random initialization) unless 
                                    initialized using the Weights class.
        """
        super(Model, self).__init__()

        # Validate user-provided arguments.
        # if not isinstance(backbone, Backbone):
        #     raise ValueError("Invalid backbone.")
        # if head and not isinstance(head, Head):
        #    raise ValueError("Invalid head.")
        if head and (num_categories is None):
            raise ValueError("Must specify num_categories if head is desired.")

        self.backbone = self._initialize_backbone(in_chans, backbone, multi_image, weights)
        self.embed_dim = self.backbone.embed_dim
        self.img_size = img_size

        if fpn:
            self.fpn = self._initialize_fpn(self.backbone.out_channels, weights)
            self.upsample = Upsample(self.fpn.out_channels)
        else:
            self.fpn = None

        if head:
            self.head = self._initialize_head(head, self.fpn.out_channels, num_categories) if fpn else self._initialize_head(head, self.backbone.out_channels, num_categories)
        else:
            self.head = None

        self.name = "satlas_pretrain"

    def _initialize_backbone(self, num_channels, backbone_arch, multi_image, weights):
        # Load backbone model according to specified architecture.
        if backbone_arch == Backbone.SWINB or backbone_arch == 'Backbone.SWINB':
            backbone = SwinBackbone(num_channels, arch='swinb')
        elif backbone_arch == Backbone.SWINT or backbone_arch == 'Backbone.SWINT':
            backbone = SwinBackbone(num_channels, arch='swint')
        elif backbone_arch == Backbone.RESNET50 or backbone_arch == 'Backbone.RESNET50':
            backbone = ResnetBackbone(num_channels, arch='resnet50')
        elif backbone_arch == Backbone.RESNET152 or backbone_arch == 'Backbone.RESNET152':
            backbone = ResnetBackbone(num_channels, arch='resnet152')
        else:
            raise ValueError("Unsupported backbone architecture.")
        
        # If using a model for multi-image, need the Aggretation to wrap underlying backbone model.
        prefix, prefix_allowed_count = None, None
        if backbone_arch in [Backbone.RESNET50, Backbone.RESNET152]:
            prefix_allowed_count = 0
        elif multi_image:
            backbone = AggregationBackbone(num_channels, backbone)
            prefix_allowed_count = 2
        else:
            prefix_allowed_count = 1

        # Load pretrained weights into the intialized backbone if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(weights, 'backbone', 'backbone.', prefix_allowed_count)
            backbone.load_state_dict(state_dict)

        return backbone

    def _initialize_fpn(self, backbone_channels, weights):
        fpn = FPN(backbone_channels)

        # Load pretrained weights into the intialized FPN if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(weights, 'fpn', 'intermediates.0.', 0)
            fpn.load_state_dict(state_dict)
        return fpn

    def _initialize_head(self, head, backbone_channels, num_categories):
        # Initialize the head (classification, detection, etc.) if specified
        if head == Head.CLASSIFY:
            return SimpleHead('classification', backbone_channels, num_categories)
        elif head == Head.MULTICLASSIFY:
            return SimpleHead('multi-label-classification', backbone_channels, num_categories)
        elif head == Head.SEGMENT or head == 'segment':
            return SimpleHead('segment', backbone_channels, num_categories)
        elif head == Head.BINSEGMENT:
            return SimpleHead('bin_segment', backbone_channels, num_categories)
        elif head == Head.REGRESS:
            return SimpleHead('regress', backbone_channels, num_categories)
        elif head == Head.DETECT:
            return FRCNNHead('detect', backbone_channels, num_categories)
        elif head == Head.INSTANCE:
            return FRCNNHead('instance', backbone_channels, num_categories)
        return None

    def forward(self, imgs, targets=None):
        # Define forward pass
        x = self.backbone(imgs)
        if self.fpn:
            x = self.fpn(x)
            x = self.upsample(x)
        if self.head:
            x, loss = self.head(imgs, x, targets)
            return x, loss
        return x


if __name__ == "__main__":
    weights_manager = Weights()

    # Test loading in all available pretrained backbone models, without FPN or Head.
    # Test feeding in a random tensor as input.
    for model_id in SatlasPretrain_weights.keys():
        print("Attempting to load ...", model_id)
        model_info = SatlasPretrain_weights[model_id]
        model = weights_manager.get_pretrained_model(model_id)
        rand_img = torch.rand((8, model_info['num_channels'], 128, 128))
        output = model(rand_img)
        print("Successfully initialized the pretrained model with ID:", model_id)

    # Test loading in all available pretrained backbone models, with FPN, without Head.
    # Test feeding in a random tensor as input.
    for model_id in SatlasPretrain_weights.keys():
        print("Attempting to load ...", model_id, " with pretrained FPN.")
        model_info = SatlasPretrain_weights[model_id]
        model = weights_manager.get_pretrained_model(model_id, fpn=True)
        rand_img = torch.rand((8, model_info['num_channels'], 128, 128))
        output = model(rand_img)
        print("Successfully initialized the pretrained model with ID:", model_id, " with FPN.")

    # Test loading in all available pretrained backbones, with FPN and with every possible Head.
    # Test feeding in a random tensor as input. Randomly generated targets are fed into detection/instance heads.
    for model_id in SatlasPretrain_weights.keys():
        model_info = SatlasPretrain_weights[model_id]
        for head in Head:
            print("Attempting to load ...", model_id, " with pretrained FPN and randomly initialized ", head, " Head.")
            model = weights_manager.get_pretrained_model(model_id, fpn=True, head=head, num_categories=2)
            rand_img = torch.rand((1, model_info['num_channels'], 128, 128))

            rand_targets = None
            if head == Head.DETECT:
                rand_targets = [{   
                        'boxes': torch.tensor([[100, 100, 110, 110], [30, 30, 40, 40]], dtype=torch.float32),
                        'labels': torch.tensor([0,1], dtype=torch.int64)
                    }]
            elif head == Head.INSTANCE:
                rand_targets = [{
                        'boxes': torch.tensor([[100, 100, 110, 110], [30, 30, 40, 40]], dtype=torch.float32),
                        'labels': torch.tensor([0,1], dtype=torch.int64),
                        'masks': torch.zeros_like(rand_img)
                    }]
            elif head in [Head.SEGMENT, Head.BINSEGMENT, Head.REGRESS]:
                rand_targets = torch.zeros_like((rand_img))

            output, loss = model(rand_img, rand_targets)
            print("Successfully initialized the pretrained model with ID:", model_id, " with FPN and randomly initialized ", head, " Head.") 
