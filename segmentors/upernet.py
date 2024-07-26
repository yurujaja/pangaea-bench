# -*- coding: utf-8 -*-

from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn

import os
import sys
import math

from adapters.adapters import add_adapter
from utils.embeddings.patch_embed import RandomPatchEmbed, hyper_embedding
from adapters.mae import MaeEncoder
from adapters.adapter_layers import AdapterLayers
from utils.registry import SEGMENTOR_REGISTRY

@SEGMENTOR_REGISTRY.register()
class UPerNet(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, args, cfg, encoder, pool_scales=(1, 2, 3, 6)):
        super().__init__()

        # self.frozen_backbone = frozen_backbone

        self.model_name = 'UPerNet'
        self.encoder = encoder
        self.finetune = args.finetune

        
        for param in self.encoder.parameters():
            param.requires_grad = False
        # else:
        if self.finetune == "norm_tuning":
            for n, p in self.encoder.named_parameters():
                if "norm" in n:
                    p.requires_grad = True
        elif self.finetune == "bias_tuning":
            for n, p in self.encoder.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif self.finetune == "full_finetuning":
            for param in self.encoder.parameters():
                param.requires_grad = True
        elif self.finetune == "patch_embed":
            for p in self.encoder.patch_embed.parameters():
                p.requires_grad = True
        elif self.finetune == "lora":
            self.encoder = add_adapter(
                self.encoder,
                hidden_dim = 16,
                type="lora"
            )
        elif self.finetune == "lora_patch_embed":
            self.encoder = add_adapter(
                self.encoder,
                patch_embed_adapter = True,
                hidden_dim = 16,
                type="lora",
             )
        elif self.finetune == "low-rank-scaling":
            self.encoder = add_adapter(
                self.encoder, 
                type = "low-rank-scaling",
                shared = False,
                scale = 2.0,
                hidden_dim=16,
            )
        elif self.finetune == "low-rank-scaling-patch-embed":
            self.encoder = add_adapter(
                self.encoder, 
                type = "low-rank-scaling",
                patch_embed_adapter = True,
                shared = False,
                scale = 2.0,
                hidden_dim=16,
            )
        elif self.finetune == "ia3":
            self.encoder = add_adapter(
                self.encoder,
                type="ia3"
            )

        # else:
        #     sys.exit("Error, your method is not available (yet)")
        

        self.neck = Feature2Pyramid(embed_dim=cfg['in_channels'], rescales=[4, 2, 1, 0.5])

        self.align_corners = False

        self.in_channels = [cfg['in_channels'] for _ in range(4)]
        self.channels = cfg['channels']
        self.num_classes = cfg['num_classes']

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                      out_channels=self.channels,
                      kernel_size=3,
                      padding=1),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True)
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=self.channels,
                          kernel_size=1,
                          padding=0),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False)
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(in_channels=self.channels,
                          out_channels=self.channels,
                          kernel_size=3,
                          padding=1),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False)
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


        self.fpn_bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=len(self.in_channels) * self.channels,
                          out_channels=self.channels,
                          kernel_size=3,
                          padding=1),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=True)
        )

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        #inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats



    def forward(self, img, output_shape=None):
        """Forward function."""
        # if self.freezed_backbone:
        #     with torch.no_grad():
        #         feat = self.backbone(img)
        # else:
        if not self.finetune:
            with torch.no_grad():
                feat = self.encoder(img)
        else:
            feat = self.encoder(img)
        # print(feat.shape)

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        #bug fixed adding "optical"
        if output_shape is None:
            output_shape = img["optical"].shape[-2:]
        output = F.interpolate(output, size=output_shape, mode='bilinear')

        return output

@SEGMENTOR_REGISTRY.register()
class UPerNetAdapt(UPerNet):
    def __init__(self, args, cfg, encoder, pool_scales=(1, 2, 3, 6)):
        super().__init__(args, cfg, encoder, pool_scales=(1, 2, 3, 6))   

        self.spectral_patch_embed = RandomPatchEmbed(img_size=self.encoder.img_size, patch_size=self.encoder.patch_size)
        # print("CHANNELS", cfg.bands)
        # print(cfg)
        self.mae_encoder = MaeEncoder(n_bands=cfg["adaptation"]["bands"], embed_dim=self.encoder.embed_dim, checkpoint=cfg["adaptation"]["pretrained_mae_path"])

        # Adapter Layers
        self.adapter_layers = AdapterLayers(
            img_size=self.encoder.img_size,
            patch_size=self.encoder.patch_size,
            # n_bands=self.encoder.in_chans,
            pretrained_backbone=self.encoder,
            embed_dim=self.encoder.embed_dim,
            # decoder_embed_dim=scale_mae_config['decoder_embed_dim'],
            num_heads=16,
            depth=24
            )#.to(args.device)
        # adapter_layers = adapter_layers

        # self.adapter_layers 

        self.avg_pool = nn.AvgPool1d(8)

        blocks = [self.encoder.blocks[i] for i in range(1, len(self.encoder.blocks))]
        # print(blocks)
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, img, output_shape=None):
        rgb_patch, hyper_patch = img["rgb"], img["optical"]

        # print(image.keys())

        rgb_data = self.encoder.patch_embed(rgb_patch)
        # print("HERE! RGB EMBEDDER OK")
        hyper_data = hyper_embedding(hyper_patch, self.spectral_patch_embed, self.mae_encoder, None)
        # print("HERE! HYPER EMBEDDER OK")
        # print(rgb_data.shape)
        # print(hyper_data.shape)
        # REVIEW!!
        # hyper_data = self.avg_pool(hyper_data)
        # print(hyper_data.shape)

        features, rgb_norm, spectral_norm, d_loss, d_norm, rgb_d_norm = self.adapter_layers(rgb_data, hyper_data, None)
        # print("HERE! ADAPTER LAYER OK")

        # for blk in self.blocks:
        #     features = blk(features)

        # print(features.shape)
        #TOKEN ALREADY REMOVED CHECK
        feat = []
        for i, blk in enumerate(self.blocks):
            features = blk(features)
            if i+1 in self.encoder.output_layers:
                out = features.permute(0, 2, 1).view(features.shape[0], -1, self.encoder.img_size // self.encoder.patch_size,self.encoder.img_size // self.encoder.patch_size).contiguous()
                feat.append(out)
        # x = self.norm(x)
        # print(features.shape)
        # print(len(feat))

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        if output_shape is None:
            output_shape = rgb_patch.shape[-2:]
        output = F.interpolate(output, size=output_shape, mode='bilinear')

        return output, [rgb_norm, spectral_norm, d_loss, d_norm, rgb_d_norm]

@SEGMENTOR_REGISTRY.register()
class UPerNetCD(UPerNet):
    def __init__(self, args, cfg, encoder, pool_scales=(1, 2, 3, 6)):
        super().__init__(args, cfg, encoder, pool_scales=(1, 2, 3, 6))   

    def forward(self, image, output_shape=None):
        """Forward function for change detection."""
        img1, img2 = image["t0"], image["t1"]

        if not self.finetune:
            with torch.no_grad():
                feat1 = self.encoder(img1)
                feat2 = self.encoder(img2)
        else:
            feat1 = self.encoder(img1)
            feat2 = self.encoder(img2)
        #print(feat)

        feat = feat2-feat1 

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        if output_shape is None:
            output_shape = img1.shape[-2:]
        output = F.interpolate(output, size=output_shape, mode='bilinear')

        return output


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.channels,
                              kernel_size=1,
                              padding=0),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=True)
                    ))


    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 embed_dim,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)