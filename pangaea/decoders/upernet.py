# Adapted from https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.decoders.ltae import LTAE2d, LTAEChannelAdaptor
from pangaea.encoders.base import Encoder


class SegUPerNet(Decoder):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        channels: int,
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
        in_channels: list[int] | None = None,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = "UPerNet"
        self.encoder = encoder
        self.finetune = finetune
        self.feature_multiplier = feature_multiplier

        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.input_layers = self.encoder.output_layers
        self.input_layers_num = len(self.input_layers)

        if in_channels is None:
            self.in_channels = [
                dim * feature_multiplier for dim in self.encoder.output_dim
            ]
        else:
            self.in_channels = [dim * feature_multiplier for dim in in_channels]

        if self.encoder.pyramid_output:
            rescales = [1 for _ in range(self.input_layers_num)]
        else:
            scales = [4, 2, 1, 0.5]
            rescales = [
                scales[int(i / self.input_layers_num * 4)]
                for i in range(self.input_layers_num)
            ]

        self.neck = Feature2Pyramid(
            embed_dim=self.in_channels,
            rescales=rescales,
        )

        self.align_corners = False

        self.channels = channels
        self.num_classes = num_classes

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    padding=0,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=len(self.in_channels) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
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
        # inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T=1 H W) with C the number of encoders'bands for the given modality.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """

        # img[modality] of shape [B C T=1 H W]
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)

            # multi_temporal models can return either (B C' T=1 H' W')
            # or (B C' H' W'), we need (B C' H' W')
            if self.encoder.multi_temporal_output:
                feat = [f.squeeze(-3) for f in feat]

        else:
            # remove the temporal dim
            # [B C T=1 H W] -> [B C H W]
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
            else:
                feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        # fixed bug just for optical single modality
        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(output, size=output_shape, mode="bilinear")

        return output


class SegMTUPerNet(SegUPerNet):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        channels: int,
        multi_temporal: int,
        multi_temporal_strategy: str | None,
        pool_scales: list[int] = [1, 2, 3, 6],
        feature_multiplier: int = 1,
    ) -> None:
        decoder_in_channels = self.get_decoder_in_channels(
            multi_temporal_strategy, encoder
        )
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
            channels=channels,
            pool_scales=pool_scales,
            feature_multiplier=feature_multiplier,
            in_channels=decoder_in_channels,
        )

        self.multi_temporal = multi_temporal
        self.multi_temporal_strategy = multi_temporal_strategy

        # if the encoder deals with multi_temporal inputs and
        # returns time merged outputs then we don't need multi_temporal_strategy
        if self.encoder.multi_temporal and not self.encoder.multi_temporal_output:
            self.tmap = None

        else:
            if self.multi_temporal_strategy == "ltae":
                ltae_in_channels = max(decoder_in_channels)
                # if the encoder output channels vary we must use an adaptor before the LTAE
                if decoder_in_channels != encoder.output_dim:
                    self.ltae_adaptor = LTAEChannelAdaptor(
                        in_channels=encoder.output_dim,
                        out_channels=decoder_in_channels,
                    )
                else:
                    self.ltae_adaptor = lambda x: x
                self.tmap = LTAE2d(
                    positional_encoding=False,
                    in_channels=ltae_in_channels,
                    mlp=[ltae_in_channels, ltae_in_channels],
                    d_model=ltae_in_channels,
                )
            elif self.multi_temporal_strategy == "linear":
                self.tmap = nn.Linear(self.multi_temporal, 1)
            else:
                self.tmap = None

    def get_decoder_in_channels(
        self, multi_temporal_strategy: str | None, encoder: Encoder
    ) -> list[int]:
        if multi_temporal_strategy == "ltae":
            # if the encoder output channels vary we must use an adaptor before the LTAE
            ltae_in_channels = max(encoder.output_dim)
            if ltae_in_channels != min(encoder.output_dim):
                return [ltae_in_channels for _ in encoder.output_dim]
        return encoder.output_dim

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output for multi-temporal data.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T H W) with C the number of encoders'bands for the given modality,
            and T the number of time steps.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """
        # If the encoder handles multi_temporal we feed it with the input
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feats = self.encoder(img)
            else:
                feats = self.encoder(img)
            # multi_temporal models can return either (B C' T H' W')
            # or (B C' H' W') via internal merging strategy

        # If the encoder handles only single temporal data, we apply multi_temporal_strategy
        else:
            feats = []
            for i in range(self.multi_temporal):
                if not self.finetune:
                    with torch.no_grad():
                        feats.append(
                            self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                        )
                else:
                    feats.append(
                        self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                    )

            feats = [list(i) for i in zip(*feats)]
            # obtain features per layer
            feats = [torch.stack(feat_layers, dim=2) for feat_layers in feats]

        if self.tmap is not None:
            if self.multi_temporal_strategy == "ltae":
                feats = self.ltae_adaptor(feats)
                feats = [self.tmap(f) for f in feats]
            elif self.multi_temporal_strategy == "linear":
                feats = [self.tmap(f.permute(0, 1, 3, 4, 2)).squeeze(-1) for f in feats]

        feat = self.neck(feats)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(output, size=output_shape, mode="bilinear")

        return output


class SiamUPerNet(SegUPerNet):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        channels: int,
        strategy: str,
        pool_scales: list[int] = [1, 2, 3, 6],
    ) -> None:
        assert strategy in [
            "diff",
            "concat",
        ], "startegy must be included in [diff, concat]"
        self.strategy = strategy
        if self.strategy == "diff":
            feature_multiplier = 1
        elif self.strategy == "concat":
            feature_multiplier = 2
        else:
            raise NotImplementedError

        encoder.enforce_single_temporal()

        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
            channels=channels,
            pool_scales=pool_scales,
            feature_multiplier=feature_multiplier,
        )

    def encoder_forward(self, img: dict[str, torch.Tensor]) -> list[list[torch.Tensor]]:
        if self.encoder.multi_temporal:
            # Retains the temporal dimension
            img1 = {k: v[:, :, [0], :, :] for k, v in img.items()}
            img2 = {k: v[:, :, [1], :, :] for k, v in img.items()}

            # multi_temporal encoder returns features (B C T=1 H W)
            # or (B C T H W)
            feat1 = self.encoder(img1)
            feat2 = self.encoder(img2)

            if self.encoder.multi_temporal_output:
                feat1 = [f.squeeze(-3) for f in feat1]
                feat2 = [f.squeeze(-3) for f in feat2]

        else:
            img1 = {k: v[:, :, 0, :, :] for k, v in img.items()}
            img2 = {k: v[:, :, 1, :, :] for k, v in img.items()}

            feat1 = self.encoder(img1)
            feat2 = self.encoder(img2)

        return [feat1, feat2]

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Forward function for change detection."""

        if not self.finetune:
            with torch.no_grad():
                feat1, feat2 = self.encoder_forward(img)
        else:
            feat1, feat2 = self.encoder_forward(img)

        if self.strategy == "diff":
            feat = [f2 - f1 for f1, f2 in zip(feat1, feat2)]
        elif self.strategy == "concat":
            feat = [torch.concat((f1, f2), dim=1) for f1, f2 in zip(feat1, feat2)]
        else:
            raise NotImplementedError

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]
        output = F.interpolate(output, size=output_shape, mode="bilinear")

        return output


class SiamDiffUPerNet(SiamUPerNet):
    # Siamese UPerNet for change detection with feature differencing strategy
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        channels: int,
        pool_scales: list[int] = [1, 2, 3, 6],
    ) -> None:
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
            channels=channels,
            strategy="diff",
            pool_scales=pool_scales,
        )


class SiamConcUPerNet(SiamUPerNet):
    # Siamese UPerNet for change detection with feature concatenation strategy
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        channels: int,
        pool_scales: list[int] = [1, 2, 3, 6],
    ) -> None:
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
            channels=channels,
            strategy="concat",
            pool_scales=pool_scales,
        )


class RegUPerNet(Decoder):
    """
    UperNet customized for regression tasks.
    """

    def __init__(
        self,
        encoder: Encoder,
        finetune: bool,
        channels: int,
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
        in_channels: list[int] | None = None,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=1,
            finetune=finetune,
        )

        self.model_name = "Reg_UPerNet"
        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.input_layers = self.encoder.output_layers
        self.input_layers_num = len(self.input_layers)

        if in_channels is None:
            self.in_channels = [
                dim * feature_multiplier for dim in self.encoder.output_dim
            ]
        else:
            self.in_channels = [dim * feature_multiplier for dim in in_channels]

        if self.encoder.pyramid_output:
            rescales = [1 for _ in range(self.input_layers_num)]
        else:
            scales = [4, 2, 1, 0.5]
            rescales = [
                scales[int(i / self.input_layers_num * 4)]
                for i in range(self.input_layers_num)
            ]

        self.neck = Feature2Pyramid(embed_dim=self.in_channels, rescales=rescales)

        self.align_corners = False

        self.channels = channels
        self.num_classes = 1  # regression

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    padding=0,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=len(self.in_channels) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv_reg = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, img, output_shape=None):
        if not self.finetune:
            with torch.no_grad():
                feat = self.encoder(img)
        else:
            feat = self.encoder(img)

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        reg_logit = self.conv_reg(feat)
        output = torch.relu(reg_logit)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]
        output = F.interpolate(output, size=output_shape, mode="bilinear")

        return output


class RegMTUPerNet(RegUPerNet):
    def __init__(
        self,
        encoder: Encoder,
        finetune: bool,
        channels: int,
        multi_temporal: bool | int,
        multi_temporal_strategy: str | None,
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
    ):
        decoder_in_channels = self.get_decoder_in_channels(
            multi_temporal_strategy, encoder
        )
        super().__init__(
            encoder=encoder,
            finetune=finetune,
            channels=channels,
            pool_scales=pool_scales,
            feature_multiplier=feature_multiplier,
            in_channels=decoder_in_channels,
        )
        self.model_name = "Reg_MT_UPerNet"
        self.multi_temporal = multi_temporal
        self.multi_temporal_strategy = multi_temporal_strategy

        if self.multi_temporal_strategy == "ltae":
            ltae_in_channels = max(decoder_in_channels)
            # if the encoder output channels vary we must use an adaptor before the LTAE
            if decoder_in_channels != encoder.output_dim:
                self.ltae_adaptor = LTAEChannelAdaptor(
                    in_channels=encoder.output_dim,
                    out_channels=decoder_in_channels,
                )
            else:
                self.ltae_adaptor = lambda x: x
            self.tmap = LTAE2d(
                positional_encoding=False,
                in_channels=ltae_in_channels,
                mlp=[ltae_in_channels, ltae_in_channels],
                d_model=ltae_in_channels,
            )
        elif self.multi_temporal_strategy == "linear":
            self.tmap = nn.Linear(self.multi_temporal, 1)
        else:
            self.tmap = None

    def get_decoder_in_channels(
        self, multi_temporal_strategy: str | None, encoder: Encoder
    ) -> list[int]:
        if multi_temporal_strategy == "ltae" and encoder.multi_temporal_output:
            # if the encoder output channels vary we must use an adaptor before the LTAE
            ltae_in_channels = max(encoder.output_dim)
            if ltae_in_channels != min(encoder.output_dim):
                return [ltae_in_channels for _ in encoder.output_dim]
        return encoder.output_dim

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        # If the encoder handles multi_temporal we feed it with the input
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feats = self.encoder(img)
            else:
                feats = self.encoder(img)
        else:
            feats = []
            for i in range(self.multi_temporal):
                if not self.finetune:
                    with torch.no_grad():
                        feats.append(
                            self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                        )
                else:
                    feats.append(
                        self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                    )

            feats = [list(i) for i in zip(*feats)]
            feats = [torch.stack(feat_layers, dim=2) for feat_layers in feats]

        if self.tmap is not None:
            if self.multi_temporal_strategy == "ltae":
                feats = self.ltae_adaptor(feats)
                feats = [self.tmap(f) for f in feats]
            elif self.multi_temporal_strategy == "linear":
                feats = [self.tmap(f.permute(0, 1, 3, 4, 2)).squeeze(-1) for f in feats]

        feat = self.neck(feats)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        reg_logit = self.conv_reg(feat)
        output = torch.relu(reg_logit)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]
        output = F.interpolate(output, size=output_shape, mode="bilinear")

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
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
    """

    def __init__(
        self,
        embed_dim,
        rescales=(4, 2, 1, 0.5),
    ):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        self.ops = nn.ModuleList()

        for i, k in enumerate(self.rescales):
            if k == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                        nn.SyncBatchNorm(embed_dim[i]),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                    )
                )
            elif k == 2:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        )
                    )
                )
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif k == 0.25:
                self.ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise KeyError(f"invalid {k} for feature2pyramid")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []

        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        return tuple(outputs)
