# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from logging import Logger

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from pangaea.encoders.unet_encoder import DoubleConv

from .base import Encoder


class VIT_Encoder(Encoder):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        encoder_weights,
        input_size,
        input_bands,
        embed_dim,
        output_layers,
        output_dim,
        download_url,
        patch_size=16,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        Encoder.__init__(
            self,
            model_name="vit_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.patch_size = patch_size
        self.in_chans = len(input_bands["optical"])
        self.patch_embed = PatchEmbed(
            input_size, patch_size, in_chans=self.in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, images):
        x = images["optical"].squeeze(2)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)

            if i in self.output_layers:
                out = x[:, 1:]
                out = (
                    out.transpose(1, 2)
                    .view(
                        x.shape[0],
                        -1,
                        self.input_size // self.patch_size,
                        self.input_size // self.patch_size,
                    )
                    .contiguous()
                )
                output.append(out)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        if self.encoder_weights is None:
            return
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
                pretrained_model.pop(name)
            else:
                pretrained_encoder[name] = pretrained_model.pop(name)

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)


class VIT_EncoderMT(Encoder):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        encoder_weights,
        input_size,
        input_bands,
        embed_dim,
        multi_temporal,
        output_layers,
        output_dim,
        download_url,
        patch_size=16,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        Encoder.__init__(
            self,
            model_name="vit_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=multi_temporal,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.patch_size = patch_size
        self.in_channels = len(input_bands["optical"])
        self.patch_embed = PatchEmbed(
            input_size, patch_size, in_chans=self.in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.time_merging = DoubleConv(
            in_ch=self.in_channels * self.multi_temporal, out_ch=self.in_channels
        )

    def forward(self, images):
        x = images["optical"]
        b, c, t, h, w = x.shape
        # merge time and channels dimension
        x = x.reshape(b, c * t, h, w)
        x = self.time_merging(x)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)

            if i in self.output_layers:
                out = x[:, 1:]
                out = (
                    out.transpose(1, 2)
                    .view(
                        x.shape[0],
                        -1,
                        self.input_size // self.patch_size,
                        self.input_size // self.patch_size,
                    )
                    .contiguous()
                )
                output.append(out)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        if self.encoder_weights is None:
            return
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
                pretrained_model.pop(name)
            else:
                pretrained_encoder[name] = pretrained_model.pop(name)

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
