#!/usr/bin/env python3

from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models.vision_transformer import DropPath, Mlp
from timm.models.layers import to_2tuple
import os

import math

# from ..models import *

class UperNetViT(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
            self,
            num_frames,
            t_patch_size,
            encoder,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=10,
            embed_dim=768,
            # depth=12,
            # num_heads=12,
            mlp_ratio=4.0,
            no_qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.5,
            norm_layer=nn.LayerNorm,
            dropout=0.5,  # 0.5
            sep_pos_embed=True,
            cls_embed=False,
            encoder_type="prithvi",
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type
        self.img_size = img_size
        self.L = int(img_size/patch_size)**2

        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1),
        )
        self.decoder = FPNHEAD()

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 8, 8),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, 4, 4),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 1024, 1, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 2, 2),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 2048, 1, 1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            nn.Dropout(0.5)
            # 2048, 16, 16
        )

        self.t = num_frames // t_patch_size
        self.fc = nn.Sequential(
            nn.Linear(self.t, 1))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_class",
        }

    def forward(self, x1):
        # TODO: to support other models
        if self.encoder_type in ["prithvi", "mae"]:
            seg1, _, _ = self.encoder.forward_encoder(x1, mask_ratio= 0.0)
        elif self.encoder_type in ["prithvi", "scale_mae", "spectral_gpt"]:
            seg1 = self.encoder(x1)
        elif self.encoder_type in ["croma"]:
            seg1 = self.encoder(x1)["optical_encodings"]
        elif self.encoder_type in ["remote_clip"]:
            seg1 = self.encoder.model.encode_image(x1)
        
        # remove the cls token
        if self.encoder_type in ["remote_clip", "scale_mae", "prithvi"]:
            seg1 = seg1[:, 1: ,:]

        N, B, C = seg1.shape
        
        # for single temporal we basically squeeze 
        if self.t == 1:
            seg1 = seg1.view([N, self.t, B, C])
        else:
            seg1 = seg1.view([N, self.t, self.L, C])


        seg1 = seg1.permute(0, 2, 3, 1)
        seg1 = self.fc(seg1)

        _, s, _, _ = seg1.shape
        w = int(math.sqrt(s, ))
        seg1 = seg1.reshape(N, w, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        m = {}

        # m[0] = self.conv0(x)  # 256,128,128
        m[0] = self.conv0(seg1)  # 256,128,128

        # m[1] = self.conv1(x)  # 512,64,64
        m[1] = self.conv1(seg1)  # 512,64,64

        # m[2] = self.conv2(x)  # 1024,32,32
        m[2] = self.conv2(seg1)  # 1024,32,32

        # m[3] = self.conv3(x)  # 2048,16,16
        m[3] = self.conv3(seg1)  # 2048,16,16

        m = list(m.values())
        x = self.decoder(m)
        x = self.cls_seg(x)
        # x = self.sm(x)

        # Match the size between output logits and input image size
        if x.shape[2:] != (self.img_size, self.img_size):
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # return {'out': x}
        return x

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=13):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            # nn.BatchNorm2d(self.out_channels),
            nn.GroupNorm(16, self.out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, channels=2048, out_channels=256):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        # b, 512, 7, 7
        x1 = self.PPMHead(input_fpn[-1])

        x = nn.functional.interpolate(x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)

        x = nn.functional.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)

        x = nn.functional.interpolate(x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x


def upernet_vit_base(encoder, num_classes = 15, in_chans = 1, t_patch_size = 1, num_frames = 1, patch_size = 16, img_size = 224, embed_dim = 768, **kwargs):
    
    model = UperNetViT(
        img_size=img_size,
        in_chans=in_chans,
        encoder = encoder,
        patch_size=patch_size,
        embed_dim=embed_dim,
        # depth=12,
        # num_heads=12,
        mlp_ratio=4,
        num_frames=num_frames,
        t_patch_size=t_patch_size,
        num_classes = num_classes, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model