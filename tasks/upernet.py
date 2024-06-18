# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT
Modifications: support different encoders and multi-temporal support
Authors: Yuru Jia, Valerio Marsocci
'''

from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models.vision_transformer import DropPath, Mlp
from timm.models.layers import to_2tuple
import os

import math

from .ltae import LTAE2d

class UperNetViT(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
            self,
            num_frames,
            encoder,
            num_classes=10,
            mt_strategy = "ltae",
            wave_list = None,
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = encoder

        self.embed_dim = self.encoder.embed_dim
        self.img_size = self.encoder.img_size 

        self.encoder_type = self.encoder.name
        self.multitemporal = True if (num_frames > 1) else False # and encoder_type != "spectral_gpt") else False
        if self.encoder_type in ("prithvi", "spectral_gpt"):
            self.L = int(self.img_size/self.encoder.patch_size)**2

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

        #for DOFA
        self.wave_list = [1,1,1] #wave_list

        if self.encoder_type == "spectral_gpt":
            self.t = self.encoder.in_chans // self.encoder.t_patch_size
            self.fc = nn.Sequential(
                nn.Linear(self.t, 1))
        
        if self.multitemporal:
            self.mt_strategy = mt_strategy
            self.num_frames = num_frames
            if mt_strategy == "linear":
                self.t_map = nn.Sequential(
                    nn.Linear(num_frames, 1))
            elif mt_strategy == "ltae":
                self.t_map = LTAE2d(positional_encoding=False, in_channels=self.embed_dim, 
                                    mlp=[self.embed_dim, self.embed_dim], d_model=self.embed_dim)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_class",
        }
    
    def encoder_single_image(self, x1):
        if self.encoder_type in ["prithvi", "ssl4eo_mae"]:
            seg1, _, _ = self.encoder.forward_encoder(x1, mask_ratio= 0.0)

        elif self.encoder_type in ["ssl4eo_dino", "ssl4eo_moco"]:
            seg1 = self.encoder(x1)

        elif self.encoder_type in ["scale_mae"]:
            seg1 = self.encoder(x1, input_res=10.0)

        elif self.encoder_type in ["dofa"]:
            seg1 = self.encoder.forward_features(x1, wave_list=self.wave_list)

        elif self.encoder_type in ["ssl4eo_data2vec"]:
            seg1 = self.encoder(x1, bool_masked_pos=None, return_all_tokens=True)

        elif self.encoder_type in ["spectral_gpt"]:
            seg1 = self.encoder(x1)
            N, B, C = seg1.shape
            seg1 = seg1.view([N, self.t, self.L, C]) #(Bs, spectral_group, channels, feature_dim)
            seg1 = seg1.permute(0, 2, 3, 1)          #(Bs, channels, feature_dim, spectral_group)
            seg1 = self.fc(seg1).squeeze(dim=-1)     #(Bs, channels, feature_dim)

        elif self.encoder_type in ["croma"]:
            if self.encoder.modality == "optical":
                seg1 = self.encoder(optical_images=x1)[f"{self.encoder.modality}_encodings"]
            elif self.encoder.modality == "SAR":
                seg1 = self.encoder(SAR_images=x1)[f"{self.encoder.modality}_encodings"]
            elif self.encoder.modality == "joint":
                seg1 = self.encoder(optical_images=x1[0], SAR_images=x1[1])[f"{self.encoder.modality}_encodings"]
        
        elif self.encoder_type in ["remote_clip"]:
            seg1 = self.encoder.model.encode_image(x1)

        elif self.encoder_type in ["gfm_swin"]:
            seg1 = self.encoder.forward_features(x1)
            
        elif self.encoder_type in ["satlas_pretrain"]:
            seg1 = self.encoder(x1)[0]

        return seg1
    
    def encoding(self, x1):
        if self.multitemporal:
            if self.encoder_type not in ("prithvi"):
                enc = []
                for i in range(x1.shape[1]):
                    enc.append(self.encoder_single_image(x1[:,i,:,:]))
                return torch.stack(enc, dim = 1)
            else:
                x1 = self.encoder_single_image(x1)
                N, B, C = x1.shape
                x1 = x1[:, 1: ,:]
                return x1.view([N, self.num_frames, self.L, C])
        else:
            return self.encoder_single_image(x1)

    def forward(self, x1):
        seg1 = self.encoding(x1)
        # remove the cls token
        if not self.multitemporal and (self.encoder_type in ["remote_clip", "ssl4eo_dino", "ssl4eo_mae", "dofa", "ssl4eo_moco", "scale_mae", "prithvi"]):  
            seg1 = seg1[:, 1: ,:]
        elif self.multitemporal and (self.encoder_type in ["remote_clip", "ssl4eo_dino", "ssl4eo_moco", "dofa", "ssl4eo_mae", "scale_mae"]):
            seg1 = seg1[:, :, 1: ,:]


        if self.multitemporal and self.encoder_type not in ["satlas_pretrain"]:
            if self.mt_strategy == "linear":
                seg1 = seg1.permute(0, 2, 3, 1)
                seg1 = self.t_map(seg1).squeeze()
            elif self.mt_strategy == "ltae":
                #TO DO clean the code
                N, t, s, c = seg1.shape 
                w = int(math.sqrt(s, ))
                seg1 = seg1.reshape(N, t, w, w, c).permute(0, 1, 4, 2, 3)
                seg1 = self.t_map(seg1).reshape(N, c, s).permute(0, 2, 1)
        
        if self.encoder_type not in ["satlas_pretrain"]:
            N, s, _ = seg1.shape
            w = int(math.sqrt(s, ))
            seg1 = seg1.reshape(N, w, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        m = {}

        m[0] = self.conv0(seg1)  # 256,128,128
        m[1] = self.conv1(seg1)  # 512,64,64
        m[2] = self.conv2(seg1)  # 1024,32,32
        m[3] = self.conv3(seg1)  # 2048,16,16

        m = list(m.values())
        x = self.decoder(m)
        x = self.cls_seg(x)

        # Match the size between output logits and input image size
        if x.shape[2:] != (self.img_size, self.img_size):
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='nearest')

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


def upernet_vit_base(encoder, num_classes = 15, num_frames = 1, **kwargs):
    
    model = UperNetViT(
        encoder = encoder,
        num_frames = num_frames,
        num_classes = num_classes, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model