
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models.vision_transformer import DropPath, Mlp
import math
from timm.models.layers import to_2tuple
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class BasicBlock_us(nn.Module):
    def __init__(self, inplanes, upsamp=1):
        super(BasicBlock_us, self).__init__()
        planes = int(inplanes / upsamp)  # assumes integer result, fix later
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsamp = upsamp
        self.couple = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1)
        self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.couple(x)
        residual = self.bnc(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class VisionTransformer(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
            self,
            encoder,
            num_classes=10,
            **kwargs,
    ):
        super().__init__()

        self.embed_dim = self.encoder.embed_dim
        self.img_size = self.encoder.img_size
        self.encoder_type = self.encoder.name
        self.encoder = encoder
        self.num_classes = num_classes

        if self.encoder_type == "spectral_gpt":
            self.L = int(self.img_size/self.encoder.patch_size)**2
            
            self.t = self.encoder.in_chans // self.encoder.t_patch_size
            self.fc = nn.Sequential(
                nn.Linear(self.t, 1))

        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1),
        )

        self.sm = nn.LogSoftmax(dim=1)

        self.decoder = FPNHEAD()

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 8, 8),  # 2048, 16, 16
            # nn.Dropout(0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, 4, 4),  # 2048, 16, 16
            # nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 1024, 1, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 2, 2),  # 2048, 16, 16
            # nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 2048, 1, 1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            # nn.Dropout(0.5)
            # 2048, 16, 16
        )
        self.fc = nn.Sequential(
            nn.Linear(4, 1))

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N = x.shape[0]
        N, T, H, W, p, u, t, h, w = (N, 12, 128, 128, 8, 3, 4, 16, 16)

        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs

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
            seg1 = seg1[:, 1: ,:]

        elif self.encoder_type in ["scale_mae", "ssl4eo_dino", "ssl4eo_moco"]:
            seg1 = self.encoder(x1)
            seg1 = seg1[:, 1: ,:]

        elif self.encoder_type in ["dofa"]:
            seg1 = self.encoder.forward_features(x1, wave_list=self.wave_list)
            seg1 = seg1[:, 1: ,:]

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
            seg1 = seg1[:, 1: ,:]

        elif self.encoder_type in ["gfm_swin"]:
            seg1 = self.encoder.forward_features(x1)

        return seg1

    def forward(self, x1, x2):

        xx1 = self.encoder_single_image(x1)
        xx2 = self.encoder_single_image(x2)

        x = xx1 - xx2  # B, 256, 768
        N, s, _ = x.shape
        w = int(math.sqrt(s, ))
        x = x.reshape(N, w, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        m = {}

        m[0] = self.conv0(x)  # 256,128,128

        m[1] = self.conv1(x)  # 512,64,64

        m[2] = self.conv2(x)  # 1024,32,32

        m[3] = self.conv3(x)  # 2048,16,16

        m = list(m.values())
        x = self.decoder(m)
        x = self.cls_seg(x)
        x = self.sm(x)

        # Match the size between output logits and input image size
        if x.shape[2:] != (self.img_size, self.img_size):
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='nearest')


        return x


class BasicBlock(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None, ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation,
                               dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

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
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=31):
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
            # nn.Dropout(0.5)
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

def cd_vit(encoder, num_classes = 2, **kwargs):
    model = VisionTransformer(
        encoder = encoder,
        num_classes = num_classes,
        **kwargs,
    )
    return model


if __name__ == '__main__':
    input1 = torch.rand(2, 12, 128, 128)
    input2 = torch.rand(2, 12, 128, 128)


    model = cd_vit()
    output = model(input1, input2)
    print((output.shape))




