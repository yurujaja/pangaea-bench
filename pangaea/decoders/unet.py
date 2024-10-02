import torch.nn.functional as F
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Sequence
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder


class UNet(Decoder):
    """
    UNet implementation designed for supervised semantic segmentation tasks. 

    Key Features:
    - Fully supervised: Requires training from scratch, with no pre-trained weights used.
    - Single temporal input: Designed to process single-frame inputs.

    Args:
        encoder (Encoder): The encoder module (U-Net's down-sampling path), expected to provide feature maps at multiple scales.
        num_classes (int): Number of output classes for segmentation.
        finetune (bool): Whether the model is to be fine-tuned (should always be true for UNet).

    Returns:
        torch.Tensor: Output segmentation map.
    """

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )
        assert self.finetune  # the UNet encoder should always be trained

        self.model_name = 'UNet_SingleTemporal_SemanticSegmentation'
        self.align_corners = False
        self.topology = encoder.topology

        self.decoder = UNet_Decoder(self.topology)
        self.conv_seg = OutConv(self.topology[0], self.num_classes)

    def forward(self, img, output_shape=None):
        """Forward function."""
        feat = self.encoder(img)
        feat = self.decoder(feat)
        output = self.conv_seg(feat)
        # output = F.interpolate(output, size=output_shape, mode='bilinear')
        return output


class SiamUNet(Decoder):
    """
    Siamese UNet designed for supervised change detection tasks, where two temporal inputs are processed by 
    a shared encoder. The decoder uses a specified strategy ('diff' or 'concat') to deal with the feature maps.

    Args:
        encoder (Encoder): The shared encoder for both inputs.
        num_classes (int): Number of output classes for the change detection map.
        finetune (bool): Whether the model is to be fine-tuned (should always be true for SiamUNet).
        strategy (str): The strategy used to combine features ('diff' or 'concat').

    Returns:
        torch.Tensor: Output change map.
    """
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        strategy: str,
    ):

        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )
        self.model_name = 'UNet_SingleTemporal_ChangeDetection'
        
        assert self.finetune  # the UNet encoder should always be trained

        self.align_corners = False

        self.strategy = strategy
        if self.strategy == 'diff':
            self.topology = encoder.topology
        elif self.strategy == 'concat':
            self.topology = [2 * features for features in encoder.topology]
        else:
            raise NotImplementedError
        
        self.decoder = UNet_Decoder(self.topology)
        self.conv_seg = OutConv(self.topology[0], self.num_classes)

    def forward(self, img, output_shape=None):
        """Forward function."""
        
        img1 = {k: v[:,:,0,:,:] for k, v in img.items()}
        img2 = {k: v[:,:,1,:,:] for k, v in img.items()}

        feat1 = self.encoder(img1)
        feat2= self.encoder(img2)
 
        if self.strategy == 'diff':
            feat = [f2 - f1 for f1, f2 in zip(feat1, feat2)]
        elif self.strategy == 'concat':
            feat = [torch.concat((f1, f2), dim=1) for f1, f2 in zip(feat1, feat2)]
        else:
            raise NotImplementedError
        
        feat = self.decoder(feat)
        output = self.conv_seg(feat)
        return output


class SiamDiffUNet(SiamUNet):
    # Siamese UNet for change detection with feature differencing strategy
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
            strategy='diff',
        )
    

class SiamConcUNet(SiamUNet):
    # Siamese UNet for change detection with feature concatenation strategy
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
            strategy='concat',
        )


class UNet_Decoder(nn.Module):
    def __init__(self, topology: Sequence[int]):
        super(UNet_Decoder, self).__init__()

        self.topology = topology

        # Variable scale
        n_layers = len(topology)
        up_topo = [topology[0]]  # topography upwards
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last_layer else topology[idx]  # last layer
            up_topo.append(out_dim)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = Up(in_dim, out_dim, DoubleConv)
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:

        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        return x1


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x