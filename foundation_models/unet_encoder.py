import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from typing import Sequence
from .base import Base_Encoder
from utils.registry import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class UNet_Encoder(Base_Encoder):
    def __init__(self, cfg, in_channels: int, topology: Sequence[int]):
        super().__init__()
        self.model_name = "unet"
        self.cfg = cfg
        self.in_channels = in_channels
        self.topology = topology
        
        self.in_conv = InConv(self.in_channels, self.topology[0], DoubleConv)
        self.encoder = Encoder(self.topology)

    def forward(self, image):
        x = image['optical']
        feat = self.in_conv(x)
        output = self.encoder(feat)
        return output
    
    def load_encoder_weights(self, pretrained_path):
        missing, incompatible_shape = None, None
        return missing, incompatible_shape


class Encoder(nn.Module):
    def __init__(self, topology: Sequence[int]):
        super(Encoder, self).__init__()

        self.topology = topology

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer
            layer = Down(in_dim, out_dim, DoubleConv)
            down_dict[f'down{idx + 1}'] = layer
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> list:

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        inputs.reverse()
        return inputs


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


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x