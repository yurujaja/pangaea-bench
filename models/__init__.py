#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .spectralgpt import VisionTransformer
from .prithvi import MaskedAutoencoderViT
from .scalemae import ScaleMAE_baseline
from .croma import CROMA
from .remoteclip import RemoteCLIP

spectral_gpt_vit_base = VisionTransformer
prithvi_vit_base = MaskedAutoencoderViT
scale_mae_large = ScaleMAE_baseline
croma = CROMA
remote_clip = RemoteCLIP