#!/usr/bin/env python3

# from .SSL4EO.dino import vision_transformer as dino_vits
# from .SSL4EO.moco_v3 import vits as moco_vits
# from .SSL4EO.mae import models_vit as mae_vit

# moco_vit_small = moco_vits.vit_small
# dino_vit_small = dino_vits.vit_small
# mae_vit_small = mae_vit.vit_small_patch16

####-----NEW
from .spectralgpt import VisionTransformer
from .prithvi import MaskedAutoencoderViT
from .scalemae import ScaleMAE_baseline

spectral_gpt_vit_base = VisionTransformer
prithvi_vit_base = MaskedAutoencoderViT
scale_mae_large = ScaleMAE_baseline