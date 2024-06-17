#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .spectralgpt import vit_spectral_gpt #VisionTransformer
from .prithvi import MaskedAutoencoderViT
from .scalemae import ScaleMAE_baseline
from .croma import croma_vit
from .remoteclip import RemoteCLIP
from .SSL4EO_mae import mae_vit #mae_vit_base_patch16_dec512d8b, mae_vit_huge_patch14_dec512d8b, mae_vit_large_patch16_dec512d8b, mae_vit_small_patch16_dec512d8b
from .SSL4EO_dino import vit_small
from .SSL4EO_moco import moco_vit_small
from .SSL4EO_data2vec import beit_small_patch16_224 
from .DOFA import dofa_vit #vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14
from .gfm_swin import SwinTransformer as GFM_SwinTransformer
from .gfm_swin import adapt_gfm_pretrained
from .satlasnet import Model as SATLASNet

spectral_gpt_vit_base = vit_spectral_gpt
prithvi_vit_base = MaskedAutoencoderViT
scale_mae_large = ScaleMAE_baseline
croma = croma_vit
remote_clip = RemoteCLIP
ssl4eo_dino_small = vit_small
ssl4eo_moco_small = moco_vit_small
ssl4eo_data2vec_small = beit_small_patch16_224
# dofa_vit_small = vit_small_patch16
gfm_swin_base = GFM_SwinTransformer
satlasnet = SATLASNet
dofa_vit = dofa_vit
ssl4eo_mae = mae_vit
adapt_gfm_pretrained = adapt_gfm_pretrained