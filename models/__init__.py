# -*- coding: utf-8 -*-
''' 
Authors: Yuru Jia, Valerio Marsocci
'''

from .spectralgpt import vit_spectral_gpt
from .prithvi import MaskedAutoencoderViT
from .scalemae import ScaleMAE_baseline
from .croma import croma_vit
from .remoteclip import RemoteCLIP
from .SSL4EO_mae import mae_vit
from .SSL4EO_dino import vit
from .SSL4EO_moco import moco_vit
from .SSL4EO_data2vec import beit 
from .DOFA import dofa_vit
from .gfm_swin import SwinTransformer as GFM_SwinTransformer
from .gfm_swin import adapt_gfm_pretrained
from .satlasnet import Model as SATLASNet

spectral_gpt_vit_base = vit_spectral_gpt
prithvi_vit_base = MaskedAutoencoderViT
scale_mae_large = ScaleMAE_baseline
croma = croma_vit
remote_clip = RemoteCLIP
ssl4eo_dino_small = vit
ssl4eo_moco_small = moco_vit
ssl4eo_data2vec_small = beit
gfm_swin_base = GFM_SwinTransformer
satlasnet = SATLASNet
dofa_vit = dofa_vit
ssl4eo_mae = mae_vit
adapt_gfm_pretrained = adapt_gfm_pretrained