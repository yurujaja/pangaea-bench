
# from .spectralgpt import vit_spectral_gpt
# from .prithvi import MaskedAutoencoderViT
# from .scalemae import ScaleMAE_baseline
# from .croma import croma_vit
# from .remoteclip import RemoteCLIP
# from .ssl4eo_mae import mae_vit
# from .ssl4eo_dino import vit
# from .ssl4eo_moco import moco_vit
# from .ssl4eo_data2vec import beit
# from .dofa import dofa_vit
# from .gfm_swin import SwinTransformer as GFM_SwinTransformer
# from .gfm_swin import adapt_gfm_pretrained
# from .satlasnet import Model as SATLASNet
# from .satlasnet import Weights as SATLASNetWeights
from .prithvi_encoder import Prithvi_Encoder
from .remoteclip_encoder import RemoteCLIP_Encoder
from .scalemae_encoder import ScaleMAE_Encoder
from .croma_encoder import CROMA_OPTICAL_Encoder, CROMA_SAR_Encoder, CROMA_JOINT_Encoder
from .spectralgpt_encoder import SpectralGPT_Encoder
from .dofa_encoder import DOFA_Encoder
from .gfmswin_encoder import GFMSwin_Encoder
from .satlasnet_encoder import SatlasNet_Encoder
from .ssl4eo_dino_encoder import SSL4EO_DINO_Encoder
from .ssl4eo_moco_encoder import SSL4EO_MOCO_Encoder
from .ssl4eo_data2vec_encoder import SSL4EO_Data2Vec_Encoder
from .ssl4eo_mae_encoder import SSL4EO_MAE_OPTICAL_Encoder, SSL4EO_MAE_SAR_Encoder
#
# spectral_gpt_vit_base = vit_spectral_gpt
# prithvi_vit_base = MaskedAutoencoderViT
# scale_mae_large = ScaleMAE_baseline
# croma = croma_vit
# remote_clip = RemoteCLIP
# ssl4eo_dino_small = vit
# ssl4eo_moco = moco_vit
# ssl4eo_data2vec_small = beit
# gfm_swin_base = GFM_SwinTransformer
# satlasnet = SATLASNet
# dofa_vit = dofa_vit
# ssl4eo_mae = mae_vit
# adapt_gfm_pretrained = adapt_gfm_pretrained