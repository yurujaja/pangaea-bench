# Adapted from: https://github.com/zhu-xlab/SSL4EO-S12/tree/main

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

from .base import Base_Encoder 
from utils.registry import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class SSL4EO_MOCO_Encoder(Base_Encoder):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 cfg,        
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3,
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop_rate=0.0,
                 qkv_bias=True,
                 **kwargs):
        super().__init__()

        
        self.model_name = "ssl4eo_moco"
        self.output_layers = cfg['output_layers']


        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                norm_layer=norm_layer)
            for i in range(depth)])
       
        self.pos_drop = nn.Dropout(p=drop_rate)
        #self.norm = norm_layer(embed_dim)

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()
 
    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def load_encoder_weights(self, pretrained_path):
        checkpoint= torch.load(pretrained_path, map_location="cpu")
        pretrained_model = checkpoint["state_dict"]
        pretrained_model = {k.replace("module.base_encoder.",""): v for k, v in pretrained_model.items()}
      
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        msg = self.load_state_dict(pretrained_encoder, strict=False)

        return missing, incompatible_shape

    def forward(self, image):
        x = image['optical'].squeeze(2)

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                #out = self.norm(x) if i == 11 else x
                out = x[:, 1:].permute(0, 2, 1).view(x.shape[0], -1, self.img_size // self.patch_size, self.img_size // self.patch_size).contiguous()
                output.append(out)
                #x = self.norm(x)
        return output
    