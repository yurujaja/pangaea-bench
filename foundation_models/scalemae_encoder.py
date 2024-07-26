# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/mvrl/rshf
Modifications: modifications for compatibility with the benchmark
Authors: Yuru Jia, Valerio Marsocci
'''
from functools import partial
from timm.models.vision_transformer import Block, PatchEmbed

import torch
import torch.nn as nn

from .pos_embed import get_2d_sincos_pos_embed_with_resolution
from utils.registry import ENCODER_REGISTRY


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def forward(self, x):
        B, C, H, W = x.shape
        # Dropped size check in timm
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

@ENCODER_REGISTRY.register()
class ScaleMAE_Encoder(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        cfg,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),

    ):
        super().__init__()

        self.input_bands = cfg['input_bands']
        self.output_layers = cfg['output_layers']
        self.model_name = 'ScaleMAE'
        self.in_chans = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedUnSafe(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(
        #    torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        #)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        #self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def load_encoder_weights(self, pretrained_path):
        pretrained_model = torch.load(pretrained_path, map_location="cpu")['model']

        ckpt_patch_embed_weight = pretrained_model["patch_embed.proj.weight"]
        if self.in_chans % ckpt_patch_embed_weight.shape[1] == 0:
            print(
                f"Rescaling pretrained patch_embed weight to fit new {self.in_chans=}"
            )
            new_pe_weight = upscale_patch_embed(
                self.in_chans,
                ckpt_patch_embed_weight,
            )
            pretrained_model["patch_embed.proj.weight"] = new_pe_weight

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
        # embed patches
        #to deal with flop calculator, to be fixed
        x = image['optical']
        B, _, h, w = x.shape
        x = self.patch_embed(x)

        # hack: fixing input res may harm performance!!!
        input_res = torch.tensor([1.]).float()
        input_res = input_res.cpu()

        num_patches = int((h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1]))
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches ** 0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed.to(x.dtype)

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = x[:, 1:].permute(0, 2, 1).view(x.shape[0], -1, self.img_size // self.patch_size,self.img_size // self.patch_size).contiguous()
                output.append(out)

        return output


def upscale_patch_embed(target_chans, weight):
    """change pre-trained 3-channel patch embed weight
    to, e.g., 12 channes
    """
    assert target_chans % weight.shape[1] == 0
    factor = target_chans // weight.shape[1]

    return torch.concat([weight] * factor, axis=1)