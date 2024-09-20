# Adapted from: https://github.com/zhu-xlab/SSL4EO-S12/tree/main

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from utils.registry import ENCODER_REGISTRY

from .base import Base_Encoder
from .pos_embed import get_2d_sincos_pos_embed


class SSL4EO_MAE_OPTICAL_Encoder(Base_Encoder):
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.model_name = "ssl4eo_mae"
        self.output_layers = cfg["output_layers"]

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        # self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def forward(self, image):
        # embed patches
        x = image["optical"].squeeze(2)
        x = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                # out = self.norm(x) if i == 11 else x
                out = (
                    x[:, 1:]
                    .permute(0, 2, 1)
                    .view(
                        x.shape[0],
                        -1,
                        self.img_size // self.patch_size,
                        self.img_size // self.patch_size,
                    )
                    .contiguous()
                )
                output.append(out)

        return output

    def load_encoder_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        pretrained_model = checkpoint["model"]

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


@ENCODER_REGISTRY.register()
class SSL4EO_MAE_SAR_Encoder(SSL4EO_MAE_OPTICAL_Encoder):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        cfg,
        img_size=224,
        patch_size=16,
        in_chans=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            cfg,
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            norm_layer,
        )

    def forward(self, image):
        # embed patches
        x = image["sar"].squeeze(2)
        x = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                # out = self.norm(x) if i == 11 else x
                out = (
                    x[:, 1:]
                    .permute(0, 2, 1)
                    .view(
                        x.shape[0],
                        -1,
                        self.img_size // self.patch_size,
                        self.img_size // self.patch_size,
                    )
                    .contiguous()
                )
                output.append(out)

        return output

    # def load_encoder_weights(self, pretrained_path):
    #     checkpoint= torch.load(pretrained_path, map_location="cpu")
    #     pretrained_model = checkpoint["model"]

    #     k = pretrained_model.keys()
    #     pretrained_encoder = {}
    #     incompatible_shape = {}
    #     missing = {}
    #     for name, param in self.named_parameters():
    #         if name not in k:
    #             missing[name] = param.shape
    #         elif pretrained_model[name].shape != param.shape:
    #             incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
    #         else:
    #             pretrained_encoder[name] = pretrained_model[name]

    #     msg = self.load_state_dict(pretrained_encoder, strict=False)

    #     return missing, incompatible_shape
