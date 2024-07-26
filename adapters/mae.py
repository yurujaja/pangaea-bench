# Code copied and slightly modified from https://github.com/facebookresearch/mae and
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

import torch
import torch.nn as nn
from typing import Callable, Optional, Union
from enum import Enum
from itertools import repeat
import collections.abc
import numpy as np


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
to_1tuple = _ntuple(1)

FormatT = Union[str, Format]

def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim


def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def nhwc_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NCHW:
        x = x.permute(0, 3, 1, 2)
    elif fmt == Format.NLC:
        x = x.flatten(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(1, 2).transpose(1, 2)
    return x

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

class SeqEmbed(nn.Module):
    """ Spectrum (batch_size x n_bands) to Sequence Embedding (batch_size x n_sequences x dim_embedding)
    """
    def __init__(
            self,
            n_bands: int = 310,
            seq_size: int = 5,
            in_chans: int = 1,
            embed_dim: int = 32,
            norm_layer: Optional[Callable] = None,
            flatten: bool = False,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_sp_size: bool = True,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.n_bands = n_bands
        self.num_sequences = self.n_bands // self.seq_size

        self.flatten = flatten
        self.strict_sp_size = strict_sp_size

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=seq_size, stride=seq_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, n_bands = x.shape
        x = x.unsqueeze(1)
        if self.strict_sp_size:
            _assert(n_bands == self.n_bands, f"Spectral dimension ({n_bands}) doesn't match model ({self.n_bands}).")
        else:
            _assert(
                n_bands % self.seq_size == 0,
                f"Spectral dimension ({n_bands}) should be divisible by sequence size ({self.seq_size})."
            )

        x = self.proj(x).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            qk_scale=None,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    grid = np.arange(grid_size, dtype=float)
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = np.einsum('m,d->md', grid, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb

class MaeEncoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, n_bands=310, seq_size=5, in_chans=1,
                 embed_dim=32, depth=4, num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, cls_token=True,
                 checkpoint=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.seq_embed = SeqEmbed(n_bands, seq_size, in_chans, embed_dim)
        num_sequences = self.seq_embed.num_sequences

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.is_cls_token = True
        else:
            self.is_cls_token = False

        self.pos_embed = nn.Parameter(torch.zeros(1, num_sequences + np.sum(self.is_cls_token), embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # n_params = 0
        # for param in self.parameters():
        #     n_params += param.shape.numel()
        # print(f'Encoder has {n_params} parameters.')
        # --------------------------------------------------------------------------

        if checkpoint:
            checkpoint = torch.load(checkpoint, map_location='cpu')
            params = checkpoint['state_dict']

            # seq_embed
            self.seq_embed.proj.weight.data = params['seq_embed.proj.weight']
            self.seq_embed.proj.bias.data = params['seq_embed.proj.bias']

            # pos_embed
            self.pos_embed.data = params['pos_embed']

            # cls_token
            self.cls_token.data = params['cls_token']

            # blocks
            for block_id in range(4):
                self.blocks[block_id].attn.qkv.weight.data = params[f'blocks.{block_id}.attn.qkv.weight']
                self.blocks[block_id].attn.qkv.bias.data = params[f'blocks.{block_id}.attn.qkv.bias']
                self.blocks[block_id].attn.proj.weight.data = params[f'blocks.{block_id}.attn.proj.weight']
                self.blocks[block_id].attn.proj.bias.data = params[f'blocks.{block_id}.attn.proj.bias']

                self.blocks[block_id].norm1.weight.data = params[f'blocks.{block_id}.norm1.weight']
                self.blocks[block_id].norm1.bias.data = params[f'blocks.{block_id}.norm1.bias']
                self.blocks[block_id].norm2.weight.data = params[f'blocks.{block_id}.norm2.weight']
                self.blocks[block_id].norm2.bias.data = params[f'blocks.{block_id}.norm2.bias']

                self.blocks[block_id].mlp.fc1.weight.data = params[f'blocks.{block_id}.mlp.fc1.weight']
                self.blocks[block_id].mlp.fc1.bias.data = params[f'blocks.{block_id}.mlp.fc1.bias']
                self.blocks[block_id].mlp.fc2.weight.data = params[f'blocks.{block_id}.mlp.fc2.weight']
                self.blocks[block_id].mlp.fc2.bias.data = params[f'blocks.{block_id}.mlp.fc2.bias']

            # norm
            self.norm.weight.data = params['norm.weight']
            self.norm.bias.data = params['norm.bias']
                
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        # embed patches
        x = self.seq_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, np.sum(self.is_cls_token):, :]

        # append cls token
        if self.is_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x