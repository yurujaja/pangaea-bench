import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
from typing import Optional, Tuple, Type

# from models.common import MLPBlock, LayerScale
from timm.models.vision_transformer import Block
from timm.layers import DropPath
import pdb

from typing import Optional, Tuple, Type

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
class Affine_Coupling(nn.Module):
    """Affine coupling block for invertible NNs
    Source code: https://github.com/xqding/RealNVP/blob/master/script/RealNVP_2D.py
    """
    def __init__(self, mask): #, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = mask.shape[1]
        # self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)

        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim, self.input_dim)
        # self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        # self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        # self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        # init.normal_(self.scale)

        ## layers used to compute translation in affine transformation
        self.translation_fc1 = nn.Linear(self.input_dim, self.input_dim)
        # self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        # self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fc1(x*self.mask))
        # s = torch.relu(self.scale_fc2(s))
        # s = torch.relu(self.scale_fc3(s)) * self.scale
        return s

    def _compute_translation(self, x):
        ## compute translation using unchanged part of x with a neural network
        t = torch.relu(self.translation_fc1(x*self.mask))
        # t = torch.relu(self.translation_fc2(t))
        # t = self.translation_fc3(t)
        return t

    def forward(self, x):
        ## convert latent space variable to observed variable
        s = self._compute_scale(x)
        t = self._compute_translation(x)

        y = self.mask*x + (1-self.mask)*(x*torch.exp(s) + t)
        logdet = torch.sum((1 - self.mask)*s, -1)

        return y, logdet

    def inverse(self, y):
        ## convert observed varible to latent space variable
        s = self._compute_scale(y)
        t = self._compute_translation(y)

        x = self.mask*y + (1-self.mask)*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), -1)

        return x, logdet


class InvertibleBlock(nn.Module):
    def __init__(self, masks): #, hidden_dim):
        super(InvertibleBlock, self).__init__()
        # self.hidden_dim = hidden_dim
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])

        self.affine_couplings = nn.ModuleList(
            [Affine_Coupling(self.masks[i])
             for i in range(len(self.masks))])

    def forward(self, x):
        ## convert latent space variables into observed variables
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet

        ## a normalization layer is added such that the observed variables is within
        ## the range of [-4, 4].
        # logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(y))**2))), -1)
        # y = 4*torch.tanh(y)
        # logdet_tot = logdet_tot + logdet

        return y, logdet_tot


class SpatialSpectralAttention(nn.Module):
    def __init__(
            self,
            pretrained_params,
            dim: int,
            adapter_scale: float,
            spectral_scale: float,
            num_heads: int = 8,
            qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = float(2 * self.head_dim) ** -0.5
        self.adapter_scale = adapter_scale
        self.spectral_scale = spectral_scale

        self.rgb_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.spectral_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Load_pretrained_params
        self.rgb_qkv.weight.data = pretrained_params.attn.qkv.weight.data
        self.rgb_qkv.weight.requires_grad = False

        self.rgb_qkv.bias.data = pretrained_params.attn.qkv.bias.data
        self.rgb_qkv.bias.requires_grad = False

        self.proj.weight.data = pretrained_params.attn.proj.weight.data
        self.proj.weight.requires_grad = False

        self.proj.bias.data = pretrained_params.attn.proj.bias.data
        self.proj.bias.requires_grad = False

    def forward_rgb(self, rgb_embedding: torch.Tensor) -> torch.Tensor:
        B, L, D = rgb_embedding.shape

        rgb_qkv = self.rgb_qkv(rgb_embedding).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        rgb_q, rgb_k, rgb_v = rgb_qkv.unbind(0)
        rgb_q = rgb_q * self.scale
        attn = rgb_q @ rgb_k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        x = attn @ rgb_v
        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        return x

    def forward(self, rgb_embedding: torch.Tensor, spectral_embedding: torch.Tensor) -> torch.Tensor:
        B, L, D = rgb_embedding.shape
        
        rgb_qkv = self.rgb_qkv(rgb_embedding).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        rgb_q, rgb_k, rgb_v = rgb_qkv.unbind(0)
        rgb_q = rgb_q * self.scale

        spectral_qkv = self.spectral_qkv(spectral_embedding).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        spectral_q, spectral_k, spectral_v = spectral_qkv.unbind(0)
        spectral_q = spectral_q * self.scale

        rgb2rgb_attn = rgb_q @ rgb_k.transpose(-2, -1)
        rgb2sp_attn = rgb_q @ spectral_k.transpose(-2, -1)
        sp2rgb_attn = spectral_q @ rgb_k.transpose(-2, -1)
        sp2sp_attn = spectral_q @ spectral_k.transpose(-2, -1)

        attn = rgb2rgb_attn + rgb2sp_attn + sp2rgb_attn + sp2sp_attn
        attn = attn.softmax(dim=-1)

        v = self.adapter_scale * rgb_v + self.spectral_scale * spectral_v
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)

        return x


class AdapterBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        pretrained_backbone: nn.Module,
        dim: int,
        num_heads: int,
        adapter_scale: float,
        spectral_scale: float,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        drop_path: float = 0.,
        init_values: Optional[float] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = SpatialSpectralAttention(
            pretrained_backbone.blocks[0],
            dim,
            adapter_scale,
            spectral_scale,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.adapter_scale = adapter_scale
        self.spectral_scale = spectral_scale

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.mlp.lin1.weight.data = pretrained_backbone.blocks[0].mlp.fc1.weight.data
        self.mlp.lin1.bias.data = pretrained_backbone.blocks[0].mlp.fc1.bias.data
        self.mlp.lin2.weight.data = pretrained_backbone.blocks[0].mlp.fc2.weight.data
        self.mlp.lin2.bias.data = pretrained_backbone.blocks[0].mlp.fc2.bias.data

        self.norm1.weight.data = pretrained_backbone.blocks[0].norm1.weight.data
        self.norm1.weight.requires_grad = False
        self.norm1.bias.data = pretrained_backbone.blocks[0].norm1.bias.data
        self.norm1.bias.requires_grad = False
        self.norm2.weight.data = pretrained_backbone.blocks[0].norm2.weight.data
        self.norm2.weight.requires_grad = False
        self.norm2.bias.data = pretrained_backbone.blocks[0].norm2.bias.data
        self.norm2.bias.requires_grad = False

        for param in self.mlp.parameters():
            param.requires_grad = False

    def forward_rgb_displacement(self, rgb_embedding):
        rgb_embedding = self.norm1(rgb_embedding)
        displacement = self.drop_path1(self.ls1(self.attn.forward_rgb(rgb_embedding)))
        return displacement

    def forward(self, rgb_embedding: torch.Tensor, spectral_embedding: torch.Tensor) -> torch.Tensor:
        shortcut_1 = rgb_embedding + self.spectral_scale * spectral_embedding
        rgb_embedding = self.norm1(rgb_embedding)
        displacement_1 = self.drop_path1(self.ls1(self.attn(rgb_embedding, spectral_embedding)))

        shortcut_2 = shortcut_1 + displacement_1
        displacement_2 = self.drop_path2(self.ls2(self.mlp(self.norm2(shortcut_2))))
        out = shortcut_2 + displacement_2

        return out, displacement_1

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class AdapterLayers(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        # n_bands: int,
        pretrained_backbone: nn.Module,
        embed_dim: int = 256,
        decoder_embed_dim: int = 64,
        depth: int = 8,
        decoder_depth: int = 2,
        num_heads: int = 12,
        decoder_num_heads: int = 32,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.img_size = img_size
        self.patch_size = patch_size
        # self.n_bands = n_bands

        # print(self.embed_dim)
        # print(num_heads)

        # self.pos_embed = pretrained_backbone.pos_embed
        # self.pos_embed.requires_grad = False

        masks = self.random_mask(embed_dim)
        self.T = InvertibleBlock(masks)

        self.block = AdapterBlock(
            pretrained_backbone,
            dim=embed_dim,
            num_heads=num_heads,
            adapter_scale=1/depth,
            spectral_scale=1/depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def random_mask(self, embed_dim):
        mask = torch.zeros((1, embed_dim))
        inds = np.random.choice(np.arange(embed_dim), size=embed_dim//2, replace=False)
        mask[:, inds] = 1
        mask_c = torch.ones((1, embed_dim)) - mask
        return [mask.long(), mask_c.long()]

    # def embed_hyp(self, rgb_embedding: torch.Tensor, spectral_embedding: torch.Tensor) -> torch.Tensor:
    #     spectral_embedding, _ = self.T(spectral_embedding)
    #     return rgb_embedding + spectral_embedding
    
    # def embed_rgb(self, rgb_embedding):
    #     x = rgb_embedding + self.pos_embed
    #     x = self.block.forward_rgb(x)
    #     return x
    
    def forward(self, rgb_embedding: torch.Tensor, spectral_embedding: torch.Tensor, ids_keep) -> torch.Tensor:
        # rgb_embedding includes positional embedding
        x = rgb_embedding
        spectral_embedding, _ = self.T(spectral_embedding)
        assert torch.isnan(spectral_embedding).sum().item() == 0, "Found NaN in spectral embedding"
        # spectral_embedding = self.T(spectral_embedding)
        rgb_norm = torch.linalg.norm(rgb_embedding, dim=-1).mean().item()
        spectral_norm = torch.linalg.norm(spectral_embedding, dim=-1).mean().item()

        d_norm = []
        for d in range(self.depth):
            if d == 0:
                rgb_d = self.block.forward_rgb_displacement(x)
            x, displacement = self.block(x, spectral_embedding)
            assert torch.isnan(x).sum().item() == 0, "Found NaN in displacement"
            displacement_norm = torch.linalg.norm(displacement, dim=-1)
            d_norm.append(displacement_norm.unsqueeze(-1))

        d_norm = torch.cat(d_norm, dim=-1)
        d_norm = torch.sum(d_norm, dim=-1)
        d_norm = d_norm.mean()
        rgb_d_norm = torch.linalg.norm(rgb_d, dim=-1).mean()
        d_loss = torch.abs((d_norm - rgb_d_norm) / d_norm)

        return x, rgb_norm, spectral_norm, d_loss, d_norm.item(), rgb_d_norm.item()
