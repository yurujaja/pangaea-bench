import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.embeddings.pos_embed import get_2d_sincos_pos_embed_with_resolution


class RandomPatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size: int = 96,
            patch_size: int = 12,
            mask_ratio : float = 0.90,
            n_features : int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]
        self.num_patches = (img_size // self.patch_size)**2
        self.mask_ratio = mask_ratio
        self.n_features = n_features

    def random_mask(self, x):
        batch_size, n_channels, H, W = x.shape
        L = self.patch_size**2
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(batch_size, self.num_patches, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]
        return ids_keep

    def forward(self, x):
        """
        Randomly selects pixels within a patch
        :returns: a (batch_size, n_channels, n_patches, n_pixels_in_patch * (1 - mask_ratio)) tensor
        """
        # TODO: replace by patchify method
        # Toy test 1: H = 24, W = 24, patch_size = 12
        # A = torch.zeros((24, 24)); A[:12, 12:] = 1; A[12:, :12] = 2; A[12:, 12:] = 3
        # fig = plt.figure(); plt.imshow(A); plt.show()
        # B = A.view(2, 12, 2, 12); B = B.transpose(1, 2).contiguous(); B = B.view(4, -1)
        # fig = plt.figure(); plt.plot(B.view(-1)); plt.show()
        batch_size, n_channels, H, W = x.shape
        ids_keep = self.random_mask(x)
        x = x.view(
            batch_size,
            n_channels,
            int(self.num_patches**0.5), 
            self.patch_size, 
            int(self.num_patches**0.5), 
            self.patch_size
            )
        x = x.transpose(3, 4).contiguous()
        x = x.view(
            batch_size,
            n_channels,
            self.num_patches,
            self.patch_size,
            self.patch_size
        )
        x = x.view(
            batch_size,
            n_channels,
            self.num_patches,
            int(self.patch_size**2)
        )
        ids_keep = ids_keep.unsqueeze(1).repeat(1, n_channels, 1, 1)
        x = torch.gather(x, dim=-1, index=ids_keep)
        return x, ids_keep
    
def hyper_embedding(hyper_patch, spectral_patch_embed, hyper_encoder, ids_keep):
    # Spectral embedding
    hyper_data, _ = spectral_patch_embed(hyper_patch) # Random sampling of pixels per patch
    hyper_data = hyper_data.permute(0, 2, 3, 1).contiguous()
    batch_size, n_patches, n_pixels, n_bands = hyper_data.shape
    hyper_data = hyper_data.view(-1, n_bands)
    hyper_data = hyper_encoder(hyper_data) # Feature extraction
    hyper_data = hyper_data[:, 0] # Take [CLS] token
    hyper_data = hyper_data.view(batch_size, n_patches, n_pixels, -1)

    # Pixel-set encoding
    mean_data = torch.mean(hyper_data, dim=2, keepdim=True)
    std_data = torch.std(hyper_data, dim=2, keepdim=True)
    min_data, _ = torch.min(hyper_data, 2, keepdim=True)
    max_data, _ = torch.min(hyper_data, 2, keepdim=True)
    q25_data = torch.quantile(hyper_data, q=0.25, dim=2, keepdim=True)
    q75_data = torch.quantile(hyper_data, q=0.75, dim=2, keepdim=True)
    q10_data = torch.quantile(hyper_data, q=0.1, dim=2, keepdim=True)
    q90_data = torch.quantile(hyper_data, q=0.9, dim=2, keepdim=True)
    # q40_data = torch.quantile(hyper_data, q=0.4, dim=2, keepdim=True)
    # q60_data = torch.quantile(hyper_data, q=0.6, dim=2, keepdim=True)
    hyper_data = torch.cat((
        mean_data,
        std_data,
        min_data,
        max_data,
        q25_data,
        q75_data,
        # q40_data,
        # q60_data,
        q10_data,
        q90_data),
        dim=2)
    hyper_data = hyper_data.view(hyper_data.shape[0], hyper_data.shape[1], -1)
    # Keep only visible patches given ids_keep
    #REVIEW THIS PART!
    if ids_keep != None:
        ids_keep = ids_keep.unsqueeze(-1).repeat(1, 1, hyper_data.shape[-1])
        hyper_data = torch.gather(hyper_data, dim=1, index=ids_keep)
    return hyper_data

# def rgb_embedding(rgb_patch, rgb_encoder, mask_ratio, input_res):
#     img_size = rgb_patch.shape[-1]
#     rgb_data = torch.nn.functional.interpolate(
#         rgb_patch,
#         (2*img_size, 2*img_size), 
#         mode='nearest-exact'
#         )
#     rgb_data = rgb_encoder.patch_embed(rgb_data)
#     pos_embed = get_2d_sincos_pos_embed_with_resolution(
#         rgb_data.shape[-1],
#         int(rgb_data.shape[1]**0.5),
#         input_res,
#         cls_token=True,
#         device=rgb_data.device,
#     )
#     pos_embed_cls, pos_embed = pos_embed[:, :1, :], pos_embed[:, 1:, :]
#     rgb_data = rgb_data + pos_embed
#     rgb_data, mask, ids_restore, ids_keep = rgb_encoder.random_masking(rgb_data, mask_ratio)
    
#     return rgb_data, mask, ids_restore, ids_keep, pos_embed_cls