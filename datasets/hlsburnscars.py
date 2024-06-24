import os
import torch
import numpy as np
import rasterio
import yaml
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

class BurnScarsDataset(Dataset):
    BAND_STATS = {
        'B2': {'mean': 0.033349706741586264, 'std': 0.02269135568823774},
        'B3': {'mean': 0.05701185520536176, 'std': 0.026807560223070237},
        'B4': {'mean': 0.05889748132001316, 'std': 0.04004109844362779},
        'B8a': {'mean': 0.2323245113436119, 'std': 0.07791732423672691},
        'B11': {'mean': 0.1972854853760658, 'std': 0.08708738838140137},
        'B12': {'mean': 0.11944914225186566, 'std': 0.07241979477437814}
    }

    BAND_INDICES = {
        'B2': 1,   # Blue
        'B3': 2,   # Green
        'B4': 3,   # Red
        'B8a': 4,  # NIR
        'B11': 5,  # SWIR 1
        'B12': 6   # SWIR 2
    }

    def __init__(self, data_root, split, crop=(224, 224), transform=None):
        self.root_dir = os.path.join(data_root, split)
        self.image_files = self._load_image_files()
        self.bands = ["B2", "B3", "B4", "B8a","B11", "B12"]
        self.crop = crop
        self.transform = transform

    def _load_image_files(self):
        image_files = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('_merged.tif'):
                    image_files.append(os.path.join(dirpath, filename))
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = image_path.replace('_merged.tif', '.mask.tif')
        
        image = self.load_raster(image_path)
        mask = self.load_mask(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        normalized_image = self.preprocess_image(image)

        output = {
            'image': {
                's2': normalized_image,
            },
            'target': torch.tensor(mask, dtype=torch.float32),  
            'metadata': {}
        }
        
        return output


    def load_raster(self, path):
        with rasterio.open(path) as src:
            img = src.read()

            # Load specified bands
            bands_data = [img[self.BAND_INDICES[band] - 1] for band in self.bands]

            img = np.stack(bands_data)

            img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
            if self.crop:
                img = img[:, -self.crop[0]:, -self.crop[1]:]
        return img

    def load_mask(self, path):
        with rasterio.open(path) as src:
            mask = src.read(1)
            if self.crop:
                mask = mask[-self.crop[0]:, -self.crop[1]:]
        return mask

    def preprocess_image(self, image):
        means = np.array([self.BAND_STATS[band]['mean'] for band in self.bands]).reshape(-1, 1, 1)
        stds = np.array([self.BAND_STATS[band]['std'] for band in self.bands]).reshape(-1, 1, 1)
        
        # Normalize image
        normalized = ((image - means) / stds)
        normalized = torch.from_numpy(normalized).to(torch.float32)
        return normalized

    # def clip(self, band):
    #     lower_percentile = np.percentile(band, 2)
    #     upper_percentile = np.percentile(band, 98)
    #     return np.clip(band, lower_percentile, upper_percentile)

# def enhance_raster_for_visualization(raster, ref_img=None):
#     if ref_img is None:
#         ref_img = raster
#     channels = []
#     for channel in range(raster.shape[0]):
#         valid_mask = np.ones_like(ref_img[channel], dtype=bool)
#         valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
#         mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
#         normalized_raster = (raster[channel] - mins) / (maxs - mins)
#         normalized_raster[~valid_mask] = 0
#         clipped = np.clip(normalized_raster, 0, 1)
#         channels.append(clipped)
#     clipped = np.stack(channels)
#     channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
#     rgb = channels_last[..., ::-1]
#     return rgb


