import os
import time
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset

import pathlib
import urllib
import tarfile
from .utils import DownloadProgressBar

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
                'optical': normalized_image,
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
    
    @staticmethod
    def get_splits(dataset_config):
        dataset_train = BurnScarsDataset(data_root=dataset_config["data_path"], split="training")
        dataset_val = BurnScarsDataset(data_root=dataset_config["data_path"], split="validation")
        dataset_test = dataset_val
        return dataset_train, dataset_val, dataset_test
    
    @staticmethod
    def download(dataset_config:dict, silent=False):
        output_path = pathlib.Path(dataset_config["data_path"])
        url = dataset_config["download_url"]

        try:
            os.makedirs(output_path, exist_ok=False)
        except FileExistsError:
            if not silent:
                print("HLSBurnScars dataset folder exists, skipping downloading dataset.")
            return

        temp_file_name = f"temp_{hex(int(time.time()))}_hls_burn_scars.tar.gz"
        pbar = DownloadProgressBar()

        try:
            urllib.request.urlretrieve(url, output_path / temp_file_name, pbar)
        except urllib.error.HTTPError as e:
            print('Error while downloading dataset: The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
            return
        except urllib.error.URLError as e:
            print('Error while downloading dataset: Failed to reach a server.')
            print('Reason: ', e.reason)
            return

        with tarfile.open(output_path / temp_file_name, 'r') as tar:
            print(f"Extracting to {output_path} ...")
            tar.extractall(output_path)
            print("done.")

        os.remove(output_path / temp_file_name)


