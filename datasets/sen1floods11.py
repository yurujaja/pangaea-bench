# Source: https://github.com/cloudtostreet/Sen1Floods11

import os
import geopandas
import numpy as np
import pandas as pd
import rasterio 
import torch

from .utils import download_bucket_concurrently
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Sen1Floods11(torch.utils.data.Dataset):

    def __init__(self, cfg, split) -> None:
        super().__init__()

        self.root_path = cfg['root_path']
        self.data_mean = cfg['data_mean']
        self.data_std = cfg['data_std']
        self.classes = cfg['classes']
        self.distribution = cfg['distribution']
        self.class_num = len(self.classes)
        self.split = split
        
        self.split_mapping = {'train': 'train', 'val': 'valid', 'test': 'test'}

        split_file = os.path.join(self.root_path, "v1.1", f"splits/flood_handlabeled/flood_{self.split_mapping[split]}_data.csv")
        metadata_file = os.path.join(self.root_path, "v1.1", "Sen1Floods11_Metadata.geojson")
        data_root = os.path.join(self.root_path, "v1.1", "data/flood_events/HandLabeled/")

        self.metadata = geopandas.read_file(metadata_file)

        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        self.s1_image_list = [os.path.join(data_root,  'S1Hand', f[0]) for f in file_list]
        self.s2_image_list = [os.path.join(data_root,  'S2Hand', f[0].replace('S1Hand', 'S2Hand')) for f in file_list]
        self.target_list = [os.path.join(data_root, 'LabelHand', f[1]) for f in file_list]


    def __len__(self):
        return len(self.s1_image_list)

    def _get_date(self, index):
        file_name = self.s2_image_list[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            date = pd.to_datetime("13-10-1998", dayfirst=True)
        else:
            date = pd.to_datetime(self.metadata[self.metadata["location"] == location]["s2_date"].item())
        date_np = np.zeros((1, 3))
        date_np[0, 0] = date.year
        date_np[0, 1] = date.dayofyear - 1  # base 0
        date_np[0, 2] = date.hour
        return date_np

    def __getitem__(self, index):
        with rasterio.open(self.s2_image_list[index]) as src:
            s2_image = src.read()

        with rasterio.open(self.s1_image_list[index]) as src:
            s1_image = src.read()
            # Convert the missing values (clouds etc.)
            s1_image = np.nan_to_num(s1_image)

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)
        
        timestamp = self._get_date(index)

        s2_image = torch.from_numpy(s2_image).float()
        s1_image = torch.from_numpy(s1_image).float()   
        target = torch.from_numpy(target)

        weight = torch.zeros_like(target).float()
        for i, freq in enumerate(self.distribution):
            weight[target == i] = 1 - freq
        weight[target == -1] = 1e-6

        output = {
            'image': {
                'optical': s2_image,
                'sar' : s1_image,
            },
            'target': target,
            'weight': weight,
            'metadata': {
                "timestamp": timestamp,
            }
        }
        return output

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = Sen1Floods11(dataset_config, split="train")
        dataset_val = Sen1Floods11(dataset_config, split="val")
        dataset_test = Sen1Floods11(dataset_config, split="test")
        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def download(dataset_config: dict, silent=False):
        if os.path.exists(dataset_config["root_path"]):
            if not silent:
                print("Sen1Floods11 Dataset folder exists, skipping downloading dataset.")
            return
        download_bucket_concurrently(dataset_config["gcs_bucket"], dataset_config["root_path"])


