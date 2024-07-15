# Obtained from: https://github.com/synativ/RSFMs/blob/main/src/rsfms/datamodules/sen1floods11.py

import glob
import os
import geopandas
import numpy as np
import pandas as pd
import rasterio

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from .utils import download_bucket_concurrently
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Sen1Floods11(torch.utils.data.Dataset):
    """NonGeo dataset implementation for fire scars."""

    def __init__(self, cfg, split, is_train=True) -> None:
        super().__init__()

        self.root_path = cfg['root_path']
        self.data_mean = cfg['data_mean']
        self.data_std = cfg['data_std']
        self.classes = cfg['classes']
        self.class_num = len(self.classes)
        self.split = split
        self.is_train = is_train

        self.split_mapping = {'train': 'train', 'val': 'valid', 'test': 'test'}


        split_file = os.path.join(self.root_path, f"v1.1/splits/flood_handlabeled/flood_{self.split_mapping[split]}_data.csv")
        metadata_file = os.path.join(self.root_path, "v1.1/Sen1Floods11_Metadata.geojson")
        data_root = os.path.join(self.root_path, "v1.1/data/flood_events/HandLabeled/")

        self.metadata = geopandas.read_file(metadata_file)

        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        self.image_list = [os.path.join(data_root,  'S2Hand', f[0].replace('S1Hand', 'S2Hand')) for f in file_list]
        self.target_list = [os.path.join(data_root, 'LabelHand', f[1]) for f in file_list]

        self.transform = T.Compose([
            #T.Resize((self.height, self.height), antialias=False),
            T.Normalize(mean=self.data_mean['optical'], std=self.data_std['optical'])
        ])

    def __len__(self):
        return len(self.image_list)

    def _get_date(self, index):
        # move this logic to the model?
        file_name = self.image_list[index]
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
        #image = self._load_file(self.image_list[index])
        #target = self._load_file(self.target_list[index])
        with rasterio.open(self.image_list[index]) as src:
            image = src.read()
        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)
        timestamp = self._get_date(index)#.astype(np.float32)

        image = torch.from_numpy(image).float()
        image = self.transform(image)

        target = torch.from_numpy(target)


        #print(image.shape, image.dtype, image.mean(), image.max(), image.min())
        #print(target.shape, target.dtype, target.mean(), target.max(), target.min())
        #print(timestamp)
        #print(np.unique(target))

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {
                "timestamp": timestamp,
            }
        }
        return output

    def _load_file(self, path):
        data = rioxarray.open_rasterio(path)
        return data.to_numpy()


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
