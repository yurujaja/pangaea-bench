import os
import time
import torch
import numpy as np
import rasterio
from glob import glob

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import pathlib
import urllib
import tarfile
from .utils import DownloadProgressBar
from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class HLSBurnScars(torch.utils.data.Dataset):
    def __init__(self, cfg, split, is_train=True):

        self.root_path = cfg['root_path']
        self.data_mean = cfg['data_mean']
        self.data_std = cfg['data_std']
        self.classes = cfg['classes']
        self.class_num = len(self.classes)
        self.split = split
        self.is_train = is_train

        self.split_mapping = {'train': 'training', 'val': 'validation', 'test': 'validation'}

        self.image_list = sorted(glob(os.path.join(self.root_path, self.split_mapping[self.split], '*merged.tif')))
        self.target_list = sorted(glob(os.path.join(self.root_path, self.split_mapping[self.split], '*mask.tif')))

        self.transform = T.Compose([
            # T.Resize((self.height, self.height), antialias=False),
            T.Normalize(mean=self.data_mean['optical'], std=self.data_std['optical'])
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        with rasterio.open(self.image_list[index]) as src:
            image = src.read()
        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)

        image = torch.from_numpy(image)
        target = torch.from_numpy(target.astype(np.int64))

        invalid_mask = image == 9999
        image = self.transform(image)
        image[invalid_mask] = 0


        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {}
        }
        
        return output

    
    @staticmethod
    def get_splits(dataset_config):
        dataset_train = HLSBurnScars(dataset_config, split="train", is_train=True)
        dataset_val = HLSBurnScars(dataset_config, split="val", is_train=False)
        dataset_test = dataset_val
        return dataset_train, dataset_val, dataset_test
    
    @staticmethod
    def download(dataset_config:dict, silent=False):
        output_path = pathlib.Path(dataset_config["root_path"])
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


