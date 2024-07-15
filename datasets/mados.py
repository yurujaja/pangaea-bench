# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/gkakogeorgiou/mados
'''

import os
import time
import pathlib
import urllib.request
import urllib.error
import zipfile
import tqdm
from glob import glob
import rasterio
import numpy as np
from osgeo import gdal
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from .utils import DownloadProgressBar
from utils.registry import DATASET_REGISTRY
import matplotlib.pyplot as plt


###############################################################
# MADOS DATASET                                               #
###############################################################

@DATASET_REGISTRY.register()
class MADOS(torch.utils.data.Dataset):
    def __init__(self, cfg, split, is_train=True):

        self.root_path = cfg['root_path']
        self.data_mean = cfg['data_mean']
        self.data_std = cfg['data_std']
        self.classes = cfg['classes']
        self.class_num = len(self.classes)
        self.split = split
        self.is_train = is_train

        self.ROIs_split = np.genfromtxt(os.path.join(self.root_path, 'splits', f'{split}_X.txt'), dtype='str')

        self.image_list = []
        self.target_list = []

        self.tiles = sorted(glob(os.path.join(self.root_path, '*')))

        for tile in self.tiles:
            splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]

            for crop in splits:
                crop_name = os.path.basename(tile) + '_' + crop.split('.tif')[0]

                if crop_name in self.ROIs_split:
                    all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_' + crop))
                    all_bands = sorted(all_bands, key=self.get_band)
                    # all_bands = np.array(all_bands)

                    self.image_list.append(all_bands)

                    cl_path = os.path.join(tile, '10', os.path.basename(tile) + '_L2R_cl_' + crop)
                    self.target_list.append(cl_path)

        self.transform = T.Compose([
            #T.Resize((self.height, self.height), antialias=False),
            T.Normalize(mean=self.data_mean['optical'], std=self.data_std['optical'])
        ])
        self.normalize = T.Normalize(mean=self.data_mean['optical'], std=self.data_std['optical'])

    def __len__(self):
        return len(self.image_list)

    def getnames(self):
        return self.ROIs_split

    def __getitem__(self, index):

        all_bands = self.image_list[index]
        current_image = []
        for c, band in enumerate(all_bands):
            upscale_factor = int(os.path.basename(os.path.dirname(band))) // 10
            with rasterio.open(band, mode='r') as src:
                this_band = src.read(1,
                                     out_shape=(int(src.height * upscale_factor), int(src.width * upscale_factor)),
                                     resampling=rasterio.enums.Resampling.nearest
                                     )
                this_band = torch.from_numpy(this_band)
                #this_band[torch.isnan(this_band)] = self.data_mean['optical'][c]
                current_image.append(this_band)

        image = torch.stack(current_image)
        invalid_mask = torch.isnan(image)
        image = self.transform(image)
        image[invalid_mask] = 0


        with rasterio.open(self.target_list[index], mode='r') as src:
            target = src.read(1)
        target = torch.from_numpy(target.astype(np.int64))
        target = target - 1

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {}
        }

        return output

    @staticmethod
    def get_band(path):
        return int(path.split('_')[-2])

    @staticmethod
    def download(dataset_config: dict, silent=False):
        output_path = pathlib.Path(dataset_config["root_path"])
        url = dataset_config["download_url"]

        existing_dirs = list(output_path.glob("Scene_*"))
        if existing_dirs:
            if not silent:
                print("MADOS Dataset folder exists, skipping downloading dataset.")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        temp_file_name = f"temp_{hex(int(time.time()))}_MADOS.zip"
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

        with zipfile.ZipFile(output_path / temp_file_name, 'r') as zip_ref:
            print(f"Extracting to {output_path} ...")
            # Remove top-level dir in ZIP file for nicer data dir structure
            members = []
            for zipinfo in zip_ref.infolist():
                new_path = os.path.join(*(zipinfo.filename.split(os.path.sep)[1:]))
                zipinfo.filename = str(new_path)
                members.append(zipinfo)

            zip_ref.extractall(output_path, members)
            print("done.")

        (output_path / temp_file_name).unlink()

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = MADOS(cfg=dataset_config, split="train", is_train=True)
        dataset_val = MADOS(cfg=dataset_config, split="val", is_train=False)
        dataset_test = MADOS(cfg=dataset_config, split="test", is_train=False)
        return dataset_train, dataset_val, dataset_test
