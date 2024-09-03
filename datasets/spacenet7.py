'''
SpaceNet 7 dataset: https://spacenet.ai/sn7-challenge/
'''

import os
import time
from pathlib import Path
import urllib.request
import urllib.error
import tarfile
import shutil
import gdown

import json
from glob import glob
import rasterio
import numpy as np

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from .utils import DownloadProgressBar
from utils.registry import DATASET_REGISTRY

# train/val/test split from https://doi.org/10.3390/rs15215135
SN7_TRAIN = [
    'L15-0331E-1257N_1327_3160_13',
    'L15-0358E-1220N_1433_3310_13',
    'L15-0457E-1135N_1831_3648_13',
    'L15-0487E-1246N_1950_3207_13',
    'L15-0577E-1243N_2309_3217_13',
    'L15-0586E-1127N_2345_3680_13',
    'L15-0595E-1278N_2383_3079_13',
    'L15-0632E-0892N_2528_4620_13',
    'L15-0683E-1006N_2732_4164_13',
    'L15-0924E-1108N_3699_3757_13',
    'L15-1015E-1062N_4061_3941_13',
    'L15-1138E-1216N_4553_3325_13',
    'L15-1203E-1203N_4815_3378_13',
    'L15-1204E-1202N_4816_3380_13',
    'L15-1209E-1113N_4838_3737_13',
    'L15-1210E-1025N_4840_4088_13',
    'L15-1276E-1107N_5105_3761_13',
    'L15-1298E-1322N_5193_2903_13',
    'L15-1389E-1284N_5557_3054_13',
    'L15-1438E-1134N_5753_3655_13',
    'L15-1439E-1134N_5759_3655_13',
    'L15-1481E-1119N_5927_3715_13',
    'L15-1538E-1163N_6154_3539_13',
    'L15-1615E-1206N_6460_3366_13',
    'L15-1669E-1153N_6678_3579_13',
    'L15-1669E-1160N_6679_3549_13',
    'L15-1672E-1207N_6691_3363_13',
    'L15-1703E-1219N_6813_3313_13',
    'L15-1709E-1112N_6838_3742_13',
    'L15-1716E-1211N_6864_3345_13',
]
SN7_VAL = [
    'L15-0357E-1223N_1429_3296_13',
    'L15-0361E-1300N_1446_2989_13',
    'L15-0368E-1245N_1474_3210_13',
    'L15-0566E-1185N_2265_3451_13',
    'L15-0614E-0946N_2459_4406_13',
    'L15-0760E-0887N_3041_4643_13',
    'L15-1014E-1375N_4056_2688_13',
    'L15-1049E-1370N_4196_2710_13',
    'L15-1185E-0935N_4742_4450_13',
    'L15-1289E-1169N_5156_3514_13',
    'L15-1296E-1198N_5184_3399_13',
    'L15-1615E-1205N_6460_3370_13',
    'L15-1617E-1207N_6468_3360_13',
    'L15-1669E-1160N_6678_3548_13',
    'L15-1748E-1247N_6993_3202_13',
]
SN7_TEST = [
    'L15-0387E-1276N_1549_3087_13',
    'L15-0434E-1218N_1736_3318_13',
    'L15-0506E-1204N_2027_3374_13',
    'L15-0544E-1228N_2176_3279_13',
    'L15-0977E-1187N_3911_3441_13',
    'L15-1025E-1366N_4102_2726_13',
    'L15-1172E-1306N_4688_2967_13',
    'L15-1200E-0847N_4802_4803_13',
    'L15-1204E-1204N_4819_3372_13',
    'L15-1335E-1166N_5342_3524_13',
    'L15-1479E-1101N_5916_3785_13',
    'L15-1690E-1211N_6763_3346_13',
    'L15-1691E-1211N_6764_3347_13',
    'L15-1848E-0793N_7394_5018_13',
]


###############################################################
# SPACENET 7 DATASET                                             #
###############################################################

@DATASET_REGISTRY.register()
class SN7(torch.utils.data.Dataset):
    def __init__(self, cfg, split):

        self.root_path = Path(cfg['root_path'])
        metadata_file = self.root_path / 'metadata_train.json'
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.data_mean = cfg['data_mean']
        self.data_std = cfg['data_std']
        self.classes = cfg['classes']
        self.class_num = len(self.classes)
        self.split = split

        if split == 'train':
            self.aoi_ids = SN7_TRAIN
        elif split == 'val':
            self.aoi_ids = SN7_VAL
        elif split == 'test':
            self.aoi_ids = SN7_TEST
        else:
            raise Exception('Invalid split')

        self.items = []
        # adding timestamps (only if label exists and not masked) for each AOI
        for aoi_id in self.aoi_ids:
            timestamps = list(self.metadata[aoi_id])
            timestamps = [t for t in timestamps if not t['mask'] and t['label']]
            self.items.extend(timestamps)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        item = self.items[index]
        aoi_id, year, month = item['aoi_id'], int(item['year']), int(item['month'])

        image = self.load_planet_mosaic(aoi_id, year, month)
        target = self.load_building_label(aoi_id, year, month)

        image = torch.from_numpy(image)
        target = torch.from_numpy(target)

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {}
        }

        return output

    def load_planet_mosaic(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'images_masked'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        with rasterio.open(str(file), mode='r') as src:
            img = src.read(out_shape=(1024, 1024), resampling=rasterio.enums.Resampling.nearest)
        # 4th band (last oen) is alpha band
        img = img[:-1]
        return img.astype(np.float32)

    def load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        with rasterio.open(str(file), mode='r') as src:
            label = src.read(out_shape=(1024, 1024), resampling=rasterio.enums.Resampling.nearest)
        label = (label > 0).squeeze()
        return label.astype(np.int64)

    def load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self.load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self.load_building_label(aoi_id, year_t2, month_t2)
        change = np.logical_and(building_t1 == 0, building_t2 == 1)
        return change.astype(np.float32)

    @staticmethod
    def get_band(path):
        return int(path.split('_')[-2])

    @staticmethod
    def download(dataset_config: dict, silent=False):
        output_path = Path(dataset_config["root_path"])

        if not output_path.exists():
            output_path.mkdir()
        else:
            if not silent:
                print("SpaceNet 7 Dataset folder exists, skipping downloading dataset.")
            return

        # download from Google Drive
        url = dataset_config["download_url"]
        tar_file = output_path / f'spacenet7.tar.gz'
        gdown.download(url, str(tar_file), quiet=False)

        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                print(f"Extracting to {output_path} ...")
                tar.extractall(path=output_path)
                print(f"Successfully extracted {tar_file} to {output_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

        temp_dataset_path = output_path / 'spacenet7'
        for item in temp_dataset_path.iterdir():
            destination = output_path / item.name
            try:
                shutil.move(str(item), str(destination))
                print(f"Moved {item} to {destination}")
            except Exception as e:
                print(f"Failed to move {item}: {e}")
        tar_file.unlink()
        temp_dataset_path.unlink()

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = SN7(cfg=dataset_config, split='train')
        dataset_val = SN7(cfg=dataset_config, split='val')
        dataset_test = SN7(cfg=dataset_config, split='test')
        return dataset_train, dataset_val, dataset_test
