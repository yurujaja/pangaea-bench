import os
import time
import torch
import numpy as np
import rasterio
from typing import Sequence, Dict, Any, Union, Literal, Tuple
from sklearn.model_selection import train_test_split
from glob import glob

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import pathlib
import urllib
import tarfile

# from utils.registry import DATASET_REGISTRY
from geofm_bench.datasets.utils import DownloadProgressBar
from geofm_bench.datasets.base import GeoFMDataset

# @DATASET_REGISTRY.register()
class HLSBurnScars(GeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
    ):
        
        """Initialize the HLSBurnScars dataset.

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image. 
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality. 
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
        """
        
        super(HLSBurnScars, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.root_path = root_path
        self.classes = classes
        self.split = split
        
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download

        # ISSUE
        self.split_mapping = {'train': 'training', 'val': 'validation', 'test': 'validation'}

        self.image_list = sorted(glob(os.path.join(self.root_path, self.split_mapping[self.split], '*merged.tif')))
        self.target_list = sorted(glob(os.path.join(self.root_path, self.split_mapping[self.split], '*mask.tif')))

        # if self.split != "test":
        #     train_val_idcs = self.get_stratified_train_val_split(all_files)
        #     all_files = [all_files[i] for i in train_val_idcs[self.split]]
    
    # @staticmethod
    # def get_stratified_train_val_split(all_files, split) -> Tuple[Sequence[int], Sequence[int]]:

    #    # Fixed stratified sample to split data into train/val. 
    #    # This keeps 90% of datapoints belonging to an individual event in the training set and puts the remaining 10% in the validation set. 
    #     disaster_names = list(
    #         map(lambda path: pathlib.Path(path).name.split("_")[0], all_files))
    #     train_idxs, val_idxs = train_test_split(np.arange(len(all_files)),
    #                                             test_size=0.1,
    #                                             random_state=23,
    #                                             stratify=disaster_names)
    #     if split == "train":
    #         return train_idxs
    #     else:
    #         return val_idxs
        
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
    def get_stratified_train_val_split(all_files) -> Tuple[Sequence[int], Sequence[int]]:

       # Fixed stratified sample to split data into train/val. 
       # This keeps 90% of datapoints belonging to an individual event in the training set and puts the remaining 10% in the validation set. 
        disaster_names = list(
            map(lambda path: pathlib.Path(path).name.split("_")[0], all_files))
        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)),
                                                test_size=0.1,
                                                random_state=23,
                                                stratify=disaster_names)
        return {"train": train_idxs, "val": val_idxs}
    
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


