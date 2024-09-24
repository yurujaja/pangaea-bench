# Sources: 
# - https://github.com/PaulBorneP/Xview2_Strong_Baseline/blob/master/datasets/base_dataset.py
# - https://github.com/PaulBorneP/Xview2_Strong_Baseline/blob/master/datasets/supervised_dataset.py

from typing import Sequence, Dict, Any, Union, Literal, Tuple
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pathlib
from sklearn.model_selection import train_test_split
import cv2
import urllib
import tarfile

from geofm_bench.datasets.utils import DownloadProgressBar
from geofm_bench.datasets.base import GeoFMDataset

# @DATASET_REGISTRY.register()
class xView2(GeoFMDataset):
    # Note that this dataset must be downloaded from the competition website, which you will need to sign up for.
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
        """Initialize the xView2 dataset.

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
        super(xView2, self).__init__(
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
        self.split = split
        self.bands = bands
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

        self.all_files = self.get_all_files()
        

    def get_all_files(self) -> Sequence[str]:
        all_files = []
        if self.split == "test":
            data_dirs = [os.path.join(self.root_path, "test")]
        else:
            # Train and val consist of the smaller train and the larger tier3 set.
            data_dirs = [os.path.join(self.root_path, d) for d in ["train", "tier3"]]

        for d in data_dirs:
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
                if '_pre_disaster.png' in f:
                    all_files.append(os.path.join(d, 'images', f))
        
        if self.split != "test":
            train_val_idcs = self.get_stratified_train_val_split(all_files)
            all_files = [all_files[i] for i in train_val_idcs[self.split]]

        return all_files

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

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor,  Any, str]]:
        
        fn = self.all_files[idx]

        img_pre = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_post = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)


        #msk_pre = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        msk_post = cv2.imread(fn.replace('/images/', '/masks/').replace(
            '_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        
        #msk = np.stack([msk_pre, msk_post], axis=0)
        msk = msk_post
        img = np.stack([img_pre, img_post], axis=0) 

        # Reshaping tensors from (T, H, W, C) to (C, T, H, W)
        img = torch.from_numpy(img.transpose((3, 0, 1, 2))).float()
        # img_pre = torch.from_numpy(img_pre.transpose((2, 0, 1))).float()
        # img_post = torch.from_numpy(img_post.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk).float()


        return {
            'image': {
                    'optical': img,
                    },
            'target': msk,  
            'metadata': {"filename":fn}
        }

        # return {
        #     'image': {
        #         't0' : {
        #             'optical': img_pre,
        #             },
        #         't1': {
        #             'optical': img_post,
        #             },
        #     },
        #     'target': msk,  
        #     'metadata': {"filename":fn}
        # }

    # @staticmethod
    # def get_splits(dataset_config):
    #     dataset_train = xView2(cfg=dataset_config, split="train")
    #     dataset_val = xView2(cfg=dataset_config, split="val")
    #     dataset_test = xView2(cfg=dataset_config, split="test")
    #     return dataset_train, dataset_val, dataset_test


    @staticmethod
    def download(dataset_config:dict, silent=False):
        output_path = pathlib.Path(dataset_config["root_path"])
        url = dataset_config["download_url"]

        try:
            os.makedirs(output_path, exist_ok=False)
        except FileExistsError:
            if not silent:
                print("xView2 dataset folder exists, skipping downloading dataset.")
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