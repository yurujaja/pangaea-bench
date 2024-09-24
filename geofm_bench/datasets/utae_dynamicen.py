import os
import numpy as np
import rasterio
import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
from datetime import datetime
# import torchvision.transforms.functional as TF
# import cv2

# import random
# from PIL import Image

from geofm_bench.datasets.base import GeoFMDataset

# from utils.registry import DATASET_REGISTRY

# @DATASET_REGISTRY.register()
class DynamicEarthNet(GeoFMDataset):
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
        """Initialize the DynamicEarthNet dataset.
        Link: https://github.com/aysim/dynnet

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
        super(DynamicEarthNet, self).__init__(
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
        self.ignore_index = ignore_index
        self.split = split
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.download_url = download_url
        self.auto_download = auto_download

        self.mode = 'weekly'

        self.files = []

        reference_date = "2018-01-01"
        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.set_files()

    def set_files(self):
        self.file_list = os.path.join(self.root_path, "dynnet_training_splits", f"{self.split}" + ".txt")

        file_list = [line.rstrip().split(' ') for line in tuple(open(self.file_list, "r"))]
        #for
        self.files, self.labels, self.year_months = list(zip(*file_list))
        self.files = [f.replace('/reprocess-cropped/UTM-24000/', '/planet/') for f in self.files]

        if self.mode == 'daily':
            self.all_days = list(range(len(self.files)))

            for i in range(len(self.files)):
                self.planet, self.day = [], []
                date_count = 0
                for _, _, infiles in os.walk(os.path.join(self.root_path, self.files[i][1:])):
                    for infile in sorted(infiles):
                        if infile.startswith(self.year_months[i]):
                            self.planet.append(os.path.join(self.files[i], infile))
                            self.day.append((datetime(int(str(infile.split('.')[0])[:4]), int(str(infile.split('.')[0][5:7])),
                                                  int(str(infile.split('.')[0])[8:])) - self.reference_date).days)
                            date_count += 1
                self.all_days[i] = list(zip(self.planet, self.day))
                self.all_days[i].insert(0, date_count)

        else:
            self.planet, self.day = [], []
            if self.mode == 'weekly':
                self.dates = ['01', '05', '10', '15', '20', '25']
            elif self.mode == 'single':
                self.dates = ['01']

            for i, year_month in enumerate(self.year_months):
                for date in self.dates:
                    curr_date = year_month + '-' + date
                    self.planet.append(os.path.join(self.files[i], curr_date + '.tif'))
                    self.day.append((datetime(int(str(curr_date)[:4]), int(str(curr_date[5:7])),
                                                  int(str(curr_date)[8:])) - self.reference_date).days)
            self.planet_day = list(zip(*[iter(self.planet)] * len(self.dates), *[iter(self.day)] * len(self.dates)))


    def load_data(self, index):
        cur_images, cur_dates = [], []
        if self.mode == 'daily':
            for i in range(1, self.all_days[index][0]+1):
                img = rasterio.open(os.path.join(self.root_path, self.all_days[index][i][0][1:]))
                red = img.read(3)
                green = img.read(2)
                blue = img.read(1)
                nir = img.read(4)
                image = np.dstack((red, green, blue, nir))
                cur_images.append(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)) # np.array already\
                cur_dates.append(self.all_days[index][i][1])

            image_stack = np.concatenate(cur_images, axis=0)
            dates = torch.from_numpy(np.array(cur_dates, dtype=np.int32))
            label = rasterio.open(os.path.join(self.root_path, self.labels[index][1:]))
            label = label.read()
            mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int32)

            for i in range(self.num_classes + 1):
                if i == 6:
                    mask[label[i, :, :] == 255] = -1
                else:
                    mask[label[i, :, :] == 255] = i

            return (image_stack, dates), mask

        else:
            for i in range(len(self.dates)):
                # read .tif
                img = rasterio.open(os.path.join(self.root_path, self.planet_day[index][i][1:]))
                red = img.read(3)
                green = img.read(2)
                blue = img.read(1)
                nir = img.read(4)
                image = np.dstack((red, green, blue, nir))
                cur_images.append(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0))   # np.array already\
            image_stack = np.concatenate(cur_images, axis=0)
            dates = torch.from_numpy(np.array(self.planet_day[index][len(self.dates):], dtype=np.int32))
            label = rasterio.open(os.path.join(self.root_path, self.labels[index][1:]))
            label = label.read()
            mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int32)

            for i in range(self.num_classes + 1):
                if i == 6:
                    mask[label[i, :, :] == 255] = -1
                else:
                    mask[label[i, :, :] == 255] = i

            return (image_stack, dates), mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        (images, dates), label = self.load_data(index)

        images = torch.from_numpy(images).permute(3, 0, 1, 2)#.transpose(0, 1)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()

        output = {
            'image': {
                'optical': images,
            },
            'target': label,
            'metadata': {}
        }

        return output
        #return {'img': images, 'label': label, 'meta': dates}

    # @staticmethod
    # def get_splits(dataset_config):
    #     dataset_train = DynamicEarthNet(cfg=dataset_config, split="train")
    #     dataset_val = DynamicEarthNet(cfg=dataset_config, split="val")
    #     dataset_test = DynamicEarthNet(cfg=dataset_config, split="test")
    #     return dataset_train, dataset_val, dataset_test

    @staticmethod
    def download(dataset_config: dict, silent=False):
        pass
