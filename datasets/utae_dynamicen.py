import os
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime
import torchvision.transforms.functional as TF
import cv2

import random
from PIL import Image

from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class DynamicEarthNet(Dataset):
    #def __init__(self, root, mode, type, reference_date="2018-01-01", crop_size=512, num_classes=6, ignore_index=-1):
    def __init__(self, cfg, split, is_train=True):
        """
        Args:
            root: the root of the folder which contains planet imagery and labels
            mode: train/val/test -- selects the splits
            type: single/weekly/daily -- selects the time-period you want to use
            reference_date: for positional encoding defaults:2018-01-01
            crop_size: crop size default:1024x1024
            num_classes: for DynamicEarthNet numclasses: 6
            ignore_index: default:-1
        """
        self.root_path = cfg['root_path']
        self.data_mean = cfg['data_mean']
        self.data_std = cfg['data_std']
        self.classes = cfg['classes']
        self.ignore_index = cfg['ignore_index']
        self.class_num = len(self.classes)
        self.split = split
        self.is_train = is_train

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
                with tifffile.TiffFile.open(os.path.join(self.root_path, self.all_days[index][i][0][1:])) as img:
                    red = img.pages[2].asarray()
                    green = img.pages[1].asarray()
                    blue = img.pages[0].asarray()
                    nir = img.pages[3].asarray()
                image = np.dstack((red, green, blue, nir))
                cur_images.append(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)) # np.array already\
                cur_dates.append(self.all_days[index][i][1])

            image_stack = np.concatenate(cur_images, axis=0)
            dates = torch.from_numpy(np.array(cur_dates, dtype=np.int32))
            label = tifffile.imread(os.path.join(self.root_path, self.labels[index][1:]))
            mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int32)

            for i in range(self.class_num + 1):
                if i == 6:
                    mask[label[i, :, :] == 255] = -1
                else:
                    mask[label[i, :, :] == 255] = i

            return (image_stack, dates), mask

        else:
            for i in range(len(self.dates)):
                # read .tif
                with tifffile.TiffFile.open(os.path.join(self.root_path, self.planet_day[index][i][1:])) as img:
                    red = img.pages[2].asarray()
                    green = img.pages[1].asarray()
                    blue = img.pages[0].asarray()
                    nir = img.pages[3].asarray()
                image = np.dstack((red, green, blue, nir))
                cur_images.append(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0))   # np.array already\
            image_stack = np.concatenate(cur_images, axis=0)
            dates = torch.from_numpy(np.array(self.planet_day[index][len(self.dates):], dtype=np.int32))
            label = tifffile.imread(os.path.join(self.root_path, self.labels[index][1:]))
            mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int32)

            for i in range(self.class_num + 1):
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

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = DynamicEarthNet(cfg=dataset_config, split="train")
        dataset_val = DynamicEarthNet(cfg=dataset_config, split="val")
        dataset_test = DynamicEarthNet(cfg=dataset_config, split="test")
        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def download(dataset_config: dict, silent=False):
        pass
