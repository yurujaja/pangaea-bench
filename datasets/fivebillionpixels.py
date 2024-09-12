import os
import time
import torch
import numpy as np
import rasterio
import random
from glob import glob

from PIL import Image
import tifffile as tiff
import cv2

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import pathlib
import urllib
import tarfile
from .utils import DownloadProgressBar
from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class FiveBillionPixels(torch.utils.data.Dataset):
    """
    FiveBillionPixels
    """
    NUM_CLASSES = 24+1

    def __init__(self, cfg, split, is_train = True):
        """
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = cfg['root_path']
        # print(os.path.join(self._base_dir, split, 'imgs', '*.tif'))
        # print(os.path.join(self._base_dir, split, 'labels', '*.tif'))
        # print(self._image_dir)
        # print(self._label_dir)
        # _splits_dir = os.path.join(self._base_dir, 'list')
        # self.split = [split]

        # self.args = args

        # self.im_ids = []
        # self.images = []
        # self.labels = []

        # for splt in self.split:
        #     with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
        #         lines = f.read().splitlines()

        #     if splt == 'train':
        #         lines = random.sample(lines, len(os.listdir(os.path.join(args.target_dir, args.target))))
        #     elif split == 'val':
        #         lines = random.sample(lines, 500)
        # self.root_path = cfg['root_path']
        self.data_mean = cfg['data_mean']
        self.data_std = cfg['data_std']
        self.classes = cfg['classes']
        self.class_num = len(self.classes)
        self.split = split
        self.is_train = is_train

        self._image_dir = sorted(glob(os.path.join(self._base_dir, self.split, 'imgs', '*.tif')))
        self._label_dir = sorted(glob(os.path.join(self._base_dir, self.split, 'labels', '*.tif')))
        # print(split)
        # print(os.path.join(self._base_dir, self.split, 'imgs', '*.tif'))
        # print(os.path.join(self._base_dir, self.split, 'labels', '*.png'))
        # print(self._image_dir)
        # print((self._label_dir))
        # print(len(self._image_dir))
        # print(len(self._label_dir))

        # self.split_mapping = {'train': 'training', 'val': 'validation', 'test': 'validation'}

        # self.image_list = sorted(glob(os.path.join(self.root_path, self.split_mapping[self.split], '*merged.tif')))
        # self.target_list = sorted(glob(os.path.join(self.root_path, self.split_mapping[self.split], '*mask.tif')))


        # for ii, line in enumerate(lines):
        #     _image = os.path.join(self._image_dir, line + ".tif")
        #     _label = os.path.join(self._label_dir, line + ".png")
        #     assert os.path.isfile(_image)
        #     assert os.path.isfile(_label)
        #     self.im_ids.append(line)
        #     self.images.append(_image)
        #     self.labels.append(_label)

        # assert (len(self.images) == len(self.labels))

        # Display stats
        # print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self._image_dir)

    def __getitem__(self, index):
        # _img, _target = self._make_img_gt_point_pair(index)
        # print(index)
        # image = Image.open(self._image_dir[index]).convert('CMYK') #check it also on the normalization
        # target = Image.open(self._label_dir[index])

        image = tiff.imread(self._image_dir[index])#.convert('CMYK') #check it also on the normalization
        target = tiff.imread(self._label_dir[index]) #, cv2.IMREAD_UNCHANGED)

        # image = TF.pil_to_tensor(image)
        # target = TF.pil_to_tensor(target).squeeze(0).to(torch.int64)

        image = image.astype(np.float32)  # Convert to float32
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)

        # Tile the image and target to the fixed size specified in the config
        # image, target = self.tile_image_and_mask(image, target, self.img_size)

        image = torch.from_numpy(image).permute(2, 0, 1)
        target = torch.from_numpy(target).long()
        # print(image.shape)
        # print(target.shape)

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {}
        }
        
        return output

        # for split in self.split:
        #     if split == "train":
        #         return self.transform_tr(sample)
        #     elif split == 'val':
        #         return self.transform_val(sample)

    # def _make_img_gt_point_pair(self, index):
    #     _img = Image.open(self.images[index]).convert('CMYK')
    #     _target = Image.open(self.labels[index])

    #     return _img, _target

    # @staticmethod
    # def transform_tr(sample):
    #     composed_transforms = transforms.Compose([
    #         tr.RandomHorizontalFlip(),
    #         tr.RandomGaussianBlur(),
    #         tr.Normalize(mean=(0.506, 0.371, 0.390, 0.363),
    #                      std=(0.254, 0.244, 0.231, 0.231)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)

    # @staticmethod
    # def transform_val(sample):
    #     composed_transforms = transforms.Compose([
    #         tr.Normalize(mean=(0.506, 0.371, 0.390, 0.363),
    #                      std=(0.254, 0.244, 0.231, 0.231)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)

    # def __str__(self):
    #     return 'GID24 (split = ' + self.split[0] + ')'

    
    @staticmethod
    def get_splits(dataset_config):
        dataset_train = FiveBillionPixels(dataset_config, split="train", is_train=True)
        dataset_val = FiveBillionPixels(dataset_config, split="val", is_train=False)
        dataset_test = dataset_val
        return dataset_train, dataset_val, dataset_test
    
    @staticmethod
    def download(dataset_config:dict, silent=False):
        pass
        # output_path = pathlib.Path(dataset_config["root_path"])
        # url = dataset_config["download_url"]

        # try:
        #     os.makedirs(output_path, exist_ok=False)
        # except FileExistsError:
        #     if not silent:
        #         print("HLSBurnScars dataset folder exists, skipping downloading dataset.")
        #     return

        # temp_file_name = f"temp_{hex(int(time.time()))}_hls_burn_scars.tar.gz"
        # pbar = DownloadProgressBar()

        # try:
        #     urllib.request.urlretrieve(url, output_path / temp_file_name, pbar)
        # except urllib.error.HTTPError as e:
        #     print('Error while downloading dataset: The server couldn\'t fulfill the request.')
        #     print('Error code: ', e.code)
        #     return
        # except urllib.error.URLError as e:
        #     print('Error while downloading dataset: Failed to reach a server.')
        #     print('Reason: ', e.reason)
        #     return

        # with tarfile.open(output_path / temp_file_name, 'r') as tar:
        #     print(f"Extracting to {output_path} ...")
        #     tar.extractall(output_path)
        #     print("done.")

        # os.remove(output_path / temp_file_name)


