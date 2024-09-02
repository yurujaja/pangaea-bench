import os
import pathlib
import torch
import numpy as np
import yaml
import logging

from glob import glob
from torchvision.transforms import Normalize
from easyDataverse import Dataverse
from utils.registry import DATASET_REGISTRY
from tifffile import imread

@DATASET_REGISTRY.register()
class AI4SmallFarms(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.root_path = pathlib.Path(cfg['root_path'])
        self.split = split
        self.img_size = cfg['img_size']  # Get the tile size from the config
        self.image_dir = self.root_path.joinpath(f"sentinel-2-asia/{split}/images")
        self.mask_dir = self.root_path.joinpath(f"sentinel-2-asia/{split}/masks")

        self.image_list = sorted(glob(str(self.image_dir.joinpath("*.tif"))))
        self.mask_list = sorted(glob(str(self.mask_dir.joinpath("*.tif"))))

        self.data_mean = cfg['data_mean']['optical']
        self.data_std = cfg['data_std']['optical']

        self.transform = Normalize(mean=self.data_mean, std=self.data_std)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = imread(pathlib.Path(self.image_list[index]))
        target = imread(pathlib.Path(self.mask_list[index]))  # Assuming target is a single band

        # Convert the image and target to supported types
        image = image.astype(np.float32)  # Convert to float32
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)

        # Tile the image and target to the fixed size specified in the config
        image, target = self.tile_image_and_mask(image, target, self.img_size)

        image = torch.from_numpy(image).permute(2, 0, 1)
        target = torch.from_numpy(target).long()

        # Handle invalid data if any
        invalid_mask = torch.isnan(image)
        image[invalid_mask] = 0

        # Convert target to a boolean tensor
        target = target.bool()

        return {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {}
        }

    def tile_image_and_mask(self, image, target, tile_size):
        """Tiles the image and target to the specified tile_size."""
        # Ensure the image and target have the same dimensions
        height, width, _ = image.shape
        target_height, target_width = target.shape

        if height != target_height or width != target_width:
            raise ValueError("Image and target sizes do not match.")

        # Select the first tile (top-left corner)
        image_tile = image[:tile_size, :tile_size, :]
        target_tile = target[:tile_size, :tile_size]

        return image_tile, target_tile

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = AI4SmallFarms(cfg=dataset_config, split="train")
        dataset_val = AI4SmallFarms(cfg=dataset_config, split="validate")
        dataset_test = AI4SmallFarms(cfg=dataset_config, split="test")
        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def download(dataset_config: dict, silent=False):
        root_path = pathlib.Path(dataset_config["root_path"])

        if root_path.exists() and any(root_path.iterdir()):
            if not silent:
                print(f"Dataset already exists at {root_path}. Skipping download.")
            return

        dataverse = Dataverse(
            server_url="https://phys-techsciences.datastations.nl",
            api_token=dataset_config.get("dataverse_api_token", None)
        )

        dataset = dataverse.load_dataset(
            pid="doi:10.17026/dans-xy6-ngg6",
            version="1",
            filedir=root_path
        )

        # List of unwanted files and directories to remove
        unwanted_paths = [
            os.path.join(dataset_config["root_path"], 'easy-migration.zip'),
            os.path.join(dataset_config["root_path"], 'readme.md'),
            os.path.join(dataset_config["root_path"], 'sentinel-2-asia', 'benchmark.qgz'),
            os.path.join(dataset_config["root_path"], 'sentinel-2-asia', 'tiles_asia.gpkg'),
            os.path.join(dataset_config["root_path"], 'sentinel-2-asia', 'reference'),
            os.path.join(dataset_config["root_path"], 'sentinel-2-asia', 'test', 'output'),
            os.path.join(dataset_config["root_path"], 'sentinel-2-nl'),
        ]

        # Remove unwanted files and directories
        for path in unwanted_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    os.system(f'rm -rf {path}')  # Remove directories
                else:
                    os.remove(path)  # Remove files
                if not silent:
                    print(f"Removed unwanted path: {path}")

        if not silent:
            print(f"Downloaded dataset to {dataset_config['root_path']}")
