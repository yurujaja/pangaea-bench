import os
import pathlib
import torch
import numpy as np
from glob import glob
from pyDataverse.api import NativeApi, DataAccessApi
from tifffile import imread
from utils.registry import DATASET_REGISTRY
import requests


@DATASET_REGISTRY.register()
class AI4SmallFarms(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.root_path = pathlib.Path(cfg['root_path'])
        self.split = split
        self.image_dir = self.root_path.joinpath(f"sentinel-2-asia/{split}/images")
        self.mask_dir = self.root_path.joinpath(f"sentinel-2-asia/{split}/masks")

        self.image_list = sorted(glob(str(self.image_dir.joinpath("*.tif"))))
        self.mask_list = sorted(glob(str(self.mask_dir.joinpath("*.tif"))))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = imread(pathlib.Path(self.image_list[index]))
        target = imread(pathlib.Path(self.mask_list[index]))  # Assuming target is a single band

        # Convert the image and target to supported types
        image = image.astype(np.float32)  # Convert to float32
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)

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

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = AI4SmallFarms(cfg=dataset_config, split="train")
        dataset_val = AI4SmallFarms(cfg=dataset_config, split="validate")
        dataset_test = AI4SmallFarms(cfg=dataset_config, split="test")
        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def download(dataset_config: dict, silent=False):
        root_path = pathlib.Path(dataset_config["root_path"])

        # Create the root directory if it does not exist
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)

        if root_path.exists() and any(root_path.iterdir()):
            if not silent:
                print(f"Dataset already exists at {root_path}. Skipping download.")
            return

        # Set up the Dataverse API
        base_url = "https://phys-techsciences.datastations.nl"
        api = NativeApi(base_url)
        data_api = DataAccessApi(base_url)
        DOI = "doi:10.17026/dans-xy6-ngg6"

        # Fetch dataset files using NativeAPI
        try:
            dataset = api.get_dataset(DOI)
            files_list = dataset.json()['data']['latestVersion']['files']
        except Exception as e:
            if not silent:
                print(f"Error retrieving dataset metadata: {e}")
            return

        if not files_list:
            if not silent:
                print(f"No files found for DOI: {DOI}")
            return

        # Process each file and download
        for file in files_list:
            filename = file["dataFile"]["filename"]
            file_id = file["dataFile"]["id"]
            directory_label = file.get("directoryLabel", "")
            dv_path = os.path.join(directory_label, filename)

            # Construct the full path for the file
            file_path = root_path / dv_path

            # Create subdirectories if they do not exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if not silent:
                print(f"Downloading file: {filename}, id: {file_id} to {file_path}")

            try:
                # Download the file using its ID
                response = data_api.get_datafile(file_id, is_pid=False)
                response.raise_for_status()  # Check if the request was successful
                with open(file_path, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    if not silent:
                        print(f"File {filename} with id {file_id} not found (404). Skipping.")
                else:
                    if not silent:
                        print(f"Error downloading file {filename} with id {file_id}: {err}")
                    raise

        # **Cleanup: Remove unwanted files and directories**
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