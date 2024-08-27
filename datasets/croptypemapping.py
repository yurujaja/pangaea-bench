# Adapted from https://github.com/sustainlab-group/sustainbench/blob/main/sustainbench/datasets/croptypemapping_dataset.py

import os
import json
import gdown
import tarfile
import shutil

import numpy as np
import pandas as pd

import torch

from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CropTypeMappingSouthSudan(torch.utils.data.Dataset):
    def __init__(self, cfg, split) -> None:
        super().__init__()

        self.root_path = cfg['root_path']
        self.data_mean = cfg['data_mean']
        self.data_std = cfg['data_std']
        self.classes = cfg['classes']
        self.class_num = len(self.classes)
        self.split = split
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.split_mapping = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

        self.country = 'southsudan'
        self.grid_size = 20

        split_df = pd.read_csv(os.path.join(self.root_path, self.country, 'list_eval_partition.csv'))
        self.split_array = split_df['partition'].values

        split_mask = self.split_array == self.split_dict[split]
        self.split_indices = np.where(split_mask)[0]
        self.ori_ids = torch.from_numpy(split_df['id'].values)

    
    def __getitem__(self, idx):
        id = self.split_indices[idx]
        loc_id = f'{self.ori_ids[id]:06d}'

        images = np.load(os.path.join(self.root_path, self.country, 'npy', f'{self.country}_{loc_id}.npz'))

        s1 = torch.from_numpy(images['s1'])[:2, ...]  # only use VV and VH bands
        s2 = torch.from_numpy(images['s2'])

        s1 = self.pad_or_crop(s1)
        s2 = self.pad_or_crop(s2)

        s1 = torch.permute(s1, (0, 3, 1, 2))  # C, T, H, W
        s2 = torch.permute(s2, (0, 3, 1, 2))  # C, T, H, W

        label = np.load(os.path.join(self.root_path, self.country, 'truth', f'{self.country}_{loc_id}.npz'))['truth']
        label = self._mapping_label(label)
        label = torch.from_numpy(label)

        metadata = self.get_metadata(idx)

        output = {
            'image': {
                'optical': s2,
                'sar': s1
            },
            'target': label,
            'metadata': metadata
        }
        
        return output

    def _mapping_label(self, label):
        # The dataset uses top four crop types 
        # Reference: https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.pdf

        transformed_label = np.full_like(label, -1)
        mapping = {1: 0, 2: 1, 3: 2, 4: 3}

        for k, v in mapping.items():
            transformed_label[label == k] = v

        return transformed_label
    
    def __len__(self):
        return len(self.split_indices)

    def get_dates(self, json_file):
        """
        Converts json dates into tensor containing dates
        """
        dates = np.array(json_file['dates'])
        dates = np.char.replace(dates, '-', '')
        dates = torch.from_numpy(dates.astype(np.int32))
        return dates
    
    def get_metadata(self, idx):
        """
        Returns metadata for a given idx.
        Dates are returned as integers in format {Year}{Month}{Day}
        """
        id = self.split_indices[idx]
        loc_id = f'{self.ori_ids[id]:06d}'

        s1_json = json.loads(open(os.path.join(self.root_path, self.country, 's1', f's1_{self.country}_{loc_id}.json'), 'r').read())
        s1 = self.get_dates(s1_json)

        s2_json = json.loads(open(os.path.join(self.root_path, self.country, 's2', f's2_{self.country}_{loc_id}.json'), 'r').read())
        s2 = self.get_dates(s2_json)

        s1 = self.pad_or_crop(s1)
        s2 = self.pad_or_crop(s2)
        
        return {'s1': s1, 's2': s2}
    
    def pad_or_crop(self, tensor):
        '''
        Right pads or crops tensor to GRID_SIZE.
        '''
        # if self.grid_size >= tensor.shape[-1]:
        pad_size = self.grid_size - tensor.shape[-1]
        tensor = torch.nn.functional.pad(input=tensor, pad=(0, pad_size), value=0)
        # else:
        #     tensor = tensor[..., :self.grid_size]
        return tensor
    
    @staticmethod
    def get_splits(dataset_config):
        dataset_train = CropTypeMappingSouthSudan(cfg=dataset_config, split="train")
        dataset_val = CropTypeMappingSouthSudan(cfg=dataset_config, split="val")
        dataset_test = CropTypeMappingSouthSudan(cfg=dataset_config, split="test")
        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def download(dataset_config: dict, silent=False):
        if os.path.exists(dataset_config["root_path"]):
            if not silent:
                print("CropTypeMapping Dataset folder exists, skipping downloading dataset.")
            return

        output_path = dataset_config["root_path"]
        os.makedirs(output_path, exist_ok=True)
        url = dataset_config["download_url"]

        temp_file = os.path.join(output_path, "archive.tar.gz")

        try:
            gdown.download_folder(url=url,output=output_path, quiet=False, use_cookies=False)
        except Exception as e:
                print(f'{temp_file} may be corrupted. Please try deleting it and rerunning.\n')
                print(f"Exception: ", e)

        with tarfile.open(temp_file, 'r:gz') as tar:
            print(f"Extracting {temp_file}...")
            tar.extractall(path=output_path)
        
        os.remove(temp_file)
        for item in os.listdir(os.path.join(output_path, 'africa_crop_type_mapping_v1.0')):
            shutil.move(os.path.join(output_path, 'africa_crop_type_mapping_v1.0', item), os.path.join(output_path, item))
        os.rmdir(os.path.join(output_path, 'africa_crop_type_mapping_v1.0'))

