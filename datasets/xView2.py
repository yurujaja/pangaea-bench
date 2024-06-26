# Sources: 
# - https://github.com/PaulBorneP/Xview2_Strong_Baseline/blob/master/datasets/base_dataset.py
# - https://github.com/PaulBorneP/Xview2_Strong_Baseline/blob/master/datasets/supervised_dataset.py

from typing import Sequence, Dict, Any, Union, Literal, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pathlib
from sklearn.model_selection import train_test_split

class xView2(Dataset):
    def __init__(self, data_root: str="data/xView2", split:Literal["train", "val", "test"]="train", standardize:bool = True, transform = None) -> None:
        
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.standardize = standardize
        self.transform = transform

        self.bands = ["B4", "B3", "B2"]
        self.means = torch.tensor([66.7703, 88.4452, 85.1047])[:, None, None, None] # shape [1,3,1,1]
        self.stds =  torch.tensor([48.3066, 51.9129, 62.7612])[:, None, None, None] # shape [1,3,1,1]

        self.all_files = self.get_all_files()
        

    def get_all_files(self) -> Sequence[str]:
        all_files = []
        if self.split == "test":
            data_dirs = [os.path.join(self.data_root, "test")]
        else:
            # Train and val consist of the smaller train and the larger tier3 set.
            data_dirs = [os.path.join(self.data_root, d) for d in ["train", "tier3"]]

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

        msk_pre = cv2.imread(fn.replace('/images/', '/masks/'),
                             cv2.IMREAD_UNCHANGED)
        msk_post = cv2.imread(fn.replace('/images/', '/masks/').replace(
            '_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)

        msk = np.stack([msk_pre, msk_post], axis=0)
        img = np.stack([img_pre, img_post], axis=0) 

        # Reshaping tensors from (T, H, W, C) to (C, T, H, W)
        img = torch.from_numpy(img.transpose((3, 0, 1, 2))).float()
        msk = torch.from_numpy(msk).float()

        if self.standardize:
            img = (img - self.means) / self.stds

        if self.transform:
            img = self.transform(img)
            msk = self.transform(msk)


        return {
            'image': {
                'optical': img
            },
            'target': msk,  
            'metadata': {"filename":fn}
        }
