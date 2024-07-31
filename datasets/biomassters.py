import numpy as np
import torch
import pandas as pd
import pathlib
from .utils import read_tif
from utils.registry import DATASET_REGISTRY

s1_min = np.array([-25 , -62 , -25, -60], dtype="float32")
s1_max = np.array([ 29 ,  28,  30,  22 ], dtype="float32")
s1_mm = s1_max - s1_min

s2_max = np.array(
    [19616., 18400., 17536., 17097., 16928., 16768., 16593., 16492., 15401., 15226.,   255.],
    dtype="float32",
)

IMG_SIZE = (256, 256)


def read_imgs(chip_id: str, data_dir: pathlib.Path):
    imgs, imgs_s1, imgs_s2, mask = [], [], [], []
    for month in range(12):
        img_s1 = read_tif(data_dir.joinpath(f"{chip_id}_S1_{month:0>2}.tif"))
        m = img_s1 == -9999
        img_s1 = img_s1.astype("float32")
        img_s1 = (img_s1 - s1_min) / s1_mm
        img_s1 = np.where(m, 0, img_s1)
        filepath = data_dir.joinpath(f"{chip_id}_S2_{month:0>2}.tif")
        if filepath.exists():
            img_s2 = read_tif(filepath)
            img_s2 = img_s2.astype("float32")
            img_s2 = img_s2 / s2_max
        else:
            img_s2 = np.zeros(IMG_SIZE + (11,), dtype="float32")

        # img = np.concatenate([img_s1, img_s2], axis=2)
        img_s1 = np.transpose(img_s1, (2, 0, 1))
        img_s2 = np.transpose(img_s2, (2, 0, 1))
        imgs_s1.append(img_s1)
        imgs_s2.append(img_s2)
        mask.append(False)

    mask = np.array(mask)

    imgs_s1 = np.stack(imgs_s1, axis=1)  # [c, t, h, w]
    imgs_s2 = np.stack(imgs_s2, axis=1)  # [c, t, h, w]

    return imgs_s1, imgs_s2, mask

@DATASET_REGISTRY.register()
class BioMassters(torch.utils.data.Dataset):
    def __init__(self, cfg, split): #, augs=False):
        df_path = pathlib.Path(cfg["root_path"]).joinpath("The_BioMassters_-_features_metadata.csv.csv")
        df: pd.DataFrame = pd.read_csv(str(df_path))
        self.df = df[df.split == split].copy()
        self.dir_features = pathlib.Path(cfg["root_path"]).joinpath(f"{split}_features")
        self.dir_labels = pathlib.Path(cfg["root_path"]).joinpath( f"{split}_agbm")
        self.split = split
        # self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        # print(item.chip_id)
        # print(self.dir_features)

        imgs_s1, imgs_s2, mask = read_imgs(item.chip_id, self.dir_features)
        if self.dir_labels is not None:
            target = read_tif(self.dir_labels.joinpath(f'{item.chip_id}_agbm.tif'))
        else:
            target = item.chip_id


        # Reshaping tensors from (T, H, W, C) to (C, T, H, W)
        imgs_s1 = torch.from_numpy(imgs_s1).float()
        imgs_s2 = torch.from_numpy(imgs_s2).float()
        target = torch.from_numpy(target).float()

        return {
            'image': {
                    'optical': imgs_s2,
                    'sar' : imgs_s1,
                    },
            'target': target,  
            'metadata': {"masks":mask}
        }

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = BioMassters(cfg=dataset_config, split="test")
        dataset_val = BioMassters(cfg=dataset_config, split="test")
        dataset_test = BioMassters(cfg=dataset_config, split="test")
        return dataset_train, dataset_val, dataset_test
    
    @staticmethod
    def download(dataset_config:dict, silent=False):
        pass


if __name__ == '__main__':

    dataset = BioMassters(cfg, split = "test")

    train_dict = dataset.__getitem__(0)

    print(train_dict["image"]["optical"].shape)
    print(train_dict["image"]["sar"].shape)
    print(train_dict["target"].shape)