import json
import logging
import os

# remove warnings from geopandas
import warnings
from datetime import datetime

import geopandas as gpd
import lightning as l
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from eo_dataset.data_augmenter import DataAugmenter
from eo_dataset.utils import rdm_crop_img_msk

# remove warnings from fiona
warnings.filterwarnings("ignore", module="fiona")
warnings.filterwarnings("ignore", module="proj")


class PastisLight(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        stage: str,
        n_classes: int,
        img_size: int,
        data_augmenter: DataAugmenter,
        categorical: bool = False,
        channels: list = [2, 1, 0],
        sequence_length: int = 61,
        target="semantic",
        cache=False,
        mem16=False,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
    ):
        super(PastisLight, self).__init__()
        self.dataset_dir = dataset_dir
        self.stage = stage
        self.n_classes = n_classes
        self.img_size = img_size
        self.data_augmenter = data_augmenter
        self.categorical = categorical
        self.channels = channels
        self.sequence_length = sequence_length
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.memory = {}
        self.memory_dates = {}
        self.target = target
        self.sats = sats
        self.value_treshold = 5.0

        # Get metadata
        self.meta_patch = gpd.read_file(os.path.join(dataset_dir, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }
        # Select Fold samples
        folds = {"train": [1, 2, 3], "val": [4], "test": [5]}[stage]
        self.meta_patch = pd.concat(
            [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
        )

        # Get norms for Sentinel 2 data
        with open(os.path.join(self.dataset_dir, "NORM_S2_patch.json"), "r") as file:
            normvals = json.loads(file.read())
        selected_folds = folds if folds is not None else range(1, 6)
        means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
        stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
        self.norm = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
        self.norm = (
            torch.from_numpy(self.norm[0])[self.channels].float(),
            torch.from_numpy(self.norm[1])[self.channels].float(),
        )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        self.max_pixel = torch.tensor([22626, 21891, 22256]).reshape(1, -1, 1, 1)
        self.min_pixel = torch.tensor([-10000, -10000, -10000]).reshape(1, -1, 1, 1)

    def __len__(self):
        return self.len

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * (x - self.min_pixel) / (self.max_pixel - self.min_pixel) - 1

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def pad_tensor(self, x: torch.Tensor) -> torch.Tensor:
        pad_length = self.sequence_length - x.shape[0]
        # pad with last elt
        last_elt = x[-1:]
        seq = torch.cat([x] + [last_elt] * pad_length, dim=0)
        return seq

    def time_warping(self, time: torch.Tensor) -> torch.Tensor:
        t_min, t_max = 16, 421
        return (time - t_min) / (t_max - t_min)

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        img_path = os.path.join(self.dataset_dir, "DATA_S2", f"S2_{id_patch}.npy")
        img = np.load(img_path).astype(np.float32)
        img = torch.from_numpy(img)[:, self.channels]
        img = self.preprocess(img)

        # get MASK data
        target_path = os.path.join(
            self.dataset_dir, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
        )
        target = np.load(target_path)
        target = torch.from_numpy(target[0].astype(int))

        # get DATES data
        dates = torch.from_numpy(self.get_dates(id_patch, "S2"))
        dates = self.time_warping(dates)

        img, target = rdm_crop_img_msk(img, target, self.img_size)
        if self.data_augmenter is not None:
            data, target = self.data_augmenter(img, target)

        return {
            "img": self.pad_tensor(img),
            "t": self.pad_tensor(dates),
            "msk": target,
        }


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return d.values


class PASTISDataModule(l.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        n_classes: int,
        img_size: int,
        batch_size: int,
        num_workers: int,
        data_augmenter: DataAugmenter,
        categorical: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.n_classes = n_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_augmenter = data_augmenter
        self.categorical = categorical

    def setup(self, stage: str | None = None):
        self.train = PastisLight(
            dataset_dir=self.dataset_dir,
            stage="train",
            n_classes=self.n_classes,
            img_size=self.img_size,
            data_augmenter=self.data_augmenter,
            categorical=self.categorical,
        )

        self.val = PastisLight(
            dataset_dir=self.dataset_dir,
            stage="val",
            n_classes=self.n_classes,
            img_size=self.img_size,
            data_augmenter=None,
            categorical=self.categorical,
        )

        self.test = PastisLight(
            dataset_dir=self.dataset_dir,
            stage="test",
            n_classes=self.n_classes,
            img_size=self.img_size,
            data_augmenter=None,
            categorical=self.categorical,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    # Example of use
    data_augmenter = DataAugmenter(["rotation", "flip", "blur"], probability=0.5)
    dm = PASTISDataModule(
        dataset_dir="/share/DEEPLEARNING/datasets/PASTIS/PASTIS/",
        n_classes=18,
        img_size=64,
        batch_size=64,
        num_workers=1,
        data_augmenter=data_augmenter,
        categorical=False,
    )
    dm.setup()
    dl = dm.train_dataloader()

    max_pxl = torch.zeros(
        3,
    )
    min_pxl = float("inf") * torch.ones(
        3,
    )

    for data in tqdm(dl):
        print("img", data["img"].shape)
        sample_max = torch.amax(data["img"], dim=(0, 1, -2, -1))
        sample_min = torch.amin(data["img"], dim=(0, 1, -2, -1))
        max_pxl = torch.maximum(max_pxl, sample_max)
        min_pxl = torch.minimum(min_pxl, sample_min)
        print("Sample mean", data["img"].mean())
        print("Sample std", data["img"].std())

    print(max_pxl)
    print(min_pxl)
