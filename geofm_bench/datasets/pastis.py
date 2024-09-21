###
# Modified version of the PASTIS-HD dataset
# original code https://github.com/gastruc/OmniSat/blob/main/src/data/Pastis.py
###

import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from einops import rearrange
from torch.utils.data import Dataset


def prepare_dates(date_dict, reference_date):
    """Date formating."""
    if type(date_dict) is str:
        date_dict = json.loads(date_dict)
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return torch.tensor(d.values)


def split_image(image_tensor, nb_split, id):
    """
    Split the input image tensor into four quadrants based on the integer i.
    To use if Pastis data does not fit in your GPU memory.
    Returns the corresponding quadrant based on the value of i
    """
    if nb_split == 1:
        return image_tensor
    i1 = id // nb_split
    i2 = id % nb_split
    height, width = image_tensor.shape[-2:]
    half_height = height // nb_split
    half_width = width // nb_split
    if image_tensor.dim() == 4:
        return image_tensor[
            :,
            :,
            i1 * half_height : (i1 + 1) * half_height,
            i2 * half_width : (i2 + 1) * half_width,
        ].float()
    if image_tensor.dim() == 3:
        return image_tensor[
            :,
            i1 * half_height : (i1 + 1) * half_height,
            i2 * half_width : (i2 + 1) * half_width,
        ].float()
    if image_tensor.dim() == 2:
        return image_tensor[
            i1 * half_height : (i1 + 1) * half_height,
            i2 * half_width : (i2 + 1) * half_width,
        ].float()


class Pastis(Dataset):
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
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            folds (list): list of folds to use
            reference_date (str date): reference date for the data
            nb_split (int): number of splits from one observation
            num_classes (int): number of classes
        """
        super(Pastis, self).__init__()

        if split == "train":
            folds = [1, 2, 3]
        elif split == "val":
            folds = [4]
        elif split == "test":
            folds = [5]

        self.dataset_name = dataset_name
        self.bands = bands
        self.split = split
        self.path = root_path
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.grid_size = multi_temporal
        self.download_url = download_url
        self.auto_download = auto_download
        self.modalities = ["s2", "aerial", "s1-asc"]
        self.nb_split = 1

        reference_date = "2018-09-01"
        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.meta_patch = gpd.read_file(os.path.join(self.path, "metadata.geojson"))

        self.num_classes = 20

        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        line = self.meta_patch.iloc[i // (self.nb_split * self.nb_split)]
        name = line["ID_PATCH"]
        part = i % (self.nb_split * self.nb_split)
        label = torch.from_numpy(
            np.load(
                os.path.join(self.path, "ANNOTATIONS/TARGET_" + str(name) + ".npy")
            )[0].astype(np.int32)
        )
        # label = torch.unique(split_image(label, self.nb_split, part)).long()
        # label = torch.sum(
        #     torch.nn.functional.one_hot(label, num_classes=self.num_classes), dim=0
        # )
        # label = label[1:-1]  # remove Background and Void classes
        output = {"label": label, "name": name}

        for modality in self.modalities:
            if modality == "aerial":
                with rasterio.open(
                    os.path.join(
                        self.path,
                        "DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019/SPOT6_RVB_1M00_2019_"
                        + str(name)
                        + ".tif",
                    )
                ) as f:
                    output["aerial"] = split_image(
                        torch.FloatTensor(f.read()), self.nb_split, part
                    )
            elif modality == "s1-median":
                modality_name = "s1a"
                images = split_image(
                    torch.from_numpy(
                        np.load(
                            os.path.join(
                                self.path,
                                "DATA_{}".format(modality_name.upper()),
                                "{}_{}.npy".format(modality_name.upper(), name),
                            )
                        )
                    ),
                    self.nb_split,
                    part,
                ).to(torch.float32)
                out, _ = torch.median(images, dim=0)
                output[modality] = out
            elif modality == "s2-median":
                modality_name = "s2"
                images = split_image(
                    torch.from_numpy(
                        np.load(
                            os.path.join(
                                self.path,
                                "DATA_{}".format(modality_name.upper()),
                                "{}_{}.npy".format(modality_name.upper(), name),
                            )
                        )
                    ),
                    self.nb_split,
                    part,
                ).to(torch.float32)
                out, _ = torch.median(images, dim=0)
                output[modality] = out
            elif modality == "s1-4season-median":
                modality_name = "s1a"
                images = split_image(
                    torch.from_numpy(
                        np.load(
                            os.path.join(
                                self.path,
                                "DATA_{}".format(modality_name.upper()),
                                "{}_{}.npy".format(modality_name.upper(), name),
                            )
                        )
                    ),
                    self.nb_split,
                    part,
                ).to(torch.float32)
                dates = prepare_dates(
                    line["-".join(["dates", modality_name.upper()])],
                    self.reference_date,
                )
                l = []
                for i in range(4):
                    mask = (dates >= 92 * i) & (dates < 92 * (i + 1))
                    if sum(mask) > 0:
                        r, _ = torch.median(images[mask], dim=0)
                        l.append(r)
                    else:
                        l.append(
                            torch.zeros(
                                (images.shape[1], images.shape[-2], images.shape[-1])
                            )
                        )
                output[modality] = torch.cat(l)
            elif modality == "s2-4season-median":
                modality_name = "s2"
                images = split_image(
                    torch.from_numpy(
                        np.load(
                            os.path.join(
                                self.path,
                                "DATA_{}".format(modality_name.upper()),
                                "{}_{}.npy".format(modality_name.upper(), name),
                            )
                        )
                    ),
                    self.nb_split,
                    part,
                ).to(torch.float32)
                dates = prepare_dates(
                    line["-".join(["dates", modality_name.upper()])],
                    self.reference_date,
                )
                l = []
                for i in range(4):
                    mask = (dates >= 92 * i) & (dates < 92 * (i + 1))
                    if sum(mask) > 0:
                        r, _ = torch.median(images[mask], dim=0)
                        l.append(r)
                    else:
                        l.append(
                            torch.zeros(
                                (images.shape[1], images.shape[-2], images.shape[-1])
                            )
                        )
                output[modality] = torch.cat(l)
            else:
                if len(modality) > 3:
                    modality_name = modality[:2] + modality[3]
                    output[modality] = split_image(
                        torch.from_numpy(
                            np.load(
                                os.path.join(
                                    self.path,
                                    "DATA_{}".format(modality_name.upper()),
                                    "{}_{}.npy".format(modality_name.upper(), name),
                                )
                            )
                        ),
                        self.nb_split,
                        part,
                    )
                    output["_".join([modality, "dates"])] = prepare_dates(
                        line["-".join(["dates", modality_name.upper()])],
                        self.reference_date,
                    )
                else:
                    output[modality] = split_image(
                        torch.from_numpy(
                            np.load(
                                os.path.join(
                                    self.path,
                                    "DATA_{}".format(modality.upper()),
                                    "{}_{}.npy".format(modality.upper(), name),
                                )
                            )
                        ),
                        self.nb_split,
                        part,
                    )
                    output["_".join([modality, "dates"])] = prepare_dates(
                        line["-".join(["dates", modality.upper()])], self.reference_date
                    )
                N = len(output[modality])
                if N > 50:
                    random_indices = torch.randperm(N)[:50]
                    output[modality] = output[modality][random_indices]
                    output["_".join([modality, "dates"])] = output[
                        "_".join([modality, "dates"])
                    ][random_indices]

        optical_ts = rearrange(output["s2"], "t c h w -> c t h w")
        sar_ts = rearrange(output["s1-asc"], "t c h w -> c t h w")

        if self.grid_size == 1:
            # we only take the last frame
            optical_ts = optical_ts[:, -1]
            sar_ts = sar_ts[:, -1]
        else:
            # select evenly spaced samples
            optical_indexes = torch.linspace(
                0, optical_ts.shape[1] - 1, self.grid_size, dtype=torch.long
            )
            sar_indexes = torch.linspace(
                0, sar_ts.shape[1] - 1, self.grid_size, dtype=torch.long
            )

            optical_ts = optical_ts[:, optical_indexes]
            sar_ts = sar_ts[:, sar_indexes]

        return {
            "image": {
                "optical": optical_ts.to(torch.float32),
                "sar": sar_ts.to(torch.float32),
            },
            "target": output["label"],
            "metadata": {},
        }

    def __len__(self) -> int:
        return len(self.meta_patch) * self.nb_split * self.nb_split

    @staticmethod
    def download(dataset_config: dict, silent=False):
        pass
