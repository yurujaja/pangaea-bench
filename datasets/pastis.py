import json
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import torch
import rasterio
from datetime import datetime

# from utils.registry import DATASET_REGISTRY
from omegaconf import OmegaConf


def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name" and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    for key in ["s2", "s1-asc", "s1-des", "s1"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack(
                [
                    torch.nn.functional.pad(
                        tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0))
                    )
                    for tensor in idx
                ],
                dim=0,
            )
            output[key] = stacked_tensor.float()
            keys.remove(key)
            key = "_".join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack(
                [
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ],
                dim=0,
            )
            output[key] = stacked_tensor.long()
            keys.remove(key)
    if "name" in keys:
        output["name"] = [x["name"] for x in batch]
        keys.remove("name")
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch]).float()
    return output


def prepare_dates(date_dict, reference_date):
    """Date formating."""
    if type(date_dict) == str:
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


# @DATASET_REGISTRY.register()
class PASTIS(Dataset):
    def __init__(
        self,
        cfg: OmegaConf,
        split: str,
        is_train: bool = True,
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
        super(PASTIS, self).__init__()

        if split == "train":
            folds = [1, 2, 3]
        elif split == "val":
            folds = [4]
        elif split == "test":
            folds = [5]

        self.split = split
        self.path = cfg["root_path"]
        self.data_mean = cfg["data_mean"]
        self.data_std = cfg["data_std"]
        self.data_min = cfg["data_min"]
        self.data_max = cfg["data_max"]
        self.classes = cfg["classes"]
        self.class_num = len(self.classes)

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
        self.collate_fn = collate_fn

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
        label = torch.unique(split_image(label, self.nb_split, part)).long()
        label = torch.sum(
            torch.nn.functional.one_hot(label, num_classes=self.num_classes), dim=0
        )
        label = label[1:-1]  # remove Background and Void classes
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

        return {
            "image": {
                "optical": output["s2"].to(torch.float32),
                "sar": output["s1-asc"].to(torch.float32),
            },
            "target": output["label"],
            "metadata": {},
        }

    def __len__(self) -> int:
        return len(self.meta_patch) * self.nb_split * self.nb_split

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = PASTIS(cfg=dataset_config, split="train", is_train=True)
        dataset_val = PASTIS(cfg=dataset_config, split="val", is_train=False)
        dataset_test = PASTIS(cfg=dataset_config, split="test", is_train=False)
        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def download(dataset_config: dict, silent=False):
        pass


if __name__ == "__main__":
    from tqdm import tqdm

    class_prob = {
        "Background": 0.0,
        "Meadow": 31292,
        "Soft Winter Wheat": 8206,
        "Corn": 13123,
        "Winter Barley": 2766,
        "Winter Rapeseed": 1769,
        "Spring Barley": 908,
        "Sunflower": 1355,
        "Grapevine": 10640,
        "Beet": 871,
        "Winter Triticale": 1208,
        "Winter Durum Wheat": 1704,
        "Fruits, Vegetables, Flowers": 2619,
        "Potatoes": 551,
        "Leguminous Fodder": 3174,
        "Soybeans": 1212,
        "Orchard": 2998,
        "Mixed Cereal": 848,
        "Sorghum": 707,
        "Void Label": 35924,
    }

    # get the class weights
    class_weights = np.array([class_prob[key] for key in class_prob.keys()])
    class_weights = class_weights / class_weights.sum()
    print("Class weights: ")
    for i, key in enumerate(class_prob.keys()):
        print(key, "->", class_weights[i])
    print("_" * 100)

    cfg = {
        "root_path": "/share/DEEPLEARNING/datasets/PASTIS-HD",
        "data_mean": None,
        "data_std": None,
        "classes": {
            "0": "Background",
            "1": "Meadow",
        },
        "data_min": 0,
        "data_max": 1,
    }

    dataset = PASTIS(cfg, "train", is_train=True)
    train_dataset, val_dataset, test_dataset = PASTIS.get_splits(cfg)

    class RunningStats:
        def __init__(self, stats_dim):
            self.n = torch.zeros(stats_dim)
            self.mean = torch.zeros(stats_dim)
            self.M2 = torch.zeros(stats_dim)

            self.min = 10e10 * torch.ones(stats_dim)
            self.max = -10e10 * torch.ones(stats_dim)

        def update_tensor(self, x: torch.Tensor):
            # tensor of shape (T, C, H, W)
            for frame in x:
                frame = frame.reshape(-1, frame.shape[0])
                for pixel in frame:
                    self.update(pixel)

        def update(self, x):
            self.n += 1
            delta = x - self.mean

            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

            self.min = torch.min(self.min, x)
            self.max = torch.max(self.max, x)

        def finalize(self):
            return {
                "mean": self.mean,
                "std": torch.sqrt(self.M2 / self.n),
                "min": self.min,
                "max": self.max,
            }

    data = train_dataset.__getitem__(0)
    sar = data["image"]["sar"]
    optical = data["image"]["optical"]

    for i in tqdm(range(len(train_dataset))):
        data = train_dataset.__getitem__(i)
        sar = torch.cat([sar, data["image"]["sar"]], dim=0)
        optical = torch.cat([optical, data["image"]["optical"]], dim=0)

    print("SAR shape: ", sar.shape)
    print("Optical shape: ", optical.shape)
    reduce_dim = (0, 2, 3)
    print("_" * 100)
    print("SAR min: ", sar.amin(reduce_dim))
    print("SAR max: ", sar.amax(reduce_dim))
    print("SAR mean: ", sar.mean(reduce_dim))
    print("SAR std: ", sar.std(reduce_dim))
    print("_" * 100)
    print("Optical min: ", optical.amin(reduce_dim))
    print("Optical max: ", optical.amax(reduce_dim))
    print("Optical mean: ", optical.mean(reduce_dim))
    print("Optical std: ", optical.std(reduce_dim))
