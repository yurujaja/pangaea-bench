from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import torch
import rasterio
from datetime import datetime


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


class PASTIS(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        folds=None,
        reference_date="2018-09-01",
        nb_split=1,
        num_classes=20,
    ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (torch module): transform to apply to the data
            folds (list): list of folds to use
            reference_date (str date): reference date for the data
            nb_split (int): number of splits from one observation
            num_classes (int): number of classes
        """
        super(PASTIS, self).__init__()
        self.path = path
        self.transform = transform
        self.modalities = modalities
        self.nb_split = nb_split

        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.meta_patch = gpd.read_file(os.path.join(self.path, "metadata.geojson"))

        self.num_classes = num_classes

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

        return self.transform(output)

    def __len__(self):
        return len(self.meta_patch) * self.nb_split * self.nb_split


if __name__ == "__main__":
    path = "/share/DEEPLEARNING/datasets/PASTIS-HD"
    modalities = ["s2", "aerial"]
    transform = lambda x: x
    dataset = PASTIS(path, modalities, transform, folds=[1, 2, 3], nb_split=1)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    for i, data in enumerate(dataloader):
        print("Key: ", data.keys())
        print(data["label"].shape)
        print(data["s2"].shape)
        print(data["aerial"].shape)
        break
