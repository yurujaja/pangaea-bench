import os

from datasets.mados import MADOS
from datasets.croptypemapping import CropTypeMappingDataset
from datasets.sen1floods11 import Sen1Floods11
from datasets.hlsburnscars import BurnScarsDataset


def make_dataset(ds_name, path, **kwargs):
    datasets = {
        "mados": MADOS,
        "crop_type_mapping": CropTypeMappingDataset,
        "sen1floods11": Sen1Floods11,
        "hlsburnscars": BurnScarsDataset,
        
    }
    if ds_name not in datasets:
        raise ValueError(f"{ds_name} is not yet supported.")
    
    if ds_name == "mados":
        splits_path = os.path.join(path, "splits")
        dataset_train = MADOS(path, splits_path, "train")
        dataset_val = MADOS(path, splits_path, "val")
        dataset_test = MADOS(path, splits_path, "val")
    elif ds_name == "crop_type_mapping":
        dataset = CropTypeMappingDataset(data_dir=path, split_scheme='southsudan', calculate_bands=False, normalize=True)
        dataset_train = dataset.get_subset('train')
        dataset_val = dataset.get_subset('val')
        dataset_test = dataset.get_subset('test')
    elif ds_name == "sen1floods11":
        dataset_train = Sen1Floods11(data_root=path, split="train")
        dataset_val = Sen1Floods11(data_root=path, split="val")
        dataset_test = Sen1Floods11(data_root=path, split="test")
    elif ds_name == "hlsburnscars":
        dataset_train = BurnScarsDataset(data_root=path, split="train")
        dataset_val = BurnScarsDataset(data_root=path, split="val")
    
    return dataset_train, dataset_val, dataset_test