import utils.registry
import omegaconf
import numpy as np
import tqdm
import datasets
import torch
import pprint

configs = [
    "configs/datasets/mados.yaml",
    "configs/datasets/hlsburnscars.yaml",
    "configs/datasets/sen1floods11.yaml",
]

for config in configs:
    cfg = omegaconf.OmegaConf.load(config)
    dataset = utils.registry.DATASET_REGISTRY.get(cfg.dataset_name)
    dataset.download(cfg, silent=False)
    train_dataset, val_dataset, test_dataset = dataset.get_splits(cfg)

    min = {}
    max = {}

    for data in tqdm.tqdm(train_dataset, desc=cfg.dataset_name):
        for modality, img in data['image'].items():
            dims = [i for i in range(len(img.shape))]
            dims.pop(-3)
            img = torch.nan_to_num(img)
            local_max = torch.amax(img, dim=dims)
            local_min = torch.amin(img, dim=dims)

            if min.get(modality, None) is None:
                print(modality, local_min.shape)
                min[modality] = torch.full_like(local_min, 10e10)
                max[modality] = torch.full_like(local_max, -10e10)

            min[modality] = torch.minimum(min[modality], local_min)
            max[modality] = torch.maximum(max[modality], local_max)

    pprint.pp(cfg.dataset_name)
    pprint.pp({
        "max": max,
        "min": min
    })
