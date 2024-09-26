from utils import registry
import utils.registry
import omegaconf
import numpy as np
import tqdm
import torch


class RunningStats:
    """
    Compute running statistics for a given tensor.

    This class maintains running statistics (mean, standard deviation, min, max) 
    for a given tensor. It supports updating these statistics incrementally 
    as new data is provided.

    Attributes:
        n (int): The number of elements processed.
        sum (torch.Tensor): The running sum of the elements.
        sum_2 (torch.Tensor): The running sum of the squares of the elements.
        min (torch.Tensor): The running minimum of the elements.
        max (torch.Tensor): The running maximum of the elements.

    Methods:
        update(x, reduce_dim):
            Update the running statistics with a new tensor `x`, reducing 
            dimensions specified by `reduce_dim`.
        
        finalize():
            Finalize and return the computed statistics as a dictionary.
    """

    def __init__(self, stats_dim):
        self.n = 0
        self.sum = torch.zeros(stats_dim)
        self.sum_2 = torch.zeros(stats_dim)

        self.min = 10e10 * torch.ones(stats_dim)
        self.max = -10e10 * torch.ones(stats_dim)

    def update(self, x, reduce_dim):
        self.n += np.prod([x.shape[i] for i in reduce_dim])
        self.sum += torch.sum(x, reduce_dim)
        self.sum_2 += torch.sum(x**2, reduce_dim)

        x_min = torch.amin(x, reduce_dim)
        x_max = torch.amax(x, reduce_dim)
        self.min = torch.min(self.min, x_min)
        self.max = torch.max(self.max, x_max)

    def finalize(self):
        return {
            "mean": self.sum / self.n,
            "std": torch.sqrt(self.sum_2 / self.n - (self.sum / self.n) ** 2),
            "min": self.min,
            "max": self.max,
        }


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
    stats = {}
    data = train_dataset.__getitem__(0)

    # STATS initialization
    stats = {}
    for modality, img in data["image"].items():
        n_channels = img.shape[0]
        stats[modality] = RunningStats(n_channels)

    # STATS computation
    for data in tqdm.tqdm(train_dataset, desc=cfg.dataset_name):
        for modality, img in data["image"].items():
            reduce_dim = list(range(1, img.ndim))
            stats[modality].update(img, reduce_dim)

    # STATS finalization
    for modality, stat in stats.items():
        print(modality)
        print(stat.finalize())
        print("_" * 100)
