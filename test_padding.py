import torch
from engine import get_collate_fn
from omegaconf import OmegaConf
import random

batch = []
for i in range(5):
    T = random.randint(1, 10)
    batch.append(
        {
            "image": {
                "optical": torch.rand(3, T, 256, 256),
                "sar": torch.rand(2, T, 256, 256),
            },
            "target": torch.rand(1, 256, 256),
        }
    )
cfg = OmegaConf.load("configs/datasets/mados.yaml")
print(cfg)
cfg["bands"]["sar"] = 2
cfg["dataset"] = cfg
print(cfg.bands)
collate_fn = get_collate_fn(cfg)
output = collate_fn(batch)
print(output["image"]["optical"].shape)
print(output["image"]["sar"].shape)
print(output["target"].shape)
