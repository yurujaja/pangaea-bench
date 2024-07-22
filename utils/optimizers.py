import torch
from utils.registry import OPTIMIZER_REGISTRY

@OPTIMIZER_REGISTRY.register()
def AdamW(model, cfg):
    return torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            betas=(0.9, 0.999),
            weight_decay=cfg["weight_decay"])


@OPTIMIZER_REGISTRY.register()
def SGD(model, cfg):
    return torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            nesterov=cfg["nesterov"])