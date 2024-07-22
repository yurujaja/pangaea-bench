import torch
from utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
def CrossEntropy(cfg):
    return torch.nn.CrossEntropyLoss(ignore_index = cfg["ignore_index"])

@LOSS_REGISTRY.register()
def MSELoss(cfg):
    return torch.nn.MSELoss()