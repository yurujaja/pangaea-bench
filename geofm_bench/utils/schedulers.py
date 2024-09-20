import torch
from utils.registry import SCHEDULER_REGISTRY

@SCHEDULER_REGISTRY.register()
def MultiStepLR(optimizer, total_iters, cfg):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        [total_iters * r for r in cfg["lr_milestones"]], 
        gamma=0.1)