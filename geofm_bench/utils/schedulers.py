import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def MultiStepLR(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float]
) -> LRScheduler:
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [int(total_iters * r) for r in lr_milestones], gamma=0.1
    )

