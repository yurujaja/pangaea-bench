import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def MultiStepLR(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float]
) -> LRScheduler:
    
    """
    This module provides a utility function for creating a multi-step learning rate scheduler.

    Functions:
        MultiStepLR(optimizer: Optimizer, total_iters: int, lr_milestones: list[float]) -> LRScheduler
            Creates a MultiStepLR scheduler with specified milestones and decay factor.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        total_iters (int): The total number of iterations for training.
        lr_milestones (list[float]): A list of fractions representing the milestones at which the learning rate will be decayed.

    Returns:
        LRScheduler: A PyTorch learning rate scheduler that decays the learning rate at specified milestones.
    """

    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [int(total_iters * r) for r in lr_milestones], gamma=0.1
    )

