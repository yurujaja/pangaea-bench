import torch.nn as nn


class Adaptor(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        encoder: nn.Module,
        finetune: bool,
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
    ):
        super().__init__()
