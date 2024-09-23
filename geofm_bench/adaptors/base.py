import torch.nn as nn

from geofm_bench.foundation_models.base import FoundationModel


class Adaptor(nn.Module):
    def __init__(
        self,
        foundation_model: FoundationModel,
        num_classes: int,
        finetune: bool,
    ) -> None:
        super().__init__()
        self.foundation_model = foundation_model
        self.num_classes = num_classes
        self.finetune = finetune
