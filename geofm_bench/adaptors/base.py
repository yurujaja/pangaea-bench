import torch.nn as nn

from geofm_bench.foundation_models.base import FoundationModel


class Adaptor(nn.Module):
    """Base class for adaptors.
    """
    def __init__(
        self,
        foundation_model: FoundationModel,
        num_classes: int,
        finetune: bool,
    ) -> None:
        """Initialize the adaptor.

        Args:
            foundation_model (FoundationModel): Fondation model used.
            num_classes (int): number of classes of the task.
            finetune (bool): if the foundation model is finetuned.
        """
        super().__init__()
        self.foundation_model = foundation_model
        self.num_classes = num_classes
        self.finetune = finetune
