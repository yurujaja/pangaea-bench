import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.encoders.base import Encoder
import matplotlib.pyplot as plt

class Decoder(nn.Module):
    """Base class for decoders."""

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ) -> None:
        """Initialize the decoder.

        Args:
            encoder (Encoder): encoder used.
            num_classes (int): number of classes of the task.
            finetune (bool): whether the encoder is finetuned.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.finetune = finetune





