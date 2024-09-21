from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn


class FoundationModel(nn.Module):
    def __init__(
        self,
        input_bands: dict[str, dict[str, list[str]]],
        input_size: int,
        embed_dim: int,
        encoder_weights: str | Path,
    ) -> None:
        super().__init__()
        self.input_bands = input_bands
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.encoder_weights = encoder_weights

    def load_encoder_weights(self, logger: Logger) -> None:
        raise NotImplementedError

    def parameters_warning(
        self,
        missing: dict[str, torch.Size],
        incompatible_shape: dict[str, tuple[torch.Size, torch.Size]],
        logger: Logger,
    ) -> None:
        if missing:
            logger.warning(
                "Missing parameters:\n"
                + "\n".join("%s: %s" % (k, v) for k, v in sorted(missing.items()))
            )
        if incompatible_shape:
            logger.warning(
                "Incompatible parameters:\n"
                + "\n".join(
                    "%s: expected %s but found %s" % (k, v[0], v[1])
                    for k, v in sorted(incompatible_shape.items())
                )
            )

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self):
        pass

