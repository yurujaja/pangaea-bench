from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn


class FoundationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._input_bands = None
        self._input_size = None

    @property
    def input_bands(self) -> dict[str, list[str]]:
        raise NotImplementedError

    @input_bands.setter
    def input_bands(self, value: dict[str, list[str]]) -> None:
        self._input_bands = value

    @property
    def input_size(self) -> int:
        raise NotImplementedError

    @input_size.setter
    def input_size(self, value: int) -> None:
        self._input_size = value

    def load_encoder_weights(self, pretrained_path: str | Path, logger: Logger) -> None:
        # load weight
        # call parameters_warning
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

