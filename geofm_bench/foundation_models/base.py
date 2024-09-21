from pathlib import Path

import torch
import torch.nn as nn


class FoundationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def input_bands(self) -> dict[str, list[str]]:
        raise NotImplementedError

    @property
    def input_size(self) -> int:
        raise NotImplementedError

    def load_encoder_weights(
        self, pretrained_path: str | Path
    ) -> tuple[dict[str, torch.Size], dict[str, tuple[torch.Size, torch.Size]]]:
        pretrained_model = torch.load(pretrained_path, map_location="cpu")
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        return missing, incompatible_shape

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self):
        pass

