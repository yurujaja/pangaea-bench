import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from pangaea.encoders.base import Encoder


# Some encoders configs depend on dataset configs (via interpolation), exclude them for now.
# TODO: include dofa which depends on dataset.bands
# TODO: include prithvi which depends on dataset.multi_temporal
# TODO: include ssl4eo_mae_sar, ssl4eo_mae_optical, unet_encoder which depends on dataset.img_size
@pytest.mark.parametrize(
    "config_name",
    [
        "croma_joint",
        "croma_optical",
        "croma_sar",
        # "dofa",
        "gfmswin",
        # "prithvi"
        "remoteclip",
        "satlasnet",
        "scalemae",
        "scalemae",
        "spectralgpt",
        "ssl4eo_data2vec",
        "ssl4eo_dino",
        "ssl4eo_mae_optical",
        # "ssl4eo_mae_sar",
        # "ssl4eo_moco",
        # "unet_encoder",
    ],
)
def test_encoder_init(config_name: str) -> None:
    with initialize(version_base=None, config_path="../configs/encoder/"):
        # config is relative to a module
        encoder_config = compose(config_name=config_name)
        encoder = instantiate(encoder_config)
        assert isinstance(encoder, Encoder)


@pytest.mark.parametrize(
    "config_name",
    [
        "croma_joint",
        "croma_optical",
        "croma_sar",
        # "dofa",
        "gfmswin",
        # "prithvi"
        "remoteclip",
        "satlasnet",
        "scalemae",
        "scalemae",
        "spectralgpt",
        "ssl4eo_data2vec",
        "ssl4eo_dino",
        "ssl4eo_mae_optical",
        # "ssl4eo_mae_sar",
        # "ssl4eo_moco",
        # "unet_encoder",
    ],
)
def test_encoder_input_shape(config_name: str) -> None:
    with initialize(version_base=None, config_path="../configs/encoder/"):
        # config is relative to a module
        encoder_config = compose(config_name=config_name)
        encoder = instantiate(encoder_config)

        # fake data
        B, T = 2, 1
        data = {}
        for modality, bands in encoder.input_bands:
            n_bands = len(bands)
            H = W = encoder.input_size
            if encoder.multi_temporal:
                data[modality] = torch.randn(B, n_bands, T, H, W)
            else:
                data[modality] = torch.randn(B, n_bands, H, W)

