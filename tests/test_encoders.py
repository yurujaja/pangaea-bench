from hydra import compose, initialize
from hydra.utils import instantiate
import pytest

from geofm_bench.encoders.base import Encoder

# TODO: include dofa which depends on dataset.bands
@pytest.mark.parametrize("config_name", ["croma_joint", "croma_optical", "croma_sar", "scalemae", "gfmswin"])
def test_encoder_init(config_name: str) -> None:
    with initialize(version_base=None, config_path="../configs/encoder/"):
        # config is relative to a module
        encoder_config = compose(config_name=config_name)
        encoder = instantiate(encoder_config)
        print(encoder)
        assert isinstance(encoder, Encoder)