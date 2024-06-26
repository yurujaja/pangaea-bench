import unittest

import torch.nn as nn

from utils.configs import load_specific_config
from train import get_encoder_model

class testModelBuild(unittest.TestCase):
    def setUp(self):
        self.models = {
            'ssl4eo_moco': 'configs/models_config/ssl4eo_mae.yaml',
            'aa_encoder_config': 'configs/models_config/aa_encoder_config.yaml',
            'croma': 'configs/models_config/croma.yaml',
            'dofa': 'configs/models_config/dofa.yaml',
            'gfm': 'configs/models_config/gfm.yaml',
            'prithvi': 'configs/models_config/prithvi.yaml',
            'remoteclip': 'configs/models_config/remoteclip.yaml',
            'satlasnet': 'configs/models_config/satlasnet.yaml',
            'scale_mae': 'configs/models_config/scale_mae.yaml',
            'spectral_gpt': 'configs/models_config/spectral_gpt.yaml',
            'ssl4eo_data2vec': 'configs/models_config/ssl4eo_data2vec.yaml',
            'ssl4eo_dino': 'configs/models_config/ssl4eo_dino.yaml',
            'ssl4eo_mae': 'configs/models_config/ssl4eo_mae.yaml',
            'ssl4eo_moco': 'configs/models_config/ssl4eo_moco.yaml'
        }

    def test_build(self):
        for model in self.models.keys():
            with self.subTest(model=model):
                print(f"\nTesting {model}:")
                cfg = {'encoder_config': self.models[model]}
                model_cfg = load_specific_config(cfg, 'encoder_config')
                model = get_encoder_model(model_cfg)

                self.assertIsInstance(model, nn.Module)

                del model