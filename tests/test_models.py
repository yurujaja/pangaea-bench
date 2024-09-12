import unittest

import os

import torch.nn as nn

from omegaconf import OmegaConf

class testModelBuild(unittest.TestCase):
    def setUp(self):
        self.models = {
            'croma': 'configs/foundation_models/croma.yaml',
            'dofa': 'configs/foundation_models/dofa.yaml',
            'gfmswin': 'configs/foundation_models/gfmswin.yaml',
            'prithvi': 'configs/foundation_models/prithvi.yaml',
            'remoteclip': 'configs/foundation_models/remoteclip.yaml',
            'satlasnet': 'configs/foundation_models/satlasnet.yaml',
            'scalemae': 'configs/foundation_models/scalemae.yaml',
            'spectralgpt': 'configs/foundation_models/spectralgpt.yaml',
            'ssl4eo_data2vec': 'configs/foundation_models/ssl4eo_data2vec.yaml',
            'ssl4eo_dino': 'configs/foundation_models/ssl4eo_dino.yaml',
            'ssl4eo_mae': 'configs/foundation_models/ssl4eo_mae.yaml',
            'ssl4eo_moco': 'configs/foundation_models/ssl4eo_moco.yaml',
            'unet_encoder': 'configs/foundation_models/unet_encoder.yaml',
            'ssl4eo_moco': 'configs/models_config/ssl4eo_mae.yaml',
        }

    def test_download(self):
        from utils.configs import load_configs
        import foundation_models.utils
        from run import parser

        for model, config_path in self.models.items():
            mock_argv = [                          
                'run.py',
                '--config', 'configs/run/mados_prithvi.yaml',
                '--encoder_config', config_path
            ]
            with unittest.mock.patch('sys.argv', mock_argv):
                with self.subTest(model=model):
                    cfg = load_configs(parser)

                    if 'download_url' in cfg.encoder:
                        if os.path.isfile(cfg.encoder.encoder_weights):
                            os.remove(cfg.encoder.encoder_weights)
                    res = foundation_models.utils.download_model(cfg.encoder)
                    self.assertTrue(res)

    # def test_build(self):
    #     for model in self.models.keys():
    #         with self.subTest(model=model):
    #             print(f"\nTesting {model}:")
    #             cfg = {'encoder_config': self.models[model]}
    #             model_cfg = load_specific_config(cfg, 'encoder_config')

    #             model = make_encoder(model_cfg)
    #             self.assertIsInstance(model, nn.Module)
    #             del model