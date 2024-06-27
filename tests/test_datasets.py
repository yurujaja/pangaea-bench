import unittest

import os

from utils.configs import load_specific_config
from datasets.utils import make_dataset

class testModelBuild(unittest.TestCase):
    def setUp(self):
        # TODO should we just glob these for convinience?
        self.datasets = {
            # 'mados': 'configs/datasets_config/mados.yaml',
            'sen1floods11': 'configs/datasets_config/sen1floods11.yaml',
            # 'burn_scars': 'configs/datasets_config/burn_scars.yaml',
            # 'croptypemapping': 'configs/datasets_config/croptypemapping.yaml',
        }

    def test_download(self):
        for dataset in self.datasets.keys():
            with self.subTest(dataset=dataset):
                print(f"Testing dataset {dataset}")
                cfg = {'dataset_config': self.datasets[dataset]}
                dataset_cfg = load_specific_config(cfg, 'dataset_config')

                train_ds, val_ds, test_ds = make_dataset(dataset_cfg)