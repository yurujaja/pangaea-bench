import unittest


class testDatasetSetup(unittest.TestCase):
    def setUp(self):
        # TODO should we just glob these for convinience?
        self.datasets = {
            "ai4smallfarms": "configs/datasets/ai4smallfarms.yaml",
            "biomassters": "configs/datasets/biomassters.yaml",
            "croptypemapping": "configs/datasets/croptypemapping.yaml",
            "fivebillionpixels": "configs/datasets/fivebillionpixels.yaml",
            "hlsburnscars": "configs/datasets/hlsburnscars.yaml",
            "mados": "configs/datasets/mados.yaml",
            "sen1floods11": "configs/datasets/sen1floods11.yaml",
            "spacenet7": "configs/datasets/spacenet7.yaml",
            "spacenet7cd": "configs/datasets/spacenet7cd.yaml",
            "xview2": "configs/datasets/xview2.yaml",
        }

    def test_download(self):
        from utils.configs import load_configs
        import foundation_models.utils
        from run import parser
        from utils.registry import DATASET_REGISTRY

        for dataset in self.datasets.keys():
            for dataset, config_path in self.datasets.items():
                mock_argv = [                          
                    'run.py',
                    '--config', 'configs/run/mados_prithvi.yaml',
                    '--dataset_config', config_path
                ]
                with unittest.mock.patch('sys.argv', mock_argv):
                    with self.subTest(dataset=dataset):
                        print(f"Downloading dataset {dataset}")
                        cfg = load_configs(parser)

                        dataset = DATASET_REGISTRY.get(cfg.dataset.dataset_name)
                        dataset.download(cfg.dataset, silent=False)
                        dataset_splits = dataset.get_splits(cfg.dataset)

                        for ds in dataset_splits:
                            input = next(iter(ds))
                            self.assertTrue(input) # TODO some sanity checks here based on the config file