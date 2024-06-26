import tqdm
import importlib

# Utility progress bar handler for urlretrieve
class DownloadProgressBar:
    def __init__(self):
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm.tqdm(desc="Downloading...", total=total_size, unit="b", unit_scale=True, unit_divisor=1024)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None


def make_dataset(dataset_config):
    components = dataset_config['dataset'].split('.')
    module_string = '.'.join(components[:-1])
    class_string = components[-1]
    module = importlib.import_module(module_string)
    dataset = getattr(module, class_string)

    if hasattr(dataset, 'get_splits') and callable(dataset.get_splits):
        return dataset.get_splits(dataset_config)
    else:
        raise TypeError(f"Please make sure your dataset {dataset_config['dataset']} implements a get_splits method.")
