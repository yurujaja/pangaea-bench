import torch
from torch.utils.data import Dataset, Subset
import os

class GeoFMDataset(Dataset):
    """Base class for all datasets."""

    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
    ):
        """Initializes the dataset.

        Args:
            split (str): split of the dataset (train, val, test)
            dataset_name (str): dataset name
            multi_modal (bool): whether the dataset is multi_modal
            multi_temporal (int): number of temporal frames
            root_path (str): root path of the dataset
            classes (list): dataset classes names
            num_classes (int): number of classes
            ignore_index (int): index to ignore
            img_size (int): dataset's image size
            bands (dict[str, list[str]]): bands of the dataset
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
        """
        self.split = split
        self.dataset_name = dataset_name
        self.multi_modal = multi_modal
        self.multi_temporal = multi_temporal
        self.root_path = root_path
        self.classes = classes
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.img_size = img_size
        self.bands = bands
        self.distribution = distribution
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.download_url = download_url
        self.auto_download = auto_download

        if not os.path.exists(self.root_path):
            self.download(self)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            int: length of the dataset
        """
        raise NotImplementedError

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {
                "optical": torch.Tensor of shape (C H W) (or (C T H W) if multi-temporal dataset),
                 "sar": torch.Tensor of shape (C H W) (or (C T H W) if multi-temporal dataset)
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        raise NotImplementedError

    @staticmethod
    def download(self) -> None:
        """Download the dataset.

        Raises:
            NotImplementedError: raise if the method is not implemented
        """
        raise NotImplementedError


class GeoFMSubset(Subset):
    """Custom subset class that retains dataset attributes."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        
        # Copy relevant attributes from the original dataset
        self.dataset_name = getattr(dataset, 'dataset_name', None)
        self.root_path = getattr(dataset, 'root_path', None)
        self.auto_download = getattr(dataset, 'auto_download', None)
        self.download_url = getattr(dataset, 'download_url', None)
        self.img_size = getattr(dataset, 'img_size', None)
        self.multi_temporal = getattr(dataset, 'multi_temporal', None)
        self.multi_modal = getattr(dataset, 'multi_modal', None)
        self.ignore_index = getattr(dataset, 'ignore_index', None)
        self.num_classes = getattr(dataset, 'num_classes', None)
        self.classes = getattr(dataset, 'classes', None)
        self.distribution = getattr(dataset, 'distribution', None)
        self.bands = getattr(dataset, 'bands', None)
        self.data_mean = getattr(dataset, 'data_mean', None)
        self.data_std = getattr(dataset, 'data_std', None)
        self.data_min = getattr(dataset, 'data_min', None)
        self.data_max = getattr(dataset, 'data_max', None)
        self.split = getattr(dataset, 'split', None)

    def filter_by_indices(self, indices):
        """Apply filtering by indices directly in this subset."""
        return GeoFMSubset(self.dataset, indices)

