# Adapted from: https://github.com/sustainlab-group/sustainbench.git
# Modifications:
# - Allow for loading data directly from data_dir argument
# Author: Yuru Jia

import os
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import torch
import torchvision.transforms as transforms


class SustainBenchDataset:
    """
    Shared dataset class for all SustainBench datasets.
    Each data point in the dataset is an (x, y, metadata) tuple, where:
    - x is the input features
    - y is the target
    - metadata is a vector of relevant information, e.g., domain.
      For convenience, metadata also contains y.
    """
    DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
    DEFAULT_SPLIT_NAMES = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
    # ADD TO THIS as more dataloaders are written or as more data is added to the drive folder
    GOOGLE_DRIVE_DATASETS = {'poverty', 'africa_crop_type_mapping', 'crop_type_kenya', 'crop_yield', 'brick_kiln'}

    def __init__(self, root_dir, download, split_scheme):
        if len(self._metadata_array.shape) == 1:
            self._metadata_array = self._metadata_array.unsqueeze(1)
        self.check_init()

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the SustainBenchSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        if isinstance(self.y_array[0], Path):  # edited to fit output which are also images
            # dataset has images as an output
            y = self.get_output_image(self.y_array[idx])  # a list of image path name
        else:
            y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        return self.dataset[self.indices[idx]]

    #def eval(self, y_pred, y_true, metadata):
        """
        Args:
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
    #    raise NotImplementedError

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (SustainBenchSubset): A (potentially subsampled) subset of the SustainBenchDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = SustainBenchSubset(self, split_idx, transform)
        return subset

    def check_init(self):
        """
        Convenience function to check that the SustainBenchDataset is properly configured.
        """
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'SustainBenchDataset is missing {attr_name}.'

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Check splits
        assert self.split_dict.keys()==self.split_names.keys()
        assert 'train' in self.split_dict
        assert 'val' in self.split_dict

        ## Check that required arrays are Tensors # edited
        # assert isinstance(self.y_array, torch.Tensor), 'y_array must be a torch.Tensor'
        # assert isinstance(self.metadata_array, torch.Tensor), 'metadata_array must be a torch.Tensor'

        # Check that dimensions match
        assert len(self.y_array) == len(self.metadata_array)
        assert len(self.split_array) == len(self.metadata_array)

        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]
        # For convenience, include y in metadata_fields if y_size == 1
        if self.y_size == 1:
            assert 'y' in self.metadata_fields

    @property
    def latest_version(cls):
        def is_later(u, v):
            """Returns true if u is a later version than v."""
            u_major, u_minor = tuple(map(int, u.split('.')))
            v_major, v_minor = tuple(map(int, v.split('.')))
            if (u_major > v_major) or (
                (u_major == v_major) and (u_minor > v_minor)):
                return True
            else:
                return False

        latest_version = '0.0'
        for key in cls.versions_dict.keys():
            if is_later(key, latest_version):
                latest_version = key
        return latest_version

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset, e.g., 'amazon', 'camelyon17'.
        """
        return self._dataset_name

    @property
    def version(self):
        """
        A string that identifies the dataset version, e.g., '1.0'.
        """
        if self._version is None:
            return self.latest_version
        else:
            return self._version

    @property
    def versions_dict(self):
        """
        A dictionary where each key is a version string (e.g., '1.0')
        and each value is a dictionary containing the 'download_url' and
        'compressed_size' keys.
        'download_url' is the URL for downloading the dataset archive.
        If None, the dataset cannot be downloaded automatically
        (e.g., because it first requires accepting a usage agreement).
        'compressed_size' is the approximate size of the compressed dataset in bytes.
        """
        return self._versions_dict

    @property
    def data_dir(self):
        """
        The full path to the folder in which the dataset is stored.
        """
        return self._data_dir

    @property
    def collate(self):
        """
        Torch function to collate items in a batch.
        By default returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)

    @property
    def split_scheme(self):
        """
        A string identifier of how the split is constructed,
        e.g., 'standard', 'in-dist', 'user', etc.
        """
        return self._split_scheme

    @property
    def split_dict(self):
        """
        A dictionary mapping splits to integer identifiers (used in split_array),
        e.g., {'train': 0, 'val': 1, 'test': 2}.
        Keys should match up with split_names.
        """
        return getattr(self, '_split_dict', SustainBenchDataset.DEFAULT_SPLITS)

    @property
    def split_names(self):
        """
        A dictionary mapping splits to their pretty names,
        e.g., {'train': 'Train', 'val': 'Validation', 'test': 'Test'}.
        Keys should match up with split_dict.
        """
        return getattr(self, '_split_names', SustainBenchDataset.DEFAULT_SPLIT_NAMES)

    @property
    def split_array(self):
        """
        An array of integers, with split_array[i] representing what split the i-th data point
        belongs to.
        """
        return self._split_array

    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def y_size(self):
        """
        The number of dimensions/elements in the target, i.e., len(y_array[i]).
        For standard classification/regression tasks, y_size = 1.
        For multi-task or structured prediction settings, y_size > 1.
        Used for logging and to configure models to produce appropriately-sized output.
        """
        return self._y_size

    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)

    @property
    def is_classification(self):
        """
        Boolean. True if the task is classification, and false otherwise.
        Used for logging purposes.
        """
        return (self.n_classes is not None)

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array

    @property
    def metadata_map(self):
        """
        An optional dictionary that, for each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        For example, if we have
            metadata_fields = ['hospital', 'y']
            metadata_map = {'hospital': ['East', 'West']}
        then if metadata_array[i, 0] == 0, the i-th data point belongs to the 'East' hospital
        while if metadata_array[i, 0] == 1, it belongs to the 'West' hospital.
        """
        return getattr(self, '_metadata_map', None)

    @property
    def original_resolution(self):
        """
        Original image resolution for image datasets.
        """
        return getattr(self, '_original_resolution', None)

    def initialize_data_dir(self, root_dir, download):
        """
        Helper function for downloading/updating the dataset if required.
        Note that we only do a version check for datasets where the download_url is set.
        Currently, this includes all datasets except Yelp.
        Datasets for which we don't control the download, like Yelp,
        might not handle versions similarly.
        """
        if self.version not in self.versions_dict:
            raise ValueError(f'Version {self.version} not supported. Must be in {self.versions_dict.keys()}.')

        download_url = self.versions_dict[self.version]['download_url']

        os.makedirs(root_dir, exist_ok=True)

        data_dir = os.path.join(root_dir, f'{self.dataset_name}')
        version_file = os.path.join(data_dir, f'RELEASE_v{self.version}.txt')
        current_major_version, current_minor_version = tuple(map(int, self.version.split('.')))

        # Check if we specified the latest version. Otherwise, print a warning.
        latest_major_version, latest_minor_version = tuple(map(int, self.latest_version.split('.')))
        if latest_major_version > current_major_version:
            print(
                f'*****************************\n'
                f'{self.dataset_name} has been updated to version {self.latest_version}.\n'
                f'You are currently using version {self.version}.\n'
                f'We highly recommend updating the dataset by not specifying the older version in the command-line argument or dataset constructor.\n'
                f'See https://sustainlab-group.github.io/sustainbench for changes.\n'
                f'*****************************\n')
        elif latest_minor_version > current_minor_version:
            print(
                f'*****************************\n'
                f'{self.dataset_name} has been updated to version {self.latest_version}.\n'
                f'You are currently using version {self.version}.\n'
                f'Please consider updating the dataset.\n'
                f'See https://sustainlab-group.github.io/sustainbench for changes.\n'
                f'*****************************\n')

        # If the data_dir exists and contains the right RELEASE file,
        # we assume the dataset is correctly set up
        # if os.path.exists(data_dir) and os.path.exists(version_file):
        if os.path.exists(data_dir):
            return data_dir

        # If the data_dir exists and does not contain the right RELEASE file, but it is not empty and the download_url is not set,
        # we assume the dataset is correctly set up
        if ((os.path.exists(data_dir)) and
            (len(os.listdir(data_dir)) > 0) and
            (download_url is None)):
            return data_dir

        # Otherwise, we assume the dataset needs to be downloaded.
        if download == False and (not os.path.isdir(data_dir)):
            if download_url is None:
                raise FileNotFoundError(f'The {self.dataset_name} dataset could not be found in {data_dir}. {self.dataset_name} cannot be automatically downloaded. Please download it manually.')
            else:
                raise FileNotFoundError(f'The {self.dataset_name} dataset could not be found in {data_dir}. Initialize the dataset with download=True to download the dataset. If you are using the example script, run with --download. This might take some time for large datasets.')

        # Otherwise, proceed with downloading.
        if download_url is None:
            raise ValueError(f'Sorry, {self.dataset_name} cannot be automatically downloaded. Please download it manually.')

        from sustainbench.datasets.download_utils import  extract_archive #download_and_extract_archive
        import gdown
        print(f'Downloading dataset to {data_dir}...')

        # download Sustainbench data
        if self.dataset_name not in self.GOOGLE_DRIVE_DATASETS:
            print(f'You can also download the dataset manually at https://drive.google.com/drive/folders/1jyjK5sKGYegfHDjuVBSxCoj49TD830wL.')
            try:
                output_name = os.path.join("data", "archive.zip")
                gdown.download(download_url, output_name, quiet=True, use_cookies=False)
                extract_archive(output_name, "data", remove_finished=False)
                print("Data downloaded at {}".format(data_dir))
            except Exception as e:
                print("Unable to download data")


        # DHS data
        elif self.dataset_name == 'poverty':
            print(f'Downloading from Google Drive...')
            #for google_drive_dict in download_url:
            #    url = google_drive_dict['url']
            #    compressed_size = google_drive_dict['size']
            #    gdown.download_folder(url, quiet=True, use_cookies=False)
            try:
                gdown.download_folder(download_url, quiet=True, use_cookies=False)
            except Exception as e:
                print(f"\n{os.path.join(data_dir, 'archive.tar.gz')} may be corrupted. Please try deleting it and rerunning this command.\n")
                print(f"Exception: ", e)

        elif self.dataset_name == 'brick_kiln':
            print(f'Downloading from Google Drive...')
            try:
                gdown.download_folder(download_url, quiet=True, use_cookies=False)
                extract_archive('brick_kiln/brick_kiln_v1.0.tar.gz', root_dir, remove_finished=True)
                print(f"Data downloaded at data/brick_kiln_v1.0")
            except Exception as e:
                print(f"\n'brick_kiln/brick_kiln_v1.0.tar.gz' may be corrupted. Please try deleting it and rerunning this command.\n")
                print(f"Exception: ", e)
            return os.path.join(root_dir, 'brick_kiln_v1.0')

        # download from Google drive
        else:
            print(f'Downloading from Google Drive...')
            #for google_drive_dict in download_url:
            #    url = google_drive_dict['url']
            #    compressed_size = google_drive_dict['size']
            #    gdown.download_folder(url, quiet=True, use_cookies=False)
            try:
                gdown.download_folder(download_url, quiet=True, use_cookies=False)
            except Exception as e:
                print(f"\n{os.path.join(data_dir, 'archive.tar.gz')} may be corrupted. Please try deleting it and rerunning this command.\n")
                print(f"Exception: ", e)


        return data_dir

    @staticmethod
    def standard_eval(metric, y_pred, y_true):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results = {
            **metric.compute(y_pred, y_true),
        }
        results_str = (
            f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        )
        return results, results_str

    @staticmethod
    def standard_group_eval(metric, grouper, y_pred, y_true, metadata, aggregate=True):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - grouper (CombinatorialGrouper): Grouper object that converts metadata into groups
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results, results_str = {}, ''
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        g = grouper.metadata_to_group(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups)
        for group_idx in range(grouper.n_groups):
            group_str = grouper.group_field_str(group_idx)
            group_metric = group_results[metric.group_metric_field(group_idx)]
            group_counts = group_results[metric.group_count_field(group_idx)]
            results[f'{metric.name}_{group_str}'] = group_metric
            results[f'count_{group_str}'] = group_counts
            if group_results[metric.group_count_field(group_idx)] == 0:
                continue
            results_str += (
                f'  {grouper.group_str(group_idx)}  '
                f"[n = {group_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {group_results[metric.group_metric_field(group_idx)]:5.3f}\n")
        results[f'{metric.worst_group_metric_field}'] = group_results[f'{metric.worst_group_metric_field}']
        results_str += f"Worst-group {metric.name}: {group_results[metric.worst_group_metric_field]:.3f}\n"
        return results, results_str


class SustainBenchSubset(SustainBenchDataset):
    def __init__(self, dataset, indices, transform):
        """
        This acts like torch.utils.data.Subset, but on SustainBenchDatasets.
        We pass in transform explicitly because it can potentially vary at
        training vs. test time, if we're using data augmentation.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_scheme', '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y #, metadata

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)


BANDS = { 's1': { 'VV': 0, 'VH': 1, 'RATIO': 2},
          's2': { '10': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'RDED1': 3, 'RDED2': 4, 'RDED3': 5, 'NIR': 6, 'RDED4': 7, 'SWIR1': 8, 'SWIR2': 9},
                   '4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}},
          'planet': { '4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}}}

MEANS = { 's1': { 'ghana': torch.Tensor([-10.50, -17.24, 1.17]),
                  'southsudan': torch.Tensor([-9.02, -15.26, 1.15])},
          's2': { 'ghana': torch.Tensor([2620.00, 2519.89, 2630.31, 2739.81, 3225.22, 3562.64, 3356.57, 3788.05, 2915.40, 2102.65]),
                  'southsudan': torch.Tensor([2119.15, 2061.95, 2127.71, 2277.60, 2784.21, 3088.40, 2939.33, 3308.03, 2597.14, 1834.81])},
          'planet': { 'ghana': torch.Tensor([1264.81, 1255.25, 1271.10, 2033.22]),
                      'southsudan': torch.Tensor([1091.30, 1092.23, 1029.28, 2137.77])},
          's2_cldfltr': { 'ghana': torch.Tensor([1362.68, 1317.62, 1410.74, 1580.05, 2066.06, 2373.60, 2254.70, 2629.11, 2597.50, 1818.43]),
                  'southsudan': torch.Tensor([1137.58, 1127.62, 1173.28, 1341.70, 1877.70, 2180.27, 2072.11, 2427.68, 2308.98, 1544.26])} }

STDS = { 's1': { 'ghana': torch.Tensor([3.57, 4.86, 5.60]),
                 'southsudan': torch.Tensor([4.49, 6.68, 21.75])},
         's2': { 'ghana': torch.Tensor([2171.62, 2085.69, 2174.37, 2084.56, 2058.97, 2117.31, 1988.70, 2099.78, 1209.48, 918.19]),
                 'southsudan': torch.Tensor([2113.41, 2026.64, 2126.10, 2093.35, 2066.81, 2114.85, 2049.70, 2111.51, 1320.97, 1029.58])},
         'planet': { 'ghana': torch.Tensor([602.51, 598.66, 637.06, 966.27]),
                     'southsudan': torch.Tensor([526.06, 517.05, 543.74, 1022.14])},
         's2_cldfltr': { 'ghana': torch.Tensor([511.19, 495.87, 591.44, 590.27, 745.81, 882.05, 811.14, 959.09, 964.64, 809.53]),
                 'southsudan': torch.Tensor([548.64, 547.45, 660.28, 677.55, 896.28, 1066.91, 1006.01, 1173.19, 1167.74, 865.42])} }

# OTHER PER COUNTRY CONSTANTS
NUM_CLASSES = { 'ghana': 4,
                'southsudan': 4}

GRID_SIZE = { 'ghana': 256,
              'southsudan': 256}

CM_LABELS = { 'ghana': [0, 1, 2, 3],
              'southsudan': [0, 1, 2, 3]}

CROPS = { 'ghana': ['groundnut', 'maize', 'rice', 'soya bean'],
          'southsudan': ['sorghum', 'maize', 'rice', 'groundnut']}

IMG_DIM = 64

PLANET_DIM = 212


class CropTypeMappingDataset(SustainBenchDataset):
    """
    Supported `split_scheme`:
        'official' - same as 'ghana'
        'south-sudan'

    Input (x):
        List of three satellites, each containing C x 64 x 64 x T satellite image,
        with 12 channels from S2, 2 channels from S1, and 6 from Planet.
        Additional bands such as NDVI and GCVI are computed for Planet and S2.
        For S1, VH/VV is also computed. Time series are zero padded to 256.
        Mean/std applied on bands excluding NDVI and GCVI. Paper uses 32x32
        imagery but the public dataset/splits use 64x64 imagery, which is
        thusly provided. Bands are as follows:

        S1 - [VV, VH, RATIO]
        S2 - [BLUE, GREEN, RED, RDED1, RDED2, RDED3, NIR, RDED4, SWIR1, SWIR2, NDVI, GCVI]
        PLANET - [BLUE, GREEN, RED, NIR, NDVI, GCVI]

    Output (y):
        y is a 64x64 tensor with numbers for locations with a crop class.

    Metadata:
        Metadata contains integer in format {Year}{Month}{Day} for each image in
        respective time series. Zero padded to 256, can be used to derive a mask.

    Website: https://github.com/roserustowicz/crop-type-mapping

    Original publication:
    @InProceedings{Rustowicz_2019_CVPR_Workshops,
        author = {M Rustowicz, Rose and Cheong, Robin and Wang, Lijing and Ermon, Stefano and Burke, Marshall and Lobell, David},
        title = {Semantic Segmentation of Crop Type in Africa: A Novel Dataset and Analysis of Deep Learning Methods},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month = {June},
        year = {2019}
    }

    License:
        S1/S2 data is U.S. Public Domain.

    """
    _dataset_name = 'africa_crop_type_mapping_v1.0'
    _versions_dict = {  # TODO
        '1.0': {
            'download_url': 'https://drive.google.com/drive/folders/1WhVObtFOzYFiXBsbbrEGy1DUtv7ov7wF?usp=sharing',
            'compressed_size': None}}

    def __init__(self, version=None, data_dir=None, root_dir='data', download=False, split_scheme='official',
                 resize_planet=False, calculate_bands=True, normalize=True):
        """
        Args:
            resize_planet: True if Planet imagery will be resized to 64x64
            calculate_bands: True if aditional bands (NDVI and GCVI) will be calculated on the fly and appended
            normalize: True if bands (excluding NDVI and GCVI) wll be normalized
        """
        self._resize_planet = resize_planet
        self._calculate_bands = calculate_bands
        self._normalize = normalize

        self._version = version

        if data_dir is not None:
            self._data_dir = data_dir
        else:
            self._data_dir = self.initialize_data_dir(root_dir, download)

        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme not in ['official', 'ghana', 'southsudan']:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        if self._split_scheme in ['official', 'ghana']:
            self._country = 'ghana'
        if self._split_scheme in ['southsudan']:
            self._country = 'southsudan'

        split_df = pd.read_csv(os.path.join(self.data_dir, self._country, 'list_eval_partition.csv'))
        self._split_array = split_df['partition'].values

        # y_array stores idx ids corresponding to location. Actual y labels are
        # tensors that are loaded separately.
        self._y_array = torch.from_numpy(split_df['id'].values)
        self._y_size = (IMG_DIM, IMG_DIM)

        self._metadata_fields = ['y']
        self._metadata_array = torch.from_numpy(split_df['id'].values)

        super().__init__(root_dir, download, split_scheme)

    def __getitem__(self, idx):
        # Any transformations are handled by the SustainBenchSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.get_label(idx)
        metadata = self.get_metadata(idx)

        output = {
            'image': {
                'optical': x,
            },
            'target': y,
            'metadata': metadata
        }
        
        return output
        


    def pad(self, tensor):
        '''
        Right pads or crops tensor to GRID_SIZE.
        '''
        pad_size = GRID_SIZE[self.country] - tensor.shape[-1]
        tensor = torch.nn.functional.pad(input=tensor, pad=(0, pad_size), value=0)
        return tensor

    def get_input(self, idx):
        """
        Returns X for a given idx.
        """
        loc_id = f'{self.y_array[idx]:06d}'
        images = np.load(os.path.join(self.data_dir, self.country, 'npy', f'{self.country}_{loc_id}.npz'))

        s1 = images['s1']
        s2 = images['s2']
        planet = images['planet']

        s1 = torch.from_numpy(s1)
        s2 = torch.from_numpy(s2.astype(np.int32))
        planet = torch.from_numpy(planet.astype(np.int32))

        if self.resize_planet:
            planet = planet.permute(3, 0, 1, 2)
            planet = transforms.Resize(IMG_DIM)(planet)
            planet = planet.permute(1, 2, 3, 0)
        else:
            planet = planet.permute(3, 0, 1, 2)
            planet = transforms.CenterCrop(PLANET_DIM)(planet)
            planet = planet.permute(1, 2, 3, 0)

        # Include NDVI and GCVI for s2 and planet, calculate before normalization and numband selection
        if self.calculate_bands:
            ndvi_s2 = (s2[BANDS['s2']['10']['NIR']] - s2[BANDS['s2']['10']['RED']]) / (s2[BANDS['s2']['10']['NIR']] + s2[BANDS['s2']['10']['RED']])
            ndvi_planet = (planet[BANDS['planet']['4']['NIR']] - planet[BANDS['planet']['4']['RED']]) / (planet[BANDS['planet']['4']['NIR']] + planet[BANDS['planet']['4']['RED']])

            gcvi_s2 = (s2[BANDS['s2']['10']['NIR']] / s2[BANDS['s2']['10']['GREEN']]) - 1
            gcvi_planet = (planet[BANDS['planet']['4']['NIR']] / planet[BANDS['planet']['4']['GREEN']]) - 1

        if self.normalize:
            s1 = self.normalization(s1, 's1')
            s2 = self.normalization(s2, 's2')
            planet = self.normalization(planet, 'planet')

        # Concatenate calculated bands
        if self.calculate_bands:
            s2 = torch.cat((s2, torch.unsqueeze(ndvi_s2, 0), torch.unsqueeze(gcvi_s2, 0)), 0)
            planet = torch.cat((planet, torch.unsqueeze(ndvi_planet, 0), torch.unsqueeze(gcvi_planet, 0)), 0)

        s1 = self.pad(s1)
        s2 = self.pad(s2)
        planet = self.pad(planet)

        return {'s1': s1, 's2': s2, 'planet': planet}

    def get_label(self, idx):
        """
        Returns y for a given idx.
        """
        loc_id = f'{self.y_array[idx]:06d}'
        label = np.load(os.path.join(self.data_dir, self.country, 'truth', f'{self.country}_{loc_id}.npz'))['truth']
        label = torch.from_numpy(label)
        return label

    def get_dates(self, json_file):
        """
        Converts json dates into tensor containing dates
        """
        dates = np.array(json_file['dates'])
        dates = np.char.replace(dates, '-', '')
        dates = torch.from_numpy(dates.astype(np.int))
        return dates

    def get_metadata(self, idx):
        """
        Returns metadata for a given idx.
        Dates are returned as integers in format {Year}{Month}{Day}
        """
        loc_id = f'{self.y_array[idx]:06d}'

        s1_json = json.loads(open(os.path.join(self.data_dir, self.country, 's1', f's1_{self.country}_{loc_id}.json'), 'r').read())
        s1 = self.get_dates(s1_json)

        s2_json = json.loads(open(os.path.join(self.data_dir, self.country, 's2', f's2_{self.country}_{loc_id}.json'), 'r').read())
        s2 = self.get_dates(s2_json)

        planet_json = json.loads(open(os.path.join(self.data_dir, self.country, 'planet', f'planet_{self.country}_{loc_id}.json'), 'r').read())
        planet = self.get_dates(planet_json)

        s1 = self.pad(s1)
        s2 = self.pad(s2)
        planet = self.pad(planet)

        return {'s1': s1, 's2': s2, 'planet': planet}


    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (SustainBenchSubset): A (potentially subsampled) subset of the SustainBenchDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = SustainBenchSubset(self, split_idx, transform)
        return subset

    def normalization(self, grid, satellite):
        """ Normalization based on values defined in constants.py
        Args:
          grid - (tensor) grid to be normalized
          satellite - (str) describes source that grid is from ("s1" or "s2")
        Returns:
          grid - (tensor) a normalized version of the input grid
        """
        num_bands = grid.shape[0]
        means = MEANS[satellite][self.country]
        stds = STDS[satellite][self.country]
        grid = (grid-means[:num_bands].reshape(num_bands, 1, 1, 1))/stds[:num_bands].reshape(num_bands, 1, 1, 1)

        if satellite not in ['s1', 's2', 'planet']:
            raise ValueError("Incorrect normalization parameters")
        return grid

    def crop_segmentation_metrics(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert (y_true.shape == y_pred.shape)
        y_true = y_true.int()
        y_pred = y_pred.int()
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        print('Macro Dice/ F1 score:', f1)
        print('Accuracy score:', acc)
        return f1, acc

    def eval(self, y_pred, y_true, metadata):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model
            - y_true (Tensor): Ground-truth values
            - metadata (Tensor): Metadata
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        f1, acc = self.crop_segmentation_metrics(y_true, y_pred)
        results = [f1, acc]
        results_str = f'Dice/ F1 score: {f1}, Accuracy score: {acc}'
        return results, results_str

    @property
    def normalize(self):
        """
        True if images will be normalized.
        """
        return self._normalize

    @property
    def country(self):
        """
        String containing the country pertaining to the dataset.
        """
        return self._country

    @property
    def resize_planet(self):
        """
        True if planet satellite imagery will be resized to other satellite sizes.
        """
        return self._resize_planet

    @property
    def calculate_bands(self):
        """
        True if aditional bands (NDVI and GCVI) will be calculated on the fly and appended
        """
        return self._calculate_bands
    
'''
# test CropTypeMappingDataset
if __name__=="__main__":
    from tqdm import tqdm
    import pdb
    
    dataset = CropTypeMappingDataset(root_dir='/geomatics/gpuserver-0/yjia/', split_scheme='southsudan')
    train_dataset = dataset.get_subset('train')  # a SustainBenchSubset object
    train_dataset.get_label(0) # this would be wrong because SustainBenchSubset does not have get_label method
    pdb.set_trace()
    # val_dataset = dataset.get_subset('val')
    # test_dataset = dataset.get_subset('test')
    
    # To get the number of pixels in each class
    labels_num = {}
    label_ration = {}
    for i in range(len(dataset)):
        
        label = dataset.get_label(i)
        label_unique = label.unique().tolist()
        #print(label_unique)
        for j in label_unique:
            if j not in labels_num:
                labels_num[j] = 0
            labels_num[j] += torch.sum(label == j).item()
    
    # remove the background class
    labels_num.pop(0)

    # To get the ratio of each class
    total = sum(labels_num.values())
    for key in labels_num:
        label_ration[key] = round(labels_num[key] / total * 100)
    # print it out in human readable format with new line
    for key in label_ration:
        print(f'{key}: {label_ration[key]}%')


    print('ratio for the train dataset')
    labels_num_train = {}
    label_ratio_train = {}
    for i in tqdm(range(len(train_dataset))):

        label = train_dataset[i][1]
        #print(label.shape, label.dtype)
        label_unique_train = label.unique().tolist()
        #print(label_unique)
        for j in label_unique_train:
            if j not in labels_num_train:
                labels_num_train[j] = 0
            labels_num_train[j] += torch.sum(label == j).item()
    
    # remove the background class
    labels_num_train.pop(0)

    # To get the ratio of each class
    total = sum(labels_num_train.values())
    for key in labels_num_train:
        label_ratio_train[key] = round(labels_num[key] / total * 100)
    # print it out in human readable format with new line
    for key in label_ratio_train:
        print(f'{key}: {label_ration[key]}%')


'''
