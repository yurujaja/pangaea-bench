import os
import zipfile
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score

from sustainbench_dataset import SustainBenchDataset


NUM_CLASSES = 9

# GRID_SIZE = { 'ghana': 256,
#               'southsudan': 256}

CROPS = ['Banana', 'Beans', 'Cassava', 'Groundnut', 'Maize', 'Non-Crop',
         'Other', 'Sugar Cane', 'Sweet Potatoes']

CROP_LABELS = {}
for i, crop in enumerate(CROPS):
    CROP_LABELS[crop] = i

REGIONS = ['Bungoma', 'Busia', 'Siaya']


class CropTypeMappingKenyaDataset(SustainBenchDataset):
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
    _dataset_name = 'crop_type_kenya'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://drive.google.com/drive/folders/1Rq1F-ys-rkjftAUnbdGJ1T6udsZWnEyo?usp=sharing',
            'compressed_size': None}}
    _metadata_array = np.array([[1]])
    # _split_scheme = 

    def __init__(self, version=None, root_dir='data', download=False, training_set='random', test_set='random'):
        """
        Args:
            training_set: split data randomly, or set one region as the train/val set (ex training_set='Bungoma') and the other two as test.
        """
        self.training_set = training_set
        self.test_set = test_set

        self._version = version
        # print(root_dir)
        # self._data_dir = self.initialize_data_dir(root_dir, download)
        self._data_dir = root_dir

        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

        # Extract splits
        if self.training_set not in REGIONS + ['random']:
            raise ValueError(f'Training set {self.training_set} not recognized')
        elif self.training_set == 'random':
            self.split = 'fold_random'
        else:
            self.split = 'fold_' + self.training_set.lower() + '_test'

        # with zipfile.ZipFile('croptype_mapping_kenya/crop_type_kenya.zip', 'r') as zip_ref:
            # zip_ref.extractall(self._data_dir)
        split_df = pd.read_csv(os.path.join(self._data_dir, 'crop_type_kenya_2017_metadata.csv'))
        self._split_array = split_df[self.split].values

        # y_array stores idx of ids corresponding to location.
        # y_npys npz files of features corresponding to labels.
        # y_labels are actual y labels..
        self._y_array = torch.from_numpy(split_df['fieldID'].values)
        self._y_npys = split_df['fileName'].values
        # print(split_df['cropType'].replace(CROP_LABELS).unique())
        self._y_labels = torch.from_numpy(split_df['cropType'].replace(CROP_LABELS).values)

        super().__init__(root_dir, download, test_set)

    def __getitem__(self, idx):
        # Any transformations are handled by the SustainBenchSubset
        # since different subsets (e.g., train vs test) might have different transforms
        print(idx)
        x = self.get_input(idx)
        y = self.get_label(idx)
        return x, y

    def get_input(self, idx):
        """
        Returns X for a given idx.
        """
        # print(self._y_array -1)
        # print(((self._y_array -1) == idx))
        path = self._y_npys[idx]
        # print(self._y_npys[idx])
        # path = self._y_npys[((self._y_array) == str(idx)).nonzero(as_tuple=True)[0]]
        input = np.load(self.data_dir+"timeseries/"+path, allow_pickle=True)
        input = torch.from_numpy(input)

        return {'input': input}

    def get_label(self, idx):
        """
        Returns y for a given idx.
        """
        # path = self._y_npys[idx]
        a = self._y_labels[idx]
        return a

    def crop_segmentation_metrics(self, y_true, y_pred):
        y_true = y_true.int()
        y_pred = y_pred.int()
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        print('Macro Dice/ F1 score:', f1)
        print('Accuracy score:', acc)
        return f1, acc

    def eval(self, y_pred, y_true):
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

if __name__ == "__main__":
    test_data = CropTypeMappingKenyaDataset(root_dir = "/geomatics/gpuserver-0/vmarsocci/croptype_mapping_kenya/") #, download = True)
    print(next(iter(test_data)))
