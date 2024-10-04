import logging
import math
import random
import numbers

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from hydra.utils import instantiate

from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple
import copy


def build_preprocessor(preprocessing_cfg, dataset_cfg, encoder_cfg):
    if preprocessing_cfg is None:
        return None
    preprocessor = InitPreprocessor(dataset_cfg, encoder_cfg)

    for preprocess in preprocessing_cfg:
        preprocessor = instantiate(
            preprocess, preprocessor=preprocessor
        )

    return preprocessor

class BasePreprocessor():
    """Base class for augmentations."""

    def __init__(self, preprocessor: Callable = None) -> None:
        if isinstance(preprocessor, BasePreprocessor):
            self.preprocessor = preprocessor

            self.dataset_img_size = copy.deepcopy(preprocessor.dataset_img_size)
            self.encoder_input_size = copy.deepcopy(preprocessor.encoder_input_size)

            self.dataset_bands = copy.deepcopy(preprocessor.dataset_bands)
            self.encoder_bands = copy.deepcopy(preprocessor.encoder_bands)
            self.in_bands = copy.deepcopy(preprocessor.out_bands)
            self.out_bands = copy.deepcopy(preprocessor.out_bands)

            self.multi_modal = copy.deepcopy(preprocessor.multi_modal)
            self.multi_temporal = copy.deepcopy(preprocessor.multi_temporal)

            self.data_mean = copy.deepcopy(preprocessor.data_mean)
            self.data_std = copy.deepcopy(preprocessor.data_std)
            self.data_min = copy.deepcopy(preprocessor.data_min)
            self.data_max = copy.deepcopy(preprocessor.data_max)

            self.class_distribution = copy.deepcopy(preprocessor.class_distribution)
            self.ignore_index = copy.deepcopy(preprocessor.ignore_index)


    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:

        if self.preprocessor is not None:
            data = self.preprocessor(data)

        return data

    def check_dimension(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]):
        for k, v in data["image"].items():
            if len(v.shape) != 4:
                raise AssertionError(f"Image dimension must be 4 (C, T, H, W), Got {str(len(v.shape))}")

        if len(data["target"].shape) != 2:
           raise AssertionError(f"Target dimension must be 2 (H, W), Got {str(len(data['target'].shape))}")

    def check_size(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]):
        base_shape = data["image"][list(data["image"].keys())[0]].shape

        for k, v in data["image"].items():
            if v.shape[1:] != base_shape[1:]:
                shape = {k: tuple(v.shape[1:]) for k, v in data["image"].items()}
                raise AssertionError(f"Image size (T, H, W) from all modalities must be equal, Got {str(shape)}")

        if base_shape[-2:] != data["target"].shape[-2:]:
            raise AssertionError(f"Image size and target size (H, W) must be equal, Got {str(tuple(base_shape[-2:]))} and {str(tuple(data['target'].shape[-2:]))}")


class InitPreprocessor(BasePreprocessor):
    """Base class for augmentations."""
    def __init__(self, dataset_cfg, encoder_cfg) -> None:
        self.dataset_img_size = dataset_cfg['img_size']
        self.encoder_input_size = encoder_cfg['input_size']

        self.dataset_bands = dataset_cfg['bands']
        self.encoder_bands = encoder_cfg['input_bands']
        self.in_bands = dataset_cfg['bands']
        self.out_bands = dataset_cfg['bands']

        self.multi_modal = dataset_cfg['multi_modal']
        self.multi_temporal = dataset_cfg['multi_temporal']

        self.data_mean = {k: torch.tensor(v) for k, v in dataset_cfg['data_mean'].items()}
        self.data_std = {k: torch.tensor(v) for k, v in dataset_cfg['data_std'].items()}
        self.data_min = {k: torch.tensor(v) for k, v in dataset_cfg['data_min'].items()}
        self.data_max = {k: torch.tensor(v) for k, v in dataset_cfg['data_max'].items()}

        self.ignore_index = dataset_cfg['ignore_index']
        self.class_distribution = dataset_cfg['distribution']

        self.preprocessor = None

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:

        self.check_dimension(data)

        return data

class BandFilter(BasePreprocessor):
    """Intialize the BandFilter.
    Args:
        dataset (GeoFMDataset): dataset used.
        encoder (Encoder): encoder used.
    """
    def __init__(
            self,
            preprocessor: BasePreprocessor,
    ) -> None:
        """Intialize the BandAdaptor.
        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
        """
        super().__init__(preprocessor)

        self.out_bands = {}

        # list of length dataset_n_bands with True if the band is used in the encoder
        # and is available in the dataset
        self.used_bands_indices = {}
        for k in self.in_bands.keys():
            if k not in self.encoder_bands.keys():
                continue
            self.used_bands_indices[k] = torch.tensor(
                [self.in_bands[k].index(b) for b in self.encoder_bands[k] if b in self.in_bands[k]], dtype=torch.long
            )
            self.out_bands[k] = [self.in_bands[k][i.item()] for i in self.used_bands_indices[k]]
            self.data_mean[k] = preprocessor.data_mean[k][self.used_bands_indices[k]]
            self.data_std[k] = preprocessor.data_std[k][self.used_bands_indices[k]]
            self.data_min[k] = preprocessor.data_min[k][self.used_bands_indices[k]]
            self.data_max[k] = preprocessor.data_max[k][self.used_bands_indices[k]]

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Filter redundant bands from the data.
        Args:
            index (int): index of data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        data = self.preprocessor(data)

        data["image"] = {k: data["image"][k][self.used_bands_indices[k]] for k in self.out_bands.keys()}

        return data


class BandPadding(BasePreprocessor):
    """Intialize the BandPadding.
    Args:
        dataset (GeoFMDataset): dataset used.
        encoder (Encoder): encoder used.
        fill_value (float): fill value for padding.
    """
    def __init__(
            self,
            preprocessor: BasePreprocessor,
            fill_value: float = 0.0
    ) -> None:
        """Intialize the BandPadding.
        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
            fill_value (float): fill value for padding.
        """
        super().__init__(preprocessor)
        self.out_bands = preprocessor.encoder_bands

        self.fill_value = fill_value

        # list of length dataset_n_bands with True if the band is used in the encoder
        # and is available in the dataset
        self.avail_bands_indices, self.used_bands_indices = {}, {}
        for k in self.encoder_bands:
            self.avail_bands_indices[k] = torch.tensor(
                [self.encoder_bands[k].index(b) for b in self.encoder_bands[k] if b in self.in_bands[k]], dtype=torch.long
            )
            self.used_bands_indices[k] = torch.tensor(
                [self.in_bands[k].index(b) for b in self.encoder_bands[k] if b in self.in_bands[k]], dtype=torch.long
            )
            size = (len(self.encoder_bands[k]),)
            self.data_mean[k] = torch.full(size, fill_value=fill_value, dtype=torch.float)
            self.data_std[k] = torch.ones(size, dtype=torch.float)
            self.data_min[k] = torch.full(size, fill_value=fill_value, dtype=torch.float)
            self.data_max[k] = torch.full(size, fill_value=fill_value, dtype=torch.float)

            self.data_mean[k][self.avail_bands_indices[k]] = preprocessor.data_mean[k][self.used_bands_indices[k]]
            self.data_std[k][self.avail_bands_indices[k]] = preprocessor.data_std[k][self.used_bands_indices[k]]
            self.data_min[k][self.avail_bands_indices[k]] = preprocessor.data_min[k][self.used_bands_indices[k]]
            self.data_max[k][self.avail_bands_indices[k]] = preprocessor.data_max[k][self.used_bands_indices[k]]

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:

        if self.preprocessor is not None:
            data = self.preprocessor(data)
        for k, v in data["image"].items():
            size = [len(self.encoder_bands[k])] + list(v.shape[1:])
            data["image"][k] = torch.full(size, fill_value=self.fill_value, dtype=v.dtype)
            data["image"][k][self.avail_bands_indices[k]] = v[self.used_bands_indices[k]]

        return data

class NormalizeMeanStd(BasePreprocessor):
    def __init__(
            self,
            preprocessor: BasePreprocessor,
            channel_dim: int = 0
    ) -> None:
        """Initialize the NormalizeMeanStd.
        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
        """
        super().__init__(preprocessor)
        self.data_mean_ = preprocessor.data_mean
        self.data_std_ = preprocessor.data_std

        self.data_mean = {k: torch.zeros_like(v) for k, v in self.data_mean_.items()}
        self.data_std = {k: torch.ones_like(v) for k, v in self.data_std_.items()}
        self.data_min = {k: (v-self.data_mean_[k])/self.data_std_[k] for k, v in preprocessor.data_min.items()}
        self.data_max = {k: (v-self.data_mean_[k])/self.data_std_[k] for k, v in preprocessor.data_max.items()}

        self.channel_dim = channel_dim

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Apply Mean/Std Normalization to the data.
        Args:
            index (int): index of data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        data = self.preprocessor(data)

        for modality in self.in_bands.keys():
            statistic_shape = [1] * data["image"][modality].dim()
            statistic_shape[self.channel_dim] = -1
            data["image"][modality].sub_(self.data_mean_[modality].view(statistic_shape)).div_(self.data_std_[modality].view(statistic_shape))
        return data


class RandomCrop(BasePreprocessor):
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        size,
        pad_if_needed: bool = False,
    ) -> None:
        """Initialize the RandomCrop augmentation.
        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
            size (int): crop size.
            padding (str | None, optional): image padding. Defaults to None.
            pad_if_needed (bool, optional): whether to pad. Defaults to False.
            fill (int, optional): value for padding. Defaults to 0.
            padding_mode (str, optional): padding mode. Defaults to "constant".
        """
        super().__init__(preprocessor)

        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(input_size: Tuple[int, int], output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            input_size (tuple): input size of the data.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = input_size
        th, tw = output_size
        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def check_pad(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> Tuple[dict[str, torch.Tensor | dict[str, torch.Tensor]], int, int]:
        _, t, height, width = data["image"][list(data["image"].keys())[0]].shape

        if height < self.size[0] or width < self.size[1]:
            pad_img = max(self.size[0] - height, 0), max(self.size[1] - width, 0)
            height, width = height + 2 * pad_img[0], width + 2 * pad_img[1]
            for k, v in data["image"].items():
                padded_img = self.preprocessor.data_mean[k].reshape(-1, 1, 1, 1).repeat(1, t, height, width)
                padded_img[:, :, pad_img[0]:-pad_img[0], pad_img[1]:-pad_img[1]] = v
                data["image"][k] = padded_img

            data["target"] = TF.pad(data["target"], padding=padded_img, fill=self.ignore_index, padding_mode='constant')

        return data, height, width

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Random crop the data.
        Args:
            index (int): index of data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        data = self.preprocessor(data)
        self.check_size(data)

        if self.pad_if_needed:
            data, height, width = self.check_pad(data)
        else:
            _, _, height, width = data["image"][list(data["image"].keys())[0]].shape

        i, j, h, w = self.get_params(
            input_size=(height, width),
            output_size=self.size,
        )

        for k, v in data["image"].items():
            data["image"][k] = TF.crop(v, i, j, h, w)

        data["target"] = TF.crop(data["target"], i, j, h, w)

        return data

class RandomCropToEncoder(RandomCrop):
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        pad_if_needed: bool = False,
    ) -> None:
        """Initialize the RandomCropToEncoder augmentation.
        Apply RandomCrop to the encoder input size.
        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
            padding (str | None, optional): image padding. Defaults to None.
            pad_if_needed (bool, optional): whether to pad or not. Defaults to False.
            fill (int, optional): value for padding. Defaults to 0.
            padding_mode (str, optional): padding mode. Defaults to "constant".
        """
        size = preprocessor.encoder_input_size
        super().__init__(
            preprocessor, size, pad_if_needed
        )


class ImportanceRandomCrop(RandomCrop):
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        size,
        pad_if_needed: bool = False,
        num_trials: int = 10,
    ) -> None:
        """Initialize the RandomCrop augmentation.
        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
            size (int): crop size.
            padding (str | None, optional): image padding. Defaults to None.
            pad_if_needed (bool, optional): whether to pad. Defaults to False.
            fill (int, optional): value for padding. Defaults to 0.
            padding_mode (str, optional): padding mode. Defaults to "constant".
        """
        super().__init__(preprocessor)

        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.pad_if_needed = pad_if_needed
        self.num_trials = num_trials
        self.class_weight = 1 / self.class_distribution


    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Random crop the data.
        Args:
            index (int): index of data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        data = self.preprocessor(data)
        self.check_size(data)

        if self.pad_if_needed:
            data, height, width = self.check_pad(data)
        else:
            _, _, height, width = data["image"][list(data["image"].keys())[0]].shape

        valid = data["target"] != self.ignore_index
        weight_map = torch.full(size=data["target"].shape, fill_value=1e-6, dtype=torch.float)
        weight_map[valid] = self.class_weight[data["target"][valid]]

        crop_candidates = [self.get_params(input_size=(height, width), output_size=self.size) for _ in range(self.num_trials)]
        crop_weights = [weight_map[c[0]:c[0]+c[2], c[1]:c[1]+c[3]].sum()/(c[2]*c[3]) for c in crop_candidates]

        crop_idx = np.random.choice(self.num_trials, p=np.asarray(crop_weights))
        i, j, h, w = crop_candidates[crop_idx]

        for k, v in data["image"].items():
            data["image"][k] = TF.crop(v, i, j, h, w)

        data["target"] = TF.crop(data["target"], i, j, h, w)

        return data


class ImportanceRandomCropToEncoder(ImportanceRandomCrop):
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        pad_if_needed: bool = False,
        num_trials: int = 10,
    ) -> None:
        """Initialize the RandomCropToEncoder augmentation.
        Apply RandomCrop to the encoder input size.
        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
            padding (str | None, optional): image padding. Defaults to None.
            pad_if_needed (bool, optional): whether to pad or not. Defaults to False.
            fill (int, optional): value for padding. Defaults to 0.
            padding_mode (str, optional): padding mode. Defaults to "constant".
        """
        size = preprocessor.encoder_input_size
        super().__init__(
            preprocessor, size, pad_if_needed, num_trials
        )

class Resize(BasePreprocessor):
    def __init__(
            self,
            preprocessor: BasePreprocessor,
            size,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias: Optional[bool] = True,
            resize_target: bool = True,
    ) -> None:
        super().__init__(preprocessor)

        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if isinstance(interpolation, int):
            interpolation = TF._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.resize_target = resize_target

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:

        data = self.preprocessor(data)

        for k, v in data["image"].items():
            data["image"][k] = TF.resize(data["image"][k], self.size, interpolation=self.interpolation, antialias=self.antialias)

        if self.resize_target:
            if torch.is_floating_point(data["target"]):
                data["target"] = TF.resize(data["target"].unsqueeze(0), size=self.size, interpolation=T.InterpolationMode.BILINEAR).squeeze(0)
            else:
                data["target"] = TF.resize(data["target"].unsqueeze(0), size=self.size, interpolation=T.InterpolationMode.NEAREST).squeeze(0)

        return data

class ResizeToEncoder(Resize):
    def __init__(self,
                 preprocessor: BasePreprocessor,
                 interpolation=T.InterpolationMode.BILINEAR,
                 antialias: Optional[bool] = True,
                 resize_target: bool = True,
                 ) -> None:
        """Initialize the ResizeToEncoder augmentation.
        Resize input data to the encoder input size.
        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
        """
        super().__init__(preprocessor, preprocessor.encoder_input_size, interpolation, antialias, resize_target)


class RandomResizedCrop(BasePreprocessor):
    def __init__(self,
                 preprocessor: BasePreprocessor,
                 size,
                 scale: Tuple[float, float] = (0.08, 1.0),
                 ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
                 interpolation=T.InterpolationMode.BILINEAR,
                 antialias: Optional[bool] = True,
                 resize_target: bool = True,
                 )-> None:
        super().__init__(preprocessor)
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = TF._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio
        self.resize_target = resize_target

    @staticmethod
    def get_params(input_size: Tuple[int, int], scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        height, width = input_size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:

        data = self.preprocessor(data)
        self.check_size(data)

        _, t, h_img, w_img = data["image"][list(data["image"].keys())[0]].shape

        i, j, h, w = self.get_params((h_img, w_img), self.scale, self.ratio)

        for k, v in data["image"].items():
            data["image"][k] = TF.resized_crop(data["image"][k], i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

        if self.resize_target:
            if torch.is_floating_point(data["target"]):
                data["target"] = TF.resized_crop(data["target"].unsqueeze(0), i, j, h, w, self.size, T.InterpolationMode.BILINEAR).squeeze(0)
            else:
                data["target"] = TF.resized_crop(data["target"].unsqueeze(0), i, j, h, w, self.size, T.InterpolationMode.NEAREST).squeeze(0)

        else:
            data["target"] = TF.crop(data["target"], i, j, h, w)

        return data

class RandomResizedCropToEncoder(RandomResizedCrop):
    def __init__(self,
                 preprocessor: BasePreprocessor,
                 scale: Tuple[float, float] = (0.08, 1.0),
                 ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
                 interpolation=T.InterpolationMode.BILINEAR,
                 antialias: Optional[bool] = True,
                 resize_target: bool = True,
                 ) -> None:
        super().__init__(preprocessor, preprocessor.encoder_input_size, scale, ratio, interpolation, antialias, resize_target)



def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size