import logging
import math
import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from geofm_bench.datasets.base import GeoFMDataset
from geofm_bench.encoders.base import Encoder


class RichDataset(Dataset):
    def __init__(self, dataset: GeoFMDataset, encoder: Encoder):
        self.dataset = dataset
        # TODO: remove encoder for input_bands, input_size
        self.encoder = encoder

        # WARNING: Patch to overcome recursive wrapping issues
        self.data_mean = dataset.data_mean
        self.data_std = dataset.data_std
        self.data_min = dataset.data_min
        self.data_max = dataset.data_max
        self.classes = dataset.classes
        self.split = dataset.split
        self.ignore_index = dataset.ignore_index

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class SegPreprocessor(RichDataset):
    def __init__(self, dataset: GeoFMDataset, encoder: Encoder) -> None:
        super().__init__(dataset, encoder)

        self.preprocessor = {}
        self.preprocessor["optical"] = (
            BandAdaptor(dataset=dataset, encoder=encoder, modality="optical")
            if "optical" in dataset.bands.keys()
            else None
        )
        self.preprocessor["sar"] = (
            BandAdaptor(dataset=dataset, encoder=encoder, modality="sar")
            if "sar" in dataset.bands.keys()
            else None
        )
        for modality in self.encoder.input_bands:
            new_stats = self.preprocessor[modality].preprocess_band_statistics(
                self.dataset.data_mean[modality],
                self.dataset.data_std[modality],
                self.dataset.data_min[modality],
                self.dataset.data_max[modality],
            )

            self.dataset.data_mean[modality] = new_stats[0]
            self.dataset.data_std[modality] = new_stats[1]
            self.dataset.data_min[modality] = new_stats[2]
            self.dataset.data_max[modality] = new_stats[3]

    def __getitem__(self, index):
        data = self.dataset[index]

        # WARNING: k in self.encoder.input_bands is actually checking if
        # k is a modality of the encoder. Should be clearer

        for k, v in data["image"].items():
            if k in self.encoder.input_bands:
                data["image"][k] = self.preprocessor[k](v)

        data["target"] = data["target"].long()
        return data


class RegPreprocessor(SegPreprocessor):
    def __init__(self, dataset: GeoFMDataset, encoder: Encoder) -> None:
        super().__init__(dataset, encoder)

    def __getitem__(self, index):
        data = self.dataset[index]
        for k, v in data["image"].items():
            if k in self.encoder.input_bands:
                data["image"][k] = self.preprocessor[k](v)
        data["target"] = data["target"].float()
        return data


class BandAdaptor:
    def __init__(self, dataset: GeoFMDataset, encoder: Encoder, modality: str) -> None:
        """Intialize the BandAdaptor.

        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder unded.
            modality (str): image modality.
        """
        self.dataset_bands = dataset.bands[modality]
        self.input_bands = getattr(encoder.input_bands, modality, [])


        # list of length dataset_n_bands with True if the band is used in the encoder
        # and is available in the dataset
        self.used_bands_mask = torch.tensor(
            [b in self.input_bands for b in self.dataset_bands], dtype=torch.bool
        )
        # list of length encoder_n_bands with True if the band is available in the dataset
        # and used in the encoder
        self.avail_bands_mask = torch.tensor(
            [b in self.dataset_bands for b in self.input_bands], dtype=torch.bool
        )
        # list of length encoder_n_bands with the index of the band in the dataset
        # if the band is available in the dataset and -1 otherwise
        self.avail_bands_indices = torch.tensor(
            [
                self.dataset_bands.index(b) if b in self.dataset_bands else -1
                for b in self.input_bands
            ],
            dtype=torch.long,
        )

        # if the encoder requires bands that are not available in the dataset
        # then we need to pad the input with zeros
        self.need_padded = self.avail_bands_mask.sum() < len(self.input_bands)
        self.logger = logging.getLogger()
        self.logger.info(f"Adaptor for modality: {modality}")
        self.logger.info(
            "Available bands in dataset: {}".format(
                " ".join(str(b) for b in self.dataset_bands)
            )
        )
        self.logger.info(
            "Required bands in encoder: {}".format(
                " ".join(str(b) for b in self.input_bands)
            )
        )
        if self.need_padded:
            self.logger.info(
                "Unavailable bands {} are padded with zeros".format(
                    " ".join(
                        str(b)
                        for b in np.array(self.input_bands)[
                            self.avail_bands_mask.logical_not()
                        ]
                    )
                )
            )

    def preprocess_band_statistics(
        self,
        data_mean: list[float],
        data_std: list[float],
        data_min: list[float],
        data_max: list[float],
    ) -> tuple[
        list[float],
        list[float],
        list[float],
        list[float],
    ]:
        """Filter the statistics to match the available bands.

        Args:
            data_mean (list[float]): dataset mean (per band in dataset).
            data_std (list[float]): dataset std (per band in dataset).
            data_min (list[float]): dataset min (per band in dataset).
            data_max (list[float]): dataset max (per band in dataset).

        Returns:
            tuple[ list[float], list[float], list[float], list[float], ]: 
            dataset mean, std, min, max (per band in encoder). Pad with zeros
            if the band is required by the encoder but not included in the dataset.
        """
        data_mean = [
            data_mean[i] if i != -1 else 0.0 for i in self.avail_bands_indices.tolist()
        ]
        data_std = [
            data_std[i] if i != -1 else 1.0 for i in self.avail_bands_indices.tolist()
        ]
        data_min = [
            data_min[i] if i != -1 else -1.0 for i in self.avail_bands_indices.tolist()
        ]
        data_max = [
            data_max[i] if i != -1 else 1.0 for i in self.avail_bands_indices.tolist()
        ]
        return data_mean, data_std, data_min, data_max

    def preprocess_single_timeframe(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the preprocessing to a single timeframe, i.e. pad unavailable
        bands with zeros if needed to match encoder's bands.

        Args:
            image (torch.Tensor): input image of shape (dataset_n_bands H W).

        Returns:
            torch.Tensor: output image of shape (encoder_n_bands H W).
        """
        # add padding band at index 0 on the first dim
        padded_image = torch.cat([torch.zeros_like(image[0:1]), image], dim=0)
        # request all encoder's band. In self.avail_band_indices we have
        # -1 for bands not available in the dataset. So we add 1 to get the
        # correct index in the padded image (index 0 is the 0-padding band)
        return padded_image[self.avail_bands_indices + 1]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the preprocessing to the image. Pad unavailable bands with zeros.

        Args:
            image (torch.Tensor): image of shape (dataset_n_bands H W).

        Returns:
            torch.Tensor: output image of shape (encoder_n_bands T H W).
            In the case of single timeframe, T = 1.
        """
        # input of shape (dataset_n_bands T H W) output of shape (encoder_n_bands T H W)
        # WARNING: refactor this
        if len(image.shape) == 3: # (dataset_n_bands H W)
            # Add a time dimension so preprocessing can work on consistent images
            image = image.unsqueeze(1) # (dataset_n_bands H W)-> (dataset_n_bands 1 H W)

        if image.shape[1] != 1:
            final_image = []
            for i in range(image.shape[1]):
                final_image.append(self.preprocess_single_timeframe(image[:, i, :, :]))
            image = torch.stack(final_image, dim=1)
        else:
            image = self.preprocess_single_timeframe(image)

        print("OUTPUT SHAPE", image.shape)

        # OUTPUT SHAPE (encoder_n_bands T H W) (T = 1 in the case of single timeframe)
        return image


class BaseAugment(RichDataset):
    """Base class for augmentations.
    """

    def __init__(self, dataset: GeoFMDataset, encoder: Encoder) -> None:
        """Initalize the BaseAugment.

        Args:
            dataset (GeoFMDataset): dataset used.
            encoder (Encoder): encoder used.
        """
        super().__init__(dataset, encoder)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get item.
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
        raise NotImplementedError


class Tile(BaseAugment):
    def __init__(
        self, dataset: GeoFMDataset, encoder: Encoder, min_overlap: float = 0
    ) -> None:
        super().__init__(dataset, encoder)
        self.min_overlap = min_overlap
        # Should be the _largest_ image in the dataset to avoid problems mentioned in __getitem__
        self.input_size = self.dataset.img_size
        self.output_size = self.encoder.input_size
        if self.output_size == self.input_size:
            self.tiles_per_dim = 1
        elif self.output_size > self.input_size:
            raise ValueError(
                f"Can't tile inputs if dataset.img_size={self.input_size} < encoder.input_size={self.output_size}, use ResizeToEncoder instead."
            )
        elif self.min_overlap >= self.input_size:
            raise ValueError("min_overlap >= dataset.img_size")
        elif self.min_overlap >= self.input_size:
            raise ValueError("min_overlap >= encoder.input_size")
        else:
            self.tiles_per_dim = math.ceil(
                (self.input_size - self.min_overlap)
                / (self.output_size - self.min_overlap)
            )

        logging.getLogger().info(
            f"Tiling {self.input_size}x{self.input_size} input images to {self.tiles_per_dim * self.tiles_per_dim} {self.output_size}x{self.output_size} output images."
        )

        self.h_spacing_cache = [None] * super().__len__()
        self.w_spacing_cache = [None] * super().__len__()

        self.data_cache = (None, None)

    def __getitem__(self, index):
        if self.tiles_per_dim == 1:
            return self.dataset[index]

        dataset_index = math.floor(index / (self.tiles_per_dim * self.tiles_per_dim))
        data = self.dataset[dataset_index]
        # Calculate tile coordinates
        tile_index = index % (self.tiles_per_dim * self.tiles_per_dim)
        h_index = math.floor(tile_index / self.tiles_per_dim)
        w_index = tile_index % self.tiles_per_dim
        # Use the actual image size so we can handle data that's not always uniform.
        # This means that min_overlap might not always be respected.
        # Also, in case there was insufficient overlap (or tiles_per_dim=1) sepcified, we'll crop the image and lose info.
        input_h, input_w = data["image"][next(iter(data["image"].keys()))].shape[-2:]

        # Calculate the sizes of the labeled parts seperately to deal with aliasing when
        # tile spacing values are not exact integers
        if not self.h_spacing_cache[dataset_index]:
            float_spacing = np.linspace(
                0, input_h - self.output_size, self.tiles_per_dim
            )
            rounded_spacing = float_spacing.round().astype(int)
            labeled_sizes = np.ediff1d(rounded_spacing, to_end=self.output_size)
            self.h_spacing_cache[dataset_index] = (rounded_spacing, labeled_sizes)
        if not self.w_spacing_cache[dataset_index]:
            float_spacing = np.linspace(
                0, input_w - self.output_size, self.tiles_per_dim
            )
            rounded_spacing = float_spacing.round().astype(int)
            labeled_sizes = np.ediff1d(rounded_spacing, to_end=self.output_size)
            self.w_spacing_cache[dataset_index] = (rounded_spacing, labeled_sizes)

        h_positions, h_labeled_sizes = self.h_spacing_cache[dataset_index]
        w_positions, w_labeled_sizes = self.w_spacing_cache[dataset_index]

        h, w = h_positions[h_index], w_positions[w_index]
        h_labeled, w_labeled = h_labeled_sizes[h_index], w_labeled_sizes[w_index]

        tiled_data = {"image": {}, "target": None}
        tiled_data["image"] = {}
        for k, v in data["image"].items():
            if k in self.encoder.input_bands:
                tiled_data["image"][k] = v[
                    ..., h : h + self.output_size, w : w + self.output_size
                ].clone()

        # Place the mesaured part in the middle to help with tiling artefacts
        h_label_offset = round((self.output_size - h_labeled) / 2)
        w_label_offset = round((self.output_size - w_labeled) / 2)

        # Crop target to size
        tiled_data["target"] = data["target"][
            ..., h : h + self.output_size, w : w + self.output_size
        ].clone()

        # Ignore overlapping borders
        if h_index != 0:
            tiled_data["target"][..., 0:h_label_offset, :] = self.dataset.ignore_index
        if w_index != 0:
            tiled_data["target"][..., 0:w_label_offset] = self.dataset.ignore_index
        if h_index != self.tiles_per_dim - 1:
            tiled_data["target"][..., self.output_size - h_label_offset :, :] = (
                self.dataset.ignore_index
            )
        if w_index != self.tiles_per_dim - 1:
            tiled_data["target"][..., self.output_size - w_label_offset :] = (
                self.dataset.ignore_index
            )

        return tiled_data

    def __len__(self):
        return (super().__len__()) * (self.tiles_per_dim * self.tiles_per_dim)


class RandomFlip(BaseAugment):
    def __init__(
        self,
        dataset: GeoFMDataset,
        encoder: Encoder,
        ud_probability: float,
        lr_probability: float,
    ) -> None:
        super().__init__(dataset, encoder)
        self.ud_probability = ud_probability
        self.lr_probability = lr_probability

    def __getitem__(self, index):
        data = self.dataset[index]
        if random.random() < self.ud_probability:
            for k, v in data["image"].items():
                if k in self.encoder.input_bands:
                    data["image"][k] = torch.fliplr(v)
            data["target"] = torch.fliplr(data["target"])
        if random.random() < self.lr_probability:
            for k, v in data["image"].items():
                if k in self.encoder.input_bands:
                    data["image"][k] = torch.flipud(v)
            data["target"] = torch.flipud(data["target"])
        return data


class GammaAugment(BaseAugment):
    def __init__(
        self,
        dataset: GeoFMDataset,
        encoder: Encoder,
        probability: float,
        gamma_range: float,
    ) -> None:
        super().__init__(dataset, encoder)
        self.probability = probability
        self.gamma_range = gamma_range

    def __getitem__(self, index):
        data = self.dataset[index]
        # WARNING: Test this bit of code
        if random.random() < self.probability:
            for k, v in data["image"].items() and k in self.encoder.input_bands:
                data["image"][k] = torch.pow(v, random.uniform(*self.gamma_range))
        return data


class NormalizeMeanStd(BaseAugment):
    def __init__(self, dataset: GeoFMDataset, encoder: Encoder) -> None:
        super().__init__(dataset, encoder)
        self.data_mean_tensors = {}
        self.data_std_tensors = {}
        # Bands is a dict of {modality:[b1, b2, ...], ...} so it's keys are the modalaities in use
        for modality in self.encoder.input_bands:
            self.data_mean_tensors[modality] = torch.tensor(
                self.dataset.data_mean[modality]
            ).reshape((-1, 1, 1, 1))
            self.data_std_tensors[modality] = torch.tensor(
                self.dataset.data_std[modality]
            ).reshape((-1, 1, 1, 1))

    def __getitem__(self, index: int):
        data = self.dataset[index]
        for modality in self.encoder.input_bands:
            data["image"][modality] = (
                data["image"][modality] - self.data_mean_tensors[modality]
            ) / self.data_std_tensors[modality]
        return data


class NormalizeMinMax(BaseAugment):
    def __init__(
        self,
        dataset: GeoFMDataset,
        encoder: Encoder,
        data_min: torch.Tensor,
        data_max: torch.Tensor,
    ) -> None:
        super().__init__(dataset, encoder)
        self.normalizers = {}
        self.data_min_tensors = {}
        self.data_max_tensors = {}
        self.min = data_min
        self.max = data_max
        for modality in self.encoder.input_bands:
            self.data_min_tensors[modality] = torch.tensor(
                self.dataset.data_min[modality]
            ).reshape((-1, 1, 1, 1))
            self.data_max_tensors[modality] = torch.tensor(
                self.dataset.data_max[modality]
            ).reshape((-1, 1, 1, 1))

    def __getitem__(self, index: int):
        data = self.dataset[index]
        for modality in self.encoder.input_bands:
            data["image"][modality] = (
                (data["image"][modality] - self.data_min_tensors[modality])
                * (self.max - self.min)
                - self.min
            ) / self.data_max_tensors[modality]
        return data


class ColorAugmentation(BaseAugment):
    def __init__(
        self,
        dataset: GeoFMDataset,
        encoder: Encoder,
        brightness: float = 0,
        contrast: float = 0,
        clip: bool = False,
        br_probability: float = 0,
        ct_probability: float = 0,
    ) -> None:
        super().__init__(dataset, encoder)
        self.brightness = brightness
        self.contrast = contrast
        self.clip = clip
        self.br_probability = br_probability
        self.ct_probability = ct_probability

    def adjust_brightness(self, image, factor, clip_output):
        if isinstance(factor, float):
            factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
        while len(factor.shape) != len(image.shape):
            factor = factor[..., None]

        img_adjust = image + factor
        if clip_output:
            img_adjust = img_adjust.clamp(min=-1.0, max=1.0)

        return img_adjust

    def adjust_contrast(self, image, factor, clip_output):
        if isinstance(factor, float):
            factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
        while len(factor.shape) != len(image.shape):
            factor = factor[..., None]
        assert factor >= 0, "Contrast factor must be positive"

        img_adjust = image * factor
        if clip_output:
            img_adjust = img_adjust.clamp(min=-1.0, max=1.0)

        return img_adjust

    def __getitem__(self, index):
        data = self.dataset[index]
        for k, _ in data["image"].items():
            if k in self.encoder.input_bands:
                brightness = random.uniform(-self.brightness, self.brightness)
                if random.random() < self.br_probability:
                    data["image"][k] = self.adjust_brightness(
                        data["image"][k], brightness, self.clip
                    )

        for k, _ in data["image"].items():
            if k in self.encoder.input_bands:
                if random.random() < self.ct_probability:
                    contrast = random.uniform(1 - self.contrast, 1 + self.contrast)
                    data["image"][k] = self.adjust_contrast(
                        data["image"][k], contrast, self.clip
                    )

        return data


class Resize(BaseAugment):
    def __init__(self, dataset: GeoFMDataset, encoder: Encoder, size: int) -> None:
        super().__init__(dataset, encoder)
        self.size = (size, size)

    def __getitem__(self, index):
        data = self.dataset[index]
        for k, v in data["image"].items():
            if k in self.encoder.input_bands:
                data["image"][k] = T.Resize(self.size)(v)

        if data["target"].ndim == 2:
            data["target"] = data["target"].unsqueeze(0)
            data["target"] = T.Resize(
                self.size, interpolation=T.InterpolationMode.NEAREST
            )(data["target"])
            data["target"] = data["target"].squeeze(0)
        else:
            data["target"] = T.Resize(
                self.size, interpolation=T.InterpolationMode.NEAREST
            )(data["target"])

        return data


class ResizeToEncoder(Resize):
    def __init__(self, dataset: GeoFMDataset, encoder: Encoder) -> None:
        super().__init__(dataset, encoder, encoder.input_size)


class RandomCrop(BaseAugment):
    def __init__(
        self,
        dataset: GeoFMDataset,
        encoder: Encoder,
        size: int,
        padding: str | None = None,
        pad_if_needed: bool = False,
        fill: int = 0,
        padding_mode: str = "constant",
    ) -> None:
        super().__init__(dataset, encoder)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __getitem__(self, index):
        data = self.dataset[index]
        # Use the first image to determine parameters
        i, j, h, w = T.RandomCrop.get_params(
            data["image"][list(data["image"].keys())[0]],
            output_size=(self.size, self.size),
        )
        for k, v in data["image"].items():
            if k in self.encoder.input_bands:
                data["image"][k] = T.functional.crop(v, i, j, h, w)
        data["target"] = T.functional.crop(data["target"], i, j, h, w)

        return data


class RandomCropToEncoder(RandomCrop):
    def __init__(
        self,
        dataset: GeoFMDataset,
        encoder: Encoder,
        padding: str | None = None,
        pad_if_needed: bool = False,
        fill: int = 0,
        padding_mode: str = "constant",
    ) -> None:
        size = encoder.input_size
        super().__init__(
            dataset, encoder, size, padding, pad_if_needed, fill, padding_mode
        )


class ImportanceRandomCrop(BaseAugment):
    def __init__(
        self,
        dataset: GeoFMDataset,
        encoder: Encoder,
        size: int,
        padding: str | None = None,
        pad_if_needed: bool = False,
        fill: int = 0,
        padding_mode: str = "constant",
        n_crops: int = 10,
    ) -> None:
        super().__init__(dataset, encoder)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.n_crops = n_crops

    def __getitem__(self, index):
        data = self.dataset[index]

        # dataset needs to provide a weighting layer
        assert "weight" in data.keys()

        # candidates for random crop
        crop_candidates, crop_weights = [], []
        for _ in range(self.n_crops):
            i, j, h, w = T.RandomCrop.get_params(
                data["image"][
                    list(data["image"].keys())[0]
                ],  # Use the first image to determine parameters
                output_size=(self.size, self.size),
            )
            crop_candidates.append((i, j, h, w))

            crop_weight = T.functional.crop(data["weight"], i, j, h, w)
            crop_weights.append(torch.sum(crop_weight).item())

        crop_weights = np.array(crop_weights) / sum(crop_weights)
        crop_idx = np.random.choice(self.n_crops, p=crop_weights)
        i, j, h, w = crop_candidates[crop_idx]

        for k, v in data["image"].items():
            if k in self.encoder.input_bands:
                data["image"][k] = T.functional.crop(v, i, j, h, w)
        data["target"] = T.functional.crop(data["target"], i, j, h, w)

        return data


class ImportanceRandomCropToEncoder(ImportanceRandomCrop):
    def __init__(
        self,
        dataset: GeoFMDataset,
        encoder: Encoder,
        padding: str | None = None,
        pad_if_needed: bool = False,
        fill: int = 0,
        padding_mode: str = "constant",
        n_crops: int = 10,
    ) -> None:
        size = encoder.input_size
        super().__init__(
            dataset=dataset,
            encoder=encoder,
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode,
            n_crops=n_crops,
        )
