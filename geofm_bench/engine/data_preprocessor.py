import logging
import math
import random

import numpy as np
import omegaconf
import torch
import torchvision.transforms as T
from torch.nn import Module
from torch.utils.data import Dataset


class RichDataset(Dataset):
    def __init__(self, dataset: Dataset, foundation_model: Module):
        self.dataset = dataset
        self.foundation_model = foundation_model

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class SegPreprocessor(RichDataset):
    def __init__(self, dataset: Dataset, foundation_model: Module) -> None:
        super().__init__(dataset, foundation_model)

        self.preprocessor = {}
        self.preprocessor["optical"] = (
            BandAdaptor(cfg, "optical") if "optical" in dataset.bands.keys() else None
        )
        self.preprocessor["sar"] = (
            BandAdaptor(cfg, "sar") if "sar" in dataset.bands.keys() else None
        )
        # TO DO: other modalities

        for modality in self.foundation_model.input_bands:
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

        for k, v in data["image"].items():
            if k in self.foundation_model.input_bands:
                data["image"][k] = self.preprocessor[k](v)

        data["target"] = data["target"].long()
        return data


class RegPreprocessor(SegPreprocessor):
    def __init__(self, dataset: Dataset, foundation_model: Module) -> None:
        super().__init__(dataset, foundation_model)

    def __getitem__(self, index):
        data = self.dataset[index]
        for k, v in data["image"].items():
            if k in self.foundation_model.input_bands:
                data["image"][k] = self.preprocessor[k](v)
        data["target"] = data["target"].float()
        return data


class BandAdaptor:
    def __init__(
        self, dataset: Dataset, foundation_model: Module, modality: str
    ) -> None:
        self.dataset_bands = dataset.bands[modality]
        self.input_bands = getattr(foundation_model.input_bands, modality, [])

        self.used_bands_mask = torch.tensor(
            [b in self.input_bands for b in self.dataset_bands], dtype=torch.bool
        )
        self.avail_bands_mask = torch.tensor(
            [b in self.dataset_bands for b in self.input_bands], dtype=torch.bool
        )
        self.avail_bands_indices = torch.tensor(
            [
                self.dataset_bands.index(b) if b in self.dataset_bands else -1
                for b in self.input_bands
            ],
            dtype=torch.long,
        )

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
        padded_image = torch.cat([torch.zeros_like(image[0:1]), image], dim=0)
        image = padded_image[self.avail_bands_indices + 1]
        return image

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) == 3:
            # Add a time dimension so preprocessing can work on consistent images
            image = image.unsqueeze(1)

        if image.shape[1] != 1:
            final_image = []
            for i in range(image.shape[1]):
                final_image.append(self.preprocess_single_timeframe(image[:, i, :, :]))
            image = torch.stack(final_image, dim=1)
        else:
            image = self.preprocess_single_timeframe(image)

        return image


class BaseAugment(RichDataset):
    """Base class for augmentations.
    __getitem__ will recieve data in CxTxHxW format from the preprocessor.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, cfg, local_cfg):
        super().__init__(dataset, cfg)
        self.ignore_modalities = getattr(local_cfg, "ignore_modalities", [])


class Tile(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.min_overlap = getattr(local_cfg, "min_overlap", 0)
        # Should be the _largest_ image in the dataset to avoid problems mentioned in __getitem__
        self.input_size = cfg.dataset.img_size
        self.output_size = cfg.encoder.input_size
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
            if k not in self.ignore_modalities and k in self.encoder_cfg.input_bands:
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
            tiled_data["target"][..., 0:h_label_offset, :] = (
                self.dataset_cfg.ignore_index
            )
        if w_index != 0:
            tiled_data["target"][..., 0:w_label_offset] = self.dataset_cfg.ignore_index
        if h_index != self.tiles_per_dim - 1:
            tiled_data["target"][..., self.output_size - h_label_offset :, :] = (
                self.dataset_cfg.ignore_index
            )
        if w_index != self.tiles_per_dim - 1:
            tiled_data["target"][..., self.output_size - w_label_offset :] = (
                self.dataset_cfg.ignore_index
            )

        return tiled_data

    def __len__(self):
        return (super().__len__()) * (self.tiles_per_dim * self.tiles_per_dim)


class RandomFlip(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.ud_probability = local_cfg.ud_probability
        self.lr_probability = local_cfg.lr_probability

    def __getitem__(self, index):
        data = self.dataset[index]
        if random.random() < self.ud_probability:
            for k, v in data["image"].items():
                if (
                    k not in self.ignore_modalities
                    and k in self.encoder_cfg.input_bands
                ):
                    data["image"][k] = torch.fliplr(v)
            data["target"] = torch.fliplr(data["target"])
        if random.random() < self.lr_probability:
            for k, v in data["image"].items():
                if (
                    k not in self.ignore_modalities
                    and k in self.encoder_cfg.input_bands
                ):
                    data["image"][k] = torch.flipud(v)
            data["target"] = torch.flipud(data["target"])
        return data


class GammaAugment(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.probability = local_cfg.probability
        self.gamma_range = local_cfg.gamma_range

    def __getitem__(self, index):
        data = self.dataset[index]
        if random.random() < self.probability:
            for k, v in data["image"].items() and k in self.encoder_cfg.input_bands:
                if k not in self.ignore_modalities:
                    data["image"][k] = torch.pow(v, random.uniform(*self.gamma_range))
        return data


class NormalizeMeanStd(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.data_mean_tensors = {}
        self.data_std_tensors = {}
        # Bands is a dict of {modality:[b1, b2, ...], ...} so it's keys are the modalaities in use
        for modality in self.encoder_cfg.input_bands:
            self.data_mean_tensors[modality] = torch.tensor(
                self.data_mean[modality]
            ).reshape((-1, 1, 1, 1))
            self.data_std_tensors[modality] = torch.tensor(
                self.data_std[modality]
            ).reshape((-1, 1, 1, 1))

    def __getitem__(self, index):
        data = self.dataset[index]
        for modality in self.encoder_cfg.input_bands:
            if modality not in self.ignore_modalities:
                data["image"][modality] = (
                    data["image"][modality] - self.data_mean_tensors[modality]
                ) / self.data_std_tensors[modality]
        return data


class NormalizeMinMax(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.normalizers = {}
        self.data_min_tensors = {}
        self.data_max_tensors = {}
        self.min = local_cfg.min
        self.max = local_cfg.max
        for modality in self.encoder_cfg.input_bands:
            self.data_min_tensors[modality] = torch.tensor(
                self.data_min[modality]
            ).reshape((-1, 1, 1, 1))
            self.data_max_tensors[modality] = torch.tensor(
                self.data_max[modality]
            ).reshape((-1, 1, 1, 1))

    def __getitem__(self, index):
        data = self.dataset[index]
        for modality in self.encoder_cfg.input_bands:
            if modality not in self.ignore_modalities:
                data["image"][modality] = (
                    (data["image"][modality] - self.data_min_tensors[modality])
                    * (self.max - self.min)
                    - self.min
                ) / self.data_max_tensors[modality]
        return data


class ColorAugmentation(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.brightness = getattr(local_cfg, "brightness", 0)
        self.contrast = getattr(local_cfg, "contrast", 0)
        self.clip = getattr(local_cfg, "clip", False)
        self.br_probability = getattr(local_cfg, "br_probability", 0)
        self.ct_probability = getattr(local_cfg, "ct_probability", 0)

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

        for k, v in data["image"].items():
            if k not in self.ignore_modalities and k in self.encoder_cfg.input_bands:
                brightness = random.uniform(-self.brightness, self.brightness)
                if random.random() < self.br_probability:
                    if k not in self.ignore_modalities:
                        data["image"][k] = self.adjust_brightness(
                            data["image"][k], brightness, self.clip
                        )

        for k, v in data["image"].items():
            if k not in self.ignore_modalities and k in self.encoder_cfg.input_bands:
                if random.random() < self.ct_probability:
                    contrast = random.uniform(1 - self.contrast, 1 + self.contrast)
                    if k not in self.ignore_modalities:
                        data["image"][k] = self.adjust_contrast(
                            data["image"][k], contrast, self.clip
                        )

        return data


class Resize(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.size = (local_cfg.size, local_cfg.size)

    def __getitem__(self, index):
        data = self.dataset[index]
        for k, v in data["image"].items():
            if k not in self.ignore_modalities and k in self.encoder_cfg.input_bands:
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
    def __init__(self, dataset, cfg, local_cfg):
        if not local_cfg:
            local_cfg = omegaconf.OmegaConf.create()
        local_cfg.size = cfg.encoder.input_size
        super().__init__(dataset, cfg, local_cfg)


class RandomCrop(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.size = local_cfg.size
        self.padding = getattr(local_cfg, "padding", None)
        self.pad_if_needed = getattr(local_cfg, "pad_if_needed", False)
        self.fill = getattr(local_cfg, "fill", 0)
        self.padding_mode = getattr(local_cfg, "padding_mode", "constant")

    def __getitem__(self, index):
        data = self.dataset[index]
        # Use the first image to determine parameters
        i, j, h, w = T.RandomCrop.get_params(
            data["image"][list(data["image"].keys())[0]],
            output_size=(self.size, self.size),
        )
        for k, v in data["image"].items():
            if k not in self.ignore_modalities and k in self.encoder_cfg.input_bands:
                data["image"][k] = T.functional.crop(v, i, j, h, w)
        data["target"] = T.functional.crop(data["target"], i, j, h, w)

        return data


class RandomCropToEncoder(RandomCrop):
    def __init__(self, dataset, cfg, local_cfg):
        if not local_cfg:
            local_cfg = omegaconf.OmegaConf.create()
        local_cfg.size = cfg.encoder.input_size
        super().__init__(dataset, cfg, local_cfg)


class ImportanceRandomCrop(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.size = local_cfg.size
        self.padding = getattr(local_cfg, "padding", None)
        self.pad_if_needed = getattr(local_cfg, "pad_if_needed", False)
        self.fill = getattr(local_cfg, "fill", 0)
        self.padding_mode = getattr(local_cfg, "padding_mode", "constant")
        self.n_crops = 10  # TODO: put this one in config

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
            if k not in self.ignore_modalities and k in self.encoder_cfg.input_bands:
                data["image"][k] = T.functional.crop(v, i, j, h, w)
        data["target"] = T.functional.crop(data["target"], i, j, h, w)

        return data


class ImportanceRandomCropToEncoder(ImportanceRandomCrop):
    def __init__(self, dataset, cfg, local_cfg):
        if not local_cfg:
            local_cfg = omegaconf.OmegaConf.create()
        local_cfg.size = cfg.encoder.input_size
        super().__init__(dataset, cfg, local_cfg)
