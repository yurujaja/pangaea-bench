import random

import math

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import logging

import omegaconf

from utils.registry import AUGMENTER_REGISTRY


class RichDataset(torch.utils.data.Dataset):
    """Torch dataset wrapper with extra information
    """
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.root_cfg = cfg
        self.cfg = cfg.dataset
        self.root_path = cfg.dataset.root_path
        self.classes = cfg.dataset.classes
        self.class_num = len(self.classes)
        self.split = dataset.split

        self.data_mean = cfg.dataset.data_mean.copy()
        self.data_std = cfg.dataset.data_std.copy()
        self.data_min = cfg.dataset.data_min.copy()
        self.data_max = cfg.dataset.data_max.copy()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


@AUGMENTER_REGISTRY.register()
class SegPreprocessor(RichDataset):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg)

        self.preprocessor = {}
        self.preprocessor['optical'] = BandAdaptor(cfg, "optical") if "optical" in cfg.dataset.bands.keys() else None
        self.preprocessor['sar'] = BandAdaptor(cfg, "sar") if "sar" in cfg.dataset.bands.keys() else None
        # TO DO: other modalities

        for modality in self.cfg.bands:
            new_stats = self.preprocessor[modality].preprocess_band_statistics(
                self.data_mean[modality],
                self.data_std[modality],
                self.data_min[modality],
                self.data_max[modality]
            )
            self.data_mean[modality], self.data_std[modality], self.data_min[modality], self.data_max[modality] = new_stats

    def __getitem__(self, index):
        data = self.dataset[index]

        for k, v in data['image'].items():
            data['image'][k] = self.preprocessor[k](v)

        data['target'] = data['target'].long()
        return data


@AUGMENTER_REGISTRY.register()
class RegPreprocessor(SegPreprocessor):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg)

    def __getitem__(self, index):
        data = self.dataset[index]

        for k, v in data['image'].items():
            data['image'][k] = self.preprocessor[k](v)

        data['target'] = data['target'].float()
        return data


class BandAdaptor():
    def __init__(self, cfg, modality):
        self.dataset_bands = cfg.dataset.bands[modality]
        self.input_bands = cfg.encoder.input_bands[modality]
        self.multi_temporal = cfg.dataset.multi_temporal
        self.encoder_name = cfg.encoder.encoder_name

        self.used_bands_mask = torch.tensor([b in self.input_bands for b in self.dataset_bands], dtype=torch.bool)
        self.avail_bands_mask = torch.tensor([b in self.dataset_bands for b in self.input_bands], dtype=torch.bool)
        self.avail_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.dataset_bands else -1 for b in self.input_bands], dtype=torch.long)

        self.need_padded = self.avail_bands_mask.sum() < len(self.input_bands)

        self.logger = logging.getLogger()

        self.logger.info(f"Adaptor for modality: {modality}")
        self.logger.info("Available bands in dataset: {}".format(' '.join(str(b) for b in self.dataset_bands)))
        self.logger.info("Required bands in encoder: {}".format(' '.join(str(b) for b in self.input_bands)))
        if self.need_padded:
            self.logger.info("Unavailable bands {} are padded with zeros".format(
                ' '.join(str(b) for b in np.array(self.input_bands)[self.avail_bands_mask.logical_not()])))

    def preprocess_band_statistics(self, data_mean, data_std, data_min, data_max):
        data_mean = [data_mean[i] if i != -1 else 0.0 for i in self.avail_bands_indices.tolist()]
        data_std = [data_std[i] if i != -1 else 1.0 for i in self.avail_bands_indices.tolist()]
        data_min = [data_min[i] if i != -1 else -1.0 for i in self.avail_bands_indices.tolist()]
        data_max = [data_max[i] if i != -1 else 1.0 for i in self.avail_bands_indices.tolist()]
        return data_mean, data_std, data_min, data_max

    def preprocess_single_timeframe(self, image):
        padded_image = torch.cat([torch.zeros_like(image[0: 1]), image], dim=0)
        image = padded_image[self.avail_bands_indices + 1]
        return image

    def __call__(self, image):
        if self.multi_temporal:
            final_image = []
            for i in range(image.shape[1]):
                final_image.append(self.preprocess_single_timeframe(image[:,i,:,:]))
            image = torch.stack(final_image, dim = 1)
        else:
            image = self.preprocess_single_timeframe(image)
            # Add a time dimension so preprocessing can work on consistent images
            if len(image.shape) == 3:
                image = image.unsqueeze(1)
        return image


class BaseAugment(RichDataset):
    """Base class for augmentations.
    __getitem__ will recieve data in CxTxHxW format from the preprocessor.
    """
    def __init__(self, dataset:torch.utils.data.Dataset, cfg, local_cfg):
        super().__init__(dataset, cfg)
        self.ignore_modalities = getattr(local_cfg, 'ignore_modalities', [])


@AUGMENTER_REGISTRY.register()
class Tile(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.min_overlap = getattr(local_cfg, "min_overlap", 0)
        self.input_size = cfg.dataset.img_size # Should be the _largest_ image in the dataset to avoid problems mentioned in __getitem__
        self.output_size = cfg.encoder.input_size
        if self.output_size == self.input_size:
            self.tiles_per_dim = 1
        elif self.output_size > self.input_size:
            raise ValueError(f"Can't tile inputs if dataset.img_size={self.input_size} < encoder.input_size={self.output_size}, use Resize instead.")
        elif self.min_overlap >= self.input_size:
            raise ValueError("min_overlap >= dataset.img_size")
        elif self.min_overlap >= self.input_size:
            raise ValueError("min_overlap >= encoder.input_size")
        else:
            self.tiles_per_dim = math.ceil((self.input_size - self.min_overlap) / (self.output_size - self.min_overlap))

        self.h_spacing_cache = [None] * super().__len__()
        self.w_spacing_cache = [None] * super().__len__()


    def __getitem__(self, index):
        dataset_index = math.floor(index / (self.tiles_per_dim ^ 2))
        data = self.dataset[dataset_index]
        # Calculate tile coordinates
        tile_index = index % (self.tiles_per_dim ^ 2)
        h_index = math.floor(tile_index / self.tiles_per_dim)
        w_index = tile_index % self.tiles_per_dim
        # Use the actual image size so we can handle data that's not always uniform.
        # This means that min_overlap might not always be respected.
        # Also, in case there was insufficient overlap (or tiles_per_dim=1) sepcified, we'll crop the image and lose info.
        input_h, input_w = data['image'][next(iter(data['image'].keys()))].shape[-2:]

        if not self.h_spacing_cache[dataset_index]:
            float_spacing = np.linspace(0, input_h - self.output_size, self.tiles_per_dim)
            rounded_spacing = float_spacing.round().astype(int)
            unique_sizes = np.ediff1d(rounded_spacing, to_end=self.output_size)
            self.h_spacing_cache[dataset_index] = (rounded_spacing, unique_sizes)
        if not self.w_spacing_cache[dataset_index]:
            float_spacing = np.linspace(0, input_w - self.output_size, self.tiles_per_dim)
            rounded_spacing = float_spacing.round().astype(int)
            unique_sizes = np.ediff1d(rounded_spacing, to_end=self.output_size)
            self.w_spacing_cache[dataset_index] = (rounded_spacing, unique_sizes)
        
        h_positions, h_unique_sizes = self.h_spacing_cache[dataset_index]
        w_positions, w_unique_sizes = self.w_spacing_cache[dataset_index]

        h, w = h_positions[h_index], w_positions[w_index]
        h_unique, w_unique = h_unique_sizes[h_index], w_unique_sizes[w_index]

        for k, v in data['image'].items():
            if k not in self.ignore_modalities:
                data['image'][k] = v[..., h:self.output_size, w:self.output_size]
        
        # Place the mesaured part in the middle to help with tiling artefacts
        h_label_offset = round((self.output_size - h_unique) / 2)
        w_label_offset = round((self.output_size - w_unique) / 2)

        # Crop target to size
        data['target'] = data['target'][..., h:self.output_size, w:self.output_size]

        # Ignore overlapping borders
        if h_index != 0:
            data['target'][..., 0:h_label_offset, :] = self.cfg.ignore_index
        if w_index != 0:
            data['target'][..., 0:w_label_offset] = self.cfg.ignore_index
        if h_index != self.tiles_per_dim - 1:
            data['target'][..., self.output_size - h_label_offset:, :] = self.cfg.ignore_index
        if w_index != self.tiles_per_dim - 1:
            data['target'][..., self.output_size - w_label_offset:] = self.cfg.ignore_index

        return data

    def __len__(self):
        return (super().__len__()) * (self.tiles_per_dim ^ 2)

@AUGMENTER_REGISTRY.register()
class RandomFlip(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.ud_probability = local_cfg.ud_probability
        self.lr_probability = local_cfg.lr_probability

    def __getitem__(self, index):
        data = self.dataset[index]
        if random.random() < self.ud_probability:
            for k, v in data['image'].items():
                if k not in self.ignore_modalities:
                    data['image'][k] = torch.fliplr(v)
            data['target'] = torch.fliplr(data['target'])
        if random.random() < self.lr_probability:
            for k, v in data['image'].items():
                if k not in self.ignore_modalities:
                    data['image'][k] = torch.flipud(v)
            data['target'] = torch.flipud(data['target'])
        return data


@AUGMENTER_REGISTRY.register()
class GammaAugment(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.probability = local_cfg.probability
        self.gamma_range = local_cfg.gamma_range

    def __getitem__(self, index):
        data = self.dataset[index]
        if random.random() < self.probability:
            for k, v in data['image'].items():
                if k not in self.ignore_modalities:
                    data['image'][k] = torch.pow(v, random.uniform(*self.gamma_range))
        return data


@AUGMENTER_REGISTRY.register()
class NormalizeMeanStd(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.means = {}
        self.stds = {}
        for modality in self.cfg.bands: # Bands is a dict of {modality:[b1, b2, ...], ...} so it's keys are the modalaities in use
            self.means = torch.tensor(self.cfg.data_mean[modality]).reshape((-1,1,1,1))
            self.stds = torch.tensor(self.cfg.data_std[modality]).reshape((-1,1,1,1))

    def __getitem__(self, index):
        data = self.dataset[index]
        for modality in data['image']:
            if modality not in self.ignore_modalities:
                data['image'][modality] = (data['image'][modality] - self.means[modality])/self.stds[modality]
        return data


@AUGMENTER_REGISTRY.register()
class NormalizeMinMax(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.normalizers = {}
        self.data_mins = {}
        self.data_maxes = {}
        self.min = local_cfg.min
        self.max = local_cfg.max
        for modality in self.cfg.bands:
            self.data_mins[modality] = torch.tensor(self.cfg.data_min[modality]).reshape((-1,1,1,1))
            self.data_maxes[modality] = torch.tensor(self.cfg.data_max[modality]).reshape((-1,1,1,1))

    def __getitem__(self, index):
        data = self.dataset[index]
        for modality in data['image']:
            if modality not in self.ignore_modalities:
                data['image'][modality] = ((data['image'][modality] - self.data_mins[modality])\
                                           * (self.max - self.min) - self.min) / self.data_maxes[modality]
        return data


@AUGMENTER_REGISTRY.register()
class ColorAugmentation(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.brightness = getattr(local_cfg, 'brightness', 0)
        self.contrast = getattr(local_cfg, 'contrast', 0)
        self.clip = getattr(local_cfg, 'clip', False)
        self.br_probability = getattr(local_cfg, 'br_probability', 0)
        self.ct_probability = getattr(local_cfg, 'ct_probability', 0)
    
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
        
        for k, v in data['image'].items():
            brightness = random.uniform(-self.brightness, self.brightness)
            if random.random() < self.br_probability:
                if k not in self.ignore_modalities:
                    data['image'][k] = self.adjust_brightness(data['image'][k], brightness, self.clip)
                
        for k, v in data['image'].items():
            if random.random() < self.ct_probability:
                contrast = random.uniform(1 - self.contrast, 1 + self.contrast)
                if k not in self.ignore_modalities:
                    data['image'][k] = self.adjust_contrast(data['image'][k], contrast, self.clip)
            
        return data


@AUGMENTER_REGISTRY.register()
class Resize(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.size = (local_cfg.size, local_cfg.size)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        for k, v in data['image'].items():
            if k not in self.ignore_modalities:
                data['image'][k] = T.Resize(self.size)(v)
        
        if data['target'].ndim == 2:
            data['target'] = data['target'].unsqueeze(0)
            data['target'] = T.Resize(self.size, interpolation = T.InterpolationMode.NEAREST)(data['target'])
            data['target'] = data['target'].squeeze(0)
        else:
            data['target'] = T.Resize(self.size, interpolation = T.InterpolationMode.NEAREST)(data['target'])

        return data


@AUGMENTER_REGISTRY.register()
class RandomCrop(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.size = local_cfg.size
        self.padding = getattr(local_cfg, 'padding', None)
        self.pad_if_needed = getattr(local_cfg, 'pad_if_needed', False)
        self.fill = getattr(local_cfg, 'fill', 0)
        self.padding_mode = getattr(local_cfg, 'padding_mode', 'constant')

    def __getitem__(self, index):
        data = self.dataset[index]
        i, j, h, w = T.RandomCrop.get_params(
            data['image'][list(data['image'].keys())[0]],  # Use the first image to determine parameters
            output_size=(self.size, self.size)
        )
        for k, v in data['image'].items():
            if k not in self.ignore_modalities:
                data['image'][k] = T.functional.crop(v, i, j, h, w)
        data['target'] = T.functional.crop(data['target'], i, j, h, w)

        return data


@AUGMENTER_REGISTRY.register()
class RandomCropToEncoder(RandomCrop):
    def __init__(self, dataset, cfg, local_cfg):
        if not local_cfg:
            local_cfg = omegaconf.OmegaConf.create()
        local_cfg.size = cfg.encoder.input_size
        super().__init__(dataset, cfg, local_cfg)