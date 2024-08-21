import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import logging

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

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

@AUGMENTER_REGISTRY.register()
class SegPreprocessor(RichDataset):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg)

        self.preprocessor = {}
        
        self.preprocessor['optical'] = OpticalShapeAdaptor(cfg) if "optical" in cfg.dataset.bands.keys() else None
        self.preprocessor['sar'] = SARShapeAdaptor(cfg) if "sar" in cfg.dataset.bands.keys() else None
        # TO DO: other modalities


    def __getitem__(self, index):
        data = self.dataset[index]

        for k, v in data['image'].items():
            data['image'][k] = self.preprocessor[k](v)

        data['target'] = data['target'].long()

        return data
    
@AUGMENTER_REGISTRY.register()
class RegPreprocessor(RichDataset):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg)

    def __getitem__(self, index):
        data = self.dataset[index]

        for k, v in data['image'].items():
            data['image'][k] = self.preprocessor[k](v)

        data['target'] = data['target'].float()

        return data


class SARShapeAdaptor():
    def __init__(self, cfg):
        self.dataset_bands = cfg.dataset.bands.sar
        self.input_bands = cfg.encoder.input_bands.sar
        self.input_size = cfg.encoder.input_size
        self.multi_temporal = cfg.dataset.multi_temporal
        self.encoder_name = cfg.encoder.encoder_name

        self.used_bands_mask = torch.tensor([b in self.input_bands for b in self.dataset_bands], dtype=torch.bool)
        self.avail_bands_mask = torch.tensor([b in self.dataset_bands for b in self.input_bands], dtype=torch.bool)
        self.avail_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.dataset_bands else -1 for b in self.input_bands], dtype=torch.long)
        self.need_padded = self.avail_bands_mask.sum() < len(self.input_bands)

        self.logger = logging.getLogger()

        self.logger.info("Available bands in dataset: {}".format(' '.join(str(b) for b in self.dataset_bands)))
        self.logger.info("Required bands in encoder: {}".format(' '.join(str(b) for b in self.input_bands)))
        if self.need_padded:
            self.logger.info("Unavailable bands {} are padded with zeros".format(
                ' '.join(str(b) for b in np.array(self.input_bands)[self.avail_bands_mask.logical_not()])))


    def preprocess_single_timeframe(self, sar_image):
        padded_image = torch.cat([torch.zeros_like(sar_image[0: 1]), sar_image], dim=0)
        sar_image = padded_image[self.avail_bands_indices + 1]
        sar_image = F.interpolate(sar_image.unsqueeze(0), (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        sar_image = sar_image.squeeze(0)
        return sar_image

    def __call__(self, sar_image):
        if self.multi_temporal:
            final_image = []
            for i in range(sar_image.shape[1]):
                final_image.append(self.preprocess_single_timeframe(sar_image[:,i,:,:]))
            sar_image = torch.stack(final_image, dim = 1)
        else:
            sar_image = self.preprocess_single_timeframe(sar_image)
            # TODO move this to the Prithvi implementation lol
            if (self.encoder_name == "Prithvi_Encoder") and (len(sar_image.shape) == 3):
                sar_image = sar_image.unsqueeze(1)

        return sar_image
    
class OpticalShapeAdaptor():
    def __init__(self, cfg):
        self.dataset_bands = cfg.dataset.bands.optical
        self.input_bands = cfg.encoder.input_bands.optical
        self.input_size = cfg.encoder.input_size
        self.multi_temporal = cfg.dataset.multi_temporal
        self.encoder_name = cfg.encoder.encoder_name

        self.used_bands_mask = torch.tensor([b in self.input_bands for b in self.dataset_bands], dtype=torch.bool)
        self.avail_bands_mask = torch.tensor([b in self.dataset_bands for b in self.input_bands], dtype=torch.bool)
        self.avail_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.dataset_bands else -1 for b in self.input_bands], dtype=torch.long)
                
        self.need_padded = self.avail_bands_mask.sum() < len(self.input_bands)

        self.logger = logging.getLogger()

        self.logger.info("Available bands in dataset: {}".format(' '.join(str(b) for b in self.dataset_bands)))
        self.logger.info("Required bands in encoder: {}".format(' '.join(str(b) for b in self.input_bands)))
        if self.need_padded:
            self.logger.info("Unavailable bands {} are padded with zeros".format(
                ' '.join(str(b) for b in np.array(self.input_bands)[self.avail_bands_mask.logical_not()])))


    def preprocess_single_timeframe(self, optical_image):
        padded_image = torch.cat([torch.zeros_like(optical_image[0: 1]), optical_image], dim=0)
        optical_image = padded_image[self.avail_bands_indices + 1]
        optical_image = F.interpolate(optical_image.unsqueeze(0), (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        optical_image = optical_image.squeeze(0)
        return optical_image

    def __call__(self, optical_image):
        if self.multi_temporal:
            final_image = []
            for i in range(optical_image.shape[1]):
                final_image.append(self.preprocess_single_timeframe(optical_image[:,i,:,:]))
            optical_image = torch.stack(final_image, dim = 1)
        else:
            optical_image = self.preprocess_single_timeframe(optical_image)
            # TODO move this to the Prithvi implementation lol
            if (self.encoder_name == "Prithvi_Encoder") and (len(optical_image.shape) == 3):
                optical_image = optical_image.unsqueeze(1)

        return optical_image

class BaseAugment(RichDataset):
    def __init__(self, dataset:torch.utils.data.Dataset, cfg, local_cfg):
        super().__init__(dataset, cfg)
        self.ignore_modalities = local_cfg.ignore_modalities

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
        self.normalizers = {}
        for modality in self.cfg.bands:
            self.normalizers[modality] = T.Normalize(mean=self.cfg.data_mean[modality], std=self.cfg.data_std[modality])

    def __getitem__(self, index):
        data = self.dataset[index]
        for modality in data['image']:
            if modality not in self.ignore_modalities:
                data['image'][modality] = self.normalizers[modality](data['image'][modality])
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
            self.data_mins[modality] = torch.tensor(self.cfg.data_min[modality])
            self.data_maxes[modality] = torch.tensor(self.cfg.data_max[modality])

    def __getitem__(self, index):
        data = self.dataset[index]
        for modality in data['image']:
            if modality not in self.ignore_modalities:
                ndims = data['image'][modality].ndim
                data_mins = self.data_mins[modality].reshape((1,) * (ndims - 3) + (-1,1,1))
                data_maxes = self.data_maxes[modality].reshape_as(data_mins)
                data['image'][modality] = ((data['image'][modality] - data_mins) * (self.max - self.min) - self.min) / data_maxes
        return data



@AUGMENTER_REGISTRY.register()
class ColorAugmentation(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.brightness = getattr(local_cfg, 'brightness', 0)
        self.contrast = getattr(local_cfg, 'contrast', 0)
        self.br_probability = getattr(local_cfg, 'br_probability', 0)
        self.ct_probability = getattr(local_cfg, 'ct_probability', 0)
    
    def adjust_brightness(self, image, factor, clip_output=True):
        if isinstance(factor, float):
            factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
        while len(factor.shape) != len(image.shape):
            factor = factor[..., None]
        
        img_adjust = image + factor
        if clip_output:
            img_adjust = img_adjust.clamp(min=0.0, max=1.0)

        return img_adjust

    def adjust_contrast(self, image, factor, clip_output=True):
        if isinstance(factor, float):
            factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
        while len(factor.shape) != len(image.shape):
            factor = factor[..., None]
        assert factor >= 0, "Contrast factor must be positive"

        img_adjust = image * factor
        if clip_output:
            img_adjust = img_adjust.clamp(min=0.0, max=1.0)

        return img_adjust

    def __getitem__(self, index):
        data = self.dataset[index]
        
        for k, v in data['image'].items():
            brightness = random.uniform(0, self.brightness)
            if random.random() < self.br_probability:
                if k not in self.ignore_modalities:
                    data['image'][k] = self.adjust_brightness(data['image'][k], brightness)
                
        for k, v in data['image'].items():
            if random.random() < self.ct_probability:
                contrast = random.uniform(0, self.contrast)
                if k not in self.ignore_modalities:
                    data['image'][k] = self.adjust_contrast(data['image'][k], contrast)
            
        return data

    
@AUGMENTER_REGISTRY.register()
class Resize(BaseAugment):
    def __init__(self, dataset, cfg, local_cfg):
        super().__init__(dataset, cfg, local_cfg)
        self.size = local_cfg.size
    
    def __getitem__(self, index):
        data = self.dataset[index]
        for k, v in data['image'].items():
            if k not in self.ignore_modalities:
                data['image'][k] = T.Resize(self.size)(v)
            data['target'] = T.Resize(self.size, interpolation = T.InterpolationMode.NEAREST)(data['target'])

        return data



# TODO: Move augmentation stuff _after_ the preprocessor (will need info on which bands we kept, and will need to move the weirdo prithvi dim to the right position)
# TODO: Train time: Random crop instead of bilinear if it would be downsampling. Should increase dataset size to have "full coverage"?
# TODO: Eval time: Crop-tile, and mark as masked on overlaps 
#       -> will this skew macro stats? I think only if the number of tiles is different per input (eg. non-uniform sized datasets).