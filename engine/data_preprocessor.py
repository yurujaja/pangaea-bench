import torch
import torch.nn.functional as F

import numpy as np
import logging


class DataPreprocessor(torch.utils.data.Dataset):
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        # self.config = cfg.augmentation
        self.encoder_config = cfg.encoder
        self.data_config = cfg.dataset
        self.root_path = cfg.dataset.root_path
        self.data_mean = cfg.dataset.data_mean
        self.data_std = cfg.dataset.data_std
        self.classes = cfg.dataset.classes
        self.class_num = len(self.classes)
        self.split = dataset.split

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class SegPreprocessor(DataPreprocessor):
    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)

        self.preprocessor = {}
        
        self.preprocessor['optical'] = OpticalShapeAdaptor(cfg) if "optical" in cfg.dataset["bands"].keys() else None
        self.preprocessor['sar'] = SARShapeAdaptor(cfg) if "sar" in cfg.dataset["bands"].keys() else None
        # TO DO: other modalities


    def __getitem__(self, index):
        data = self.dataset[index]
        image = {}

        for k, v in data['image'].items():
            image[k] = self.preprocessor[k](v)

        target = data['target'].long()

        return image, target
    

class RegPreprocessor(SegPreprocessor):
    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)

    def __getitem__(self, index):
        data = self.dataset[index]
        image = {}

        for k, v in data['image'].items():
            image[k] = self.preprocessor[k](v)

        target = data['target'].float()

        return image, target


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
            if (self.encoder_name == "Prithvi_Encoder") and (len(optical_image.shape) == 3):
                optical_image = optical_image.unsqueeze(1)

        return optical_image

