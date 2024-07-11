import torch
import torch.nn.functional as F

import numpy as np

class DataPreprocessor():
    def __init__(self, args, encoder_cfg, dataset_cfg, logger):

        pass

    def __call__(self, data):
        pass

        #return image, target



class SegPreprocessor(DataPreprocessor):

    def __init__(self, args, encoder_cfg, dataset_cfg, logger):
        super().__init__(args, encoder_cfg, dataset_cfg, logger)

        self.dataset_bands = np.array(dataset_cfg["bands"]['optical'])
        self.input_bands = np.array(encoder_cfg["input_bands"]['optical'])
        self.input_size = encoder_cfg["encoder_model_args"]["img_size"]
        self.temporal_input = encoder_cfg["temporal_input"]

        self.used_bands_mask = torch.tensor([b in self.input_bands for b in self.dataset_bands], dtype=torch.bool)
        self.avail_bands_mask = torch.tensor([b in self.dataset_bands for b in self.input_bands], dtype=torch.bool)
        self.need_padded = self.avail_bands_mask.sum() < len(self.input_bands)

        self.logger = logger

        self.logger.info("Available bands in dataset: {}".format(' '.join(str(b) for b in self.dataset_bands)))
        self.logger.info("Required bands in encoder: {}".format(' '.join(str(b) for b in self.input_bands)))
        if self.need_padded:
            self.logger.info("Unavailable bands {} are padded with zeros".format(' '.join(str(b) for b in self.input_bands[self.avail_bands_mask.logical_not()])))

    def __call__(self, data):
        image = data['image']['optical']#.to(self.device)

        image = image[:, self.used_bands_mask]

        image = F.interpolate(image, (self.input_size, self.input_size), mode='bilinear', align_corners=False)

        if self.need_padded:
            padded_image = torch.zeros((image.shape[0], len(self.input_bands), self.input_size, self.input_size))
            padded_image[:, self.avail_bands_mask] = image
            image = padded_image

        if self.temporal_input and len(image.shape) == 4:
            image = image.unsqueeze(2)

        target = data['target'].long()

        return image, target






