import torch
import torch.nn.functional as F

import numpy as np
import pdb

class DataPreprocessor():
    def __init__(self, args, encoder_cfg, dataset_cfg, logger):

        pass

    def __call__(self, data):
        pass



class SegPreprocessor(DataPreprocessor):

    def __init__(self, args, encoder_cfg, dataset_cfg, logger):
        super().__init__(args, encoder_cfg, dataset_cfg, logger)

        self.preprocessor = {}
        self.preprocessor['optical'] = OpticalPreprocessor(args, encoder_cfg, dataset_cfg, logger)
        # TO DO: other modalities


    def __call__(self, data):
        image = {}

        for k, v in data['image'].items():
            image[k] = self.preprocessor[k](v)

        target = data['target'].long()

        return image, target


class OpticalPreprocessor():
    def __init__(self, args, encoder_cfg, dataset_cfg, logger):
        self.dataset_bands = dataset_cfg["bands"]['optical']
        self.input_bands = encoder_cfg["input_bands"]['optical']

        self.input_size = encoder_cfg["input_size"]
        self.temporal_input = encoder_cfg["temporal_input"]

        self.used_bands_mask = torch.tensor([b in self.input_bands for b in self.dataset_bands], dtype=torch.bool)
        self.avail_bands_mask = torch.tensor([b in self.dataset_bands for b in self.input_bands], dtype=torch.bool)
        # self.used_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.input_bands else -1 for b in self.dataset_bands], dtype=torch.long)
        self.avail_bands_indices = [self.dataset_bands.index(b) if b in self.dataset_bands else None for b in self.input_bands]
        self.need_padded = self.avail_bands_mask.sum() < len(self.input_bands)

        self.logger = logger

        self.logger.info("Available bands in dataset: {}".format(' '.join(str(b) for b in self.dataset_bands)))
        self.logger.info("Required bands in encoder: {}".format(' '.join(str(b) for b in self.input_bands)))
        if self.need_padded:
            self.logger.info("Unavailable bands {} are padded with zeros".format(
                ' '.join(str(b) for b in np.array(self.input_bands)[self.avail_bands_mask.logical_not()])))


    def __call__(self, optical_image):
        # padded_image = torch.cat([torch.zeros_like(optical_image[:, 0: 1]), optical_image], dim=1)
        # optical_image = padded_image[:, self.avail_bands_indices + 1]

        # optical_image = F.interpolate(optical_image, (self.input_size, self.input_size), mode='bilinear', align_corners=False)

        # if self.temporal_input and len(optical_image.shape) == 4:
        #     optical_image = optical_image.unsqueeze(2)

        # return optical_image



        dtype = optical_image.dtype
        device = optical_image.device

        if self.temporal_input and optical_image.dim() == 4:
            optical_image = optical_image.unsqueeze(2)

        if optical_image.dim() == 4:
            B, _, H, W = optical_image.shape
            zero_tensor = torch.zeros(B, 1, H, W, dtype=dtype, device=device)
        elif optical_image.dim() == 5:  # For tensors of shape (B, C, T, H, W)
            B, _, T, H, W = optical_image.shape
            zero_tensor = torch.zeros(B, 1, T, H, W, dtype=dtype, device=device)
        else:
            raise ValueError("Unsupported tensor shape")

        out_tensors = [optical_image[:, [i], ...] if i is not None else zero_tensor for i in self.avail_bands_indices]   
        out_tensors = torch.concat(out_tensors, dim=1).to(device)

        out_tensors = F.interpolate(out_tensors, (self.input_size, self.input_size), mode='bilinear', align_corners=False)

        return out_tensors

