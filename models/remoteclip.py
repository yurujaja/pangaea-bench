# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/ChenDelong1999/RemoteCLIP
Modifications: modifications of 'pool_type' config for compatibility with the benchmark
Authors: Yuru Jia
'''

import open_clip
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class RemoteCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name="ViT-B-32", embed_dim = 512, **kwargs):
        super().__init__()
        self.name = "remote_clip"
        self.embed_dim = embed_dim
        support_modes = {
            "ViT-B-32": {
                      "image_size": 224,
                      "layers": 12,
                      "width": 768,
                      "patch_size": 32,
                      "pool_type": "none"
                      }, 
            "ViT-L-14": {
                      "image_size": 224,
                      "layers": 24,
                      "width": 1024,
                      "patch_size": 14,
                      "pool_type": "none"
                      },
        }
        if model_name not in support_modes:
            raise ValueError(f"model_name should be one of {support_modes.keys()}")
        vision_cfg = support_modes[model_name]
        self.img_size = vision_cfg["image_size"]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                                                    model_name, 
                                                    vision_cfg=vision_cfg)
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
