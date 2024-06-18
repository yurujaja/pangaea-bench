# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/mvrl/rshf
Modifications: modifications for compatibility with the benchmark
Authors: Yuru Jia, Valerio Marsocci
'''

import open_clip
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class RemoteCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name="ViT-B-32", embed_dim = 512, img_size = 224, **kwargs):
        super().__init__()
        self.name = "remote_clip"
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)