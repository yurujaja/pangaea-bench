import open_clip
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class RemoteCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name="ViT-B-32", **kwargs):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)