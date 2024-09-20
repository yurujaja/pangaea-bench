import torch
import torch.nn as nn

class Base_Encoder(nn.Module):
    def __init__(self):
        super().__init__()


    def load_encoder_weights(self, pretrained_path):
        pretrained_model = torch.load(pretrained_path, map_location="cpu")
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        msg = self.load_state_dict(pretrained_encoder, strict=False)

        return missing, incompatible_shape


    def freeze(self):
       for param in self.parameters():
           param.requires_grad = False

    def forward(self):
        pass