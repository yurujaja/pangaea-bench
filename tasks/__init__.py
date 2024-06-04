#!/usr/bin/env python3

from .upernet import upernet_vit_base

upernet_vit_base = upernet_vit_base
# if __name__ == '__main__':

#     # import ..models import *

#     encoder = spectral_gpt_vit_base

#     input1 = torch.rand(2, 12, 128, 128)
#     # input1 = torch.rand(2, 12, 128, 128)


#     model = upernet_vit_base(encoder)
#     # model = UPerNet(3)
#     # model = vit_base_patch16()
#     output = model(input1)#["out"]
#     print((output.shape))