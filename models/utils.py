import os
import urllib.request
import urllib.error
import logging

import tqdm
import gdown
import torch
from . import prithvi_vit_base, spectral_gpt_vit_base, scale_mae_large, croma, remote_clip, ssl4eo_mae, ssl4eo_dino_small, ssl4eo_moco, ssl4eo_data2vec_small, dofa_vit, gfm_swin_base, satlasnet
from . import SATLASNetWeights
from . import adapt_gfm_pretrained
from .pos_embed import interpolate_pos_embed


class DownloadProgressBar:
    def __init__(self, text="Downloading..."):
        self.pbar = None
        self.text = text
    
    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm.tqdm(desc=self.text, total=total_size, unit="b", unit_scale=True, unit_divisor=1024)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None


def download_model(model_config):
    if "download_url" in model_config and model_config["download_url"]:
        if not os.path.isfile(model_config["encoder_weights"]):
            os.makedirs("pretrained_models", exist_ok=True)

            pbar = DownloadProgressBar(f"Downloading {model_config['encoder_weights']}")

            if model_config["download_url"].startswith("https://drive.google.com/"):
                # Google drive needs some extra stuff compared to a simple file download
                gdown.download(model_config["download_url"], model_config["encoder_weights"])
            else:
                try:
                    urllib.request.urlretrieve(model_config["download_url"], model_config["encoder_weights"], pbar)
                except urllib.error.HTTPError as e:
                    print('Error while downloading model: The server couldn\'t fulfill the request.')
                    print('Error code: ', e.code)
                    return False
                except urllib.error.URLError as e:
                    print('Error while downloading model: Failed to reach a server.')
                    print('Reason: ', e.reason)
                    return False
        return True
    else:
        return False
    
    
def make_encoder(cfg, load_pretrained=True, frozen_backbone=True):
    # create model
    encoders = {
        "prithvi": prithvi_vit_base,
        "spectral_gpt": spectral_gpt_vit_base,
        "scale_mae": scale_mae_large,
        "croma": croma,
        "remote_clip": remote_clip,
        "ssl4eo_mae": ssl4eo_mae,
        "ssl4eo_dino" : ssl4eo_dino_small,
        "ssl4eo_moco" : ssl4eo_moco,
        "ssl4eo_data2vec": ssl4eo_data2vec_small,
        "dofa": dofa_vit,
        "gfm_swin": gfm_swin_base,
        "satlas_pretrain": satlasnet,
    }

    download_model(cfg)

    encoder_name = cfg["encoder_name"]
    if encoder_name not in encoders:
        raise ValueError(f"{encoder_name} is not yet supported.")
    encoder_weights = cfg["encoder_weights"]
    encoder_model_args = cfg["encoder_model_args"]
    
    if encoder_name in ["satlas_pretrain"]:
        satlas_weights_manager = SATLASNetWeights()
        encoder_model = satlas_weights_manager.get_pretrained_model(**encoder_model_args)
        cfg["embed_dim"] = encoder_model.backbone.out_channels[0][1]

    else:
        encoder_model = encoders[encoder_name](**encoder_model_args)
    
    if encoder_weights is not None and load_pretrained:
        if encoder_name in ["satlas_pretrain"]:
            pass
        else:
            logger = logging.getLogger()
            logger.info(f"Load pre-trained checkpoint from:{encoder_weights}")

            msg = encoder_model.load_pretrained(encoder_weights)
            print(msg)
            logger.info(msg)

    if frozen_backbone:
        for param in encoder_model.parameters():
            param.requires_grad = False

    return encoder_model

