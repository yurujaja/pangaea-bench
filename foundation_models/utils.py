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
    

def load_checkpoint(encoder, ckpt_path, model="prithvi"):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    logger = logging.getLogger()
    logger.info("Load pre-trained checkpoint from: %s" % ckpt_path)

    if model in ["prithvi", "remote_clip", "dofa"]:
        checkpoint_model = checkpoint
        if model == "prithvi":
            del checkpoint_model["pos_embed"]
            del checkpoint_model["decoder_pos_embed"]
        elif model in ["remote_clip"]:
            checkpoint_model = {"model."+k: v for k, v in checkpoint_model.items()}
    elif model in ["croma"]:
        if encoder.modality in ("optical"):
            checkpoint_model = checkpoint["s2_encoder"]
            checkpoint_model = {"s2_encoder."+k: v for k, v in checkpoint_model.items()}
        elif encoder.modality in ("SAR"):
            checkpoint_model = checkpoint["s1_encoder"]
            checkpoint_model = {"s1_encoder."+k: v for k, v in checkpoint_model.items()}
        elif encoder.modality == "joint":
            checkpoint_model = checkpoint
            checkpoint_model_joint = {"cross_encoder."+k: v for k, v in checkpoint_model["joint_encoder"].items()}
            checkpoint_model_s1 = {"s1_encoder."+k: v for k, v in checkpoint_model["s1_encoder"].items()}
            checkpoint_model_s2 = {"s2_encoder."+k: v for k, v in checkpoint_model["s2_encoder"].items()}
            checkpoint_model = {**checkpoint_model_s2, **checkpoint_model_s1, **checkpoint_model_joint}
    elif model in ["spectral_gpt", "ssl4eo_data2vec", "ssl4eo_mae"]:
        checkpoint_model = checkpoint["model"]
    elif model in ["scale_mae"]:
        checkpoint_model = checkpoint["model"]
        checkpoint_model = {"model."+k: v for k, v in checkpoint_model.items()}
    elif model in ["ssl4eo_moco"]:
        checkpoint_model = checkpoint["state_dict"]
        checkpoint_model = {k.replace("module.base_encoder.",""): v for k, v in checkpoint_model.items()}
    elif model in ["ssl4eo_dino"]:
        checkpoint_model = checkpoint["teacher"]
        checkpoint_model = {k.replace("backbone.",""): v for k, v in checkpoint_model.items()}
    elif model in ["gfm_swin"]:
        checkpoint_model = adapt_gfm_pretrained(encoder, checkpoint)
    elif model in ["satlas_pretrain"]:
        logger.info("Loading pretrained model is already done when initializing the model.")


    state_dict = encoder.state_dict()

    if model == "spectral_gpt":
        interpolate_pos_embed(encoder, checkpoint_model)
        for k in [
            "patch_embed.0.proj.weight",
            "patch_embed.1.proj.weight",
            "patch_embed.2.proj.weight",
            "patch_embed.2.proj.bias",
            "head.weight",
            "head.bias",
            "pos_embed_spatial",
            "pos_embed_temporal",
        ]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    msg = encoder.load_state_dict(checkpoint_model, strict=False)
    return msg


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
            msg = load_checkpoint(encoder_model, encoder_weights, encoder_name)
            print(msg)

    if frozen_backbone:
        for param in encoder_model.parameters():
            param.requires_grad = False

    return encoder_model

