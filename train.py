# -*- coding: utf-8 -*-
# Obtained from:


#!/usr/bin/env python

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import os
import os.path as osp


# os.environ['CUDA_VISIBLE_DEVICES'] ="3"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets')))
from datasets.mados import MADOS, gen_weights, class_distr

import json
import random
import logging
import argparse
import numpy as np
import time 
from tqdm import tqdm
from os.path import dirname as up

import yaml

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import functional as F
# from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as T

from calflops import calculate_flops

sys.path.append(up(os.path.abspath(__file__)))
# print(up(os.path.abspath(__file__)))
sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'tasks'))
sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'models'))
# from marinext_wrapper import MariNext

# from tasks.models_vit_tensor_CD_2 import *
from tasks import upernet_vit_base
from models import prithvi_vit_base, spectral_gpt_vit_base, scale_mae_large

from utils.metrics import Evaluation
from utils.pos_embed import interpolate_pos_embed



def load_config(cfg_path):
    with open(cfg_path, "r") as file:
        return yaml.safe_load(file)


def load_checkpoint(encoder, ckpt_path, model="prithvi"):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    logging.info("Load pre-trained checkpoint from: %s" % ckpt_path)

    if model == "prithvi":
        checkpoint_model = checkpoint
        del checkpoint_model["pos_embed"]
        del checkpoint_model["decoder_pos_embed"]
    elif model in ["spectral_gpt"]:
        checkpoint_model = checkpoint["model"]
    elif model in ["scale_mae"]:
        checkpoint_model = checkpoint["model"]
        checkpoint_model = {"model."+k: v for k, v in checkpoint_model.items()}

    # print(checkpoint_model.keys())

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


def get_encoder_model(cfg, load_pretrained=True, frozen_backbone=True):
    # create model
    encoders = {
        "prithvi": prithvi_vit_base,
        "spectral_gpt": spectral_gpt_vit_base,
        "scale_mae": scale_mae_large,
    }
    encoder_name = cfg["encoder_name"]
    if encoder_name not in encoders:
        raise ValueError(f"{encoder_name} is not yet supported.")

    encoder_model_args = cfg["encoder_model_args"]
    encoder_model = encoders[encoder_name](**encoder_model_args)

    # load pretrained weights if there are any
    encoder_weights = cfg["encoder_weights"]
    if encoder_weights is not None and load_pretrained:
        msg = load_checkpoint(encoder_model, encoder_weights, encoder_name)
        print(msg)

    if frozen_backbone:
        for param in encoder_model.parameters():
            param.requires_grad = False

    return encoder_model


def create_task_model(cfg, encoder):
    models = {
        "upernet": upernet_vit_base,
    }
    model_name = cfg["task_model_name"]
    if model_name not in models:
        raise ValueError(f"{model_name} is not yet supported.")
    model_args = cfg["task_model_args"]
    model = models[model_name](encoder=encoder, **model_args)

    return model


def make_train_dataset(cfgs):
    pass


def VSCP(image, target):
    n_augmented = image.shape[0] // 2

    image_temp = image[: n_augmented * 2, :, :, :].copy()
    target_temp = target[: n_augmented * 2, :, :].copy()

    image_augmented = []
    target_augmented = []
    for i in range(n_augmented):
        image_temp[i, :, target_temp[i + n_augmented, :, :] != -1] = image_temp[
            i + n_augmented, :, target_temp[i + n_augmented, :, :] != -1
        ]
        image_augmented.append(image_temp[i, :, :].copy())

        target_temp[i, target_temp[i + n_augmented, :, :] != -1] = target_temp[
            i + n_augmented, target_temp[i + n_augmented, :, :] != -1
        ]
        target_augmented.append(target_temp[i, :, :].copy())

    image_augmented = np.stack(image_augmented)
    target_augmented = np.stack(target_augmented)

    return image_augmented, target_augmented


def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # DataLoader Workers Reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Train a downstreamtask with geospatial foundation models."
    )
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--workdir", default="./work-dir", help="the dir to save logs and models"
    )
    parser.add_argument("--path", default="data/MADOS", help="Path of the images")
    parser.add_argument(
        "--resume_from", type=str, help="load model from previous epoch"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args = vars(args)  # convert to ordinary dict

    # lr_steps list or single float
    """
    lr_steps = ast.literal_eval(args['lr_steps'])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise
        
    args['lr_steps'] = lr_steps
    """

    if not os.path.exists(args["workdir"]):
        os.makedirs(args["workdir"], exist_ok=True)

    return args


def adapt_input(
    tensor,
    size,
    bands=["a", "b"],
    type="image",
    encoder_type="spectral_gpt",
    device=torch.device("cuda"),
):
    # shape = tensor.shape
    sentinel2 = {
        "B1": 0,
        "B2": 1,
        "B3": 2,
        "B4": 3,
        "B5": 4,
        "B6": 5,
        "B7": 6,
        "B8": 7,
        "B8a": 8,
        "B9": 9,
        "B10": 10,
        "B11": 11,
        "B12": 12,
    }
    if type == "target":
        return T.resize(
            img=tensor, size=(size, size), interpolation=T.InterpolationMode.NEAREST
        )
    elif type == "image":
        if len(tensor.shape) == 4:
            Bs, C, H, W = tensor.shape
            n_tensor = T.resize(
                img=tensor,
                size=(size, size),
                interpolation=T.InterpolationMode.BILINEAR,
            ).float()
            if encoder_type in ("prithvi"):
                n_tensor = n_tensor.unsqueeze(dim=2)
                Te = 1
        elif len(tensor.shape) == 5:
            Bs, C, Te, H, W = tensor.shape
            n_tensor = torch.empty((Bs, C, Te, size, size)).to(device).float()

            for i in range(Te):
                n_tensor[:, :, i, :, :] = T.resize(
                    img=tensor[:, :, i, :, :],
                    size=(size, size),
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            if encoder_type in ("spectral_gpt"):
                n_tensor = n_tensor.squeeze()

        if len(bands) < C:
            indexes = [sentinel2[x] for x in bands]
            n_tensor = n_tensor.index_select(1, torch.LongTensor(indexes).to(device))
            #in rgb case, we should reorder them TO DO
        if len(bands) > C:
            # # ze = len(bands) - C
            if len(n_tensor.shape) == 4:
                zero_tensor = torch.zeros((Bs, (len(bands) - C), size, size)).to(device)
            elif len(n_tensor.shape) == 5:
                zero_tensor = torch.zeros((Bs, (len(bands) - C), Te, size, size)).to(
                    device
                )
            n_tensor = torch.concat((n_tensor, zero_tensor), dim=1)

        return n_tensor


def main(args):
    # Reproducibility
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)

    # Setup logging, make one log on every process with the configuration for debugging.
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"{os.path.splitext(osp.basename(args['config']))[0]}-{timestamp}"
    exp_dir = osp.join(args["workdir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = osp.join(exp_dir, f"{exp_name}.log")
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("Parsed task training parameters:")
    logging.info(json.dumps(args, indent=2))

    # Load Config file
    config = load_config(args["config"])
    logging.info("Loaded configuration:")
    logging.info(json.dumps(config, indent=2))

    encoder_cfg = config["gfm_encoder"]
    task_cfg = config["task"]

    encoder_name = encoder_cfg["encoder_name"]

    # Tensorboard
    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard", timestamp))

    splits_path = os.path.join(args["path"], "splits")

    # Construct Data loader
    dataset_train = MADOS(args["path"], splits_path, "train")
    dataset_val = MADOS(args["path"], splits_path, "train")

    dl_cfg = task_cfg["data_loader"]

    train_loader = DataLoader(
        dataset_train,
        batch_size=dl_cfg["batch"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        prefetch_factor=dl_cfg["prefetch_factor"],
        persistent_workers=dl_cfg["persistent_workers"],
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=dl_cfg["batch"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        prefetch_factor=dl_cfg["prefetch_factor"],
        persistent_workers=dl_cfg["persistent_workers"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info(f"Device used: {device}")

    # Get the encoder
    encoder = get_encoder_model(encoder_cfg, load_pretrained=True)
    encoder.to(device)

    model = create_task_model(task_cfg, encoder)
    model.to(device)

    # Load model from specific epoch to continue the training or start the evaluation
    if args["resume_from"] is not None:
        model_file = args["resume_from"] 

        checkpoint = torch.load(model_file, map_location="cpu")
        model.load_state_dict(checkpoint)
        logging.info("Load model files from: %s" % model_file)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Weighted Cross Entropy Loss & adam optimizer
    task_args = task_cfg["task_train_params"]
    weight = gen_weights(class_distr, c=task_args["weight_param"])

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-1,
        reduction="mean",
        weight=weight.to(device),
        label_smoothing=task_args["label_smoothing"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=task_args["lr"], weight_decay=task_args["decay"]
    )

    # Learning Rate scheduler
    if task_args["reduce_lr_on_plateau"] == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, task_args["lr_steps"], gamma=0.1, verbose=True
        )

    # Start training
    start_epoch = 1
    if args["resume_from"] is not None:       
        start_epoch = int(osp.splitext(osp.basename(args["resume_from"]))[0]) + 1
    epochs = task_args["epochs"]
    eval_every = task_args["eval_every"]

    # Write model-graph to Tensorboard
    if task_args["mode"] == "train":
        # Start Training!
        model.train()

        for epoch in range(start_epoch, epochs + 1):
            training_loss = []
            training_batches = 0

            i_board = 0
            for it, (image, target) in enumerate(tqdm(train_loader, desc="training")):
                it = len(train_loader) * (epoch - 1) + it  # global training iteration
                image = image.to(device)
                target = target.to(device)

                image = adapt_input(
                    tensor=image,
                    size=encoder_cfg["encoder_model_args"]["img_size"],
                    bands=encoder_cfg["encoder_train_params"]["bands"],
                    type="image",
                    encoder_type=encoder_name,
                )
                target = adapt_input(
                    tensor=target,
                    size=encoder_cfg["encoder_model_args"]["img_size"],
                    type="target",
                    encoder_type=encoder_name,
                )

                if epoch == start_epoch and it == 0:
                    flops, macs, params = calculate_flops(
                        model=model,
                        input_shape=tuple(image.size()),
                        output_as_string=True,
                        output_precision=4,
                    )
                    logging.info(
                        f"Model FLOPs:{flops}   MACs:{macs}    Params:{params}"
                    )

                optimizer.zero_grad()

                logits = model(image)
                loss = criterion(logits, target)
                loss.backward()

                training_batches += target.shape[0]

                training_loss.append((loss.data * target.shape[0]).tolist())

                if task_args["clip_grad"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), task_args["clip_grad"]
                    )

                optimizer.step()

                # Write running loss
                writer.add_scalar(
                    "training loss", loss, (epoch - 1) * len(train_loader) + i_board
                )
                i_board += 1

            logging.info(
                "Training loss was: " + str(sum(training_loss) / training_batches)
            )

            ckpt_path = os.path.join(exp_dir, "checkpoints", f"{epoch}.pth")

            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Save models to {ckpt_path}")

            # Start Evaluation
            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                val_loss = []
                val_batches = 0
                y_true_val = []
                y_predicted_val = []

                seed_all(0)

                with torch.no_grad():
                    for image, target in tqdm(val_loader, desc="validating"):
                        image = image.to(device)
                        target = target.to(device)

                        image = adapt_input(
                            tensor=image,
                            size=encoder_cfg["encoder_model_args"]["img_size"],
                            bands=encoder_cfg["encoder_train_params"]["bands"],
                            type="image",
                            encoder_type=encoder_name,
                        )
                        target = adapt_input(
                            tensor=target,
                            size=encoder_cfg["encoder_model_args"]["img_size"],
                            type="target",
                            encoder_type=encoder_name,
                        )

                        logits = model(image)
                        loss = criterion(logits, target)

                        # Accuracy metrics only on annotated pixels
                        logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
                        logits = logits.reshape((-1, task_args["output_channels"]))
                        target = target.reshape(-1)
                        mask = target != -1
                        logits = logits[mask]
                        target = target[mask]

                        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                        target = target.cpu().numpy()

                        val_batches += target.shape[0]
                        val_loss.append((loss.data * target.shape[0]).tolist())
                        y_predicted_val += probs.argmax(1).tolist()
                        y_true_val += target.tolist()

                    y_predicted_val = np.asarray(y_predicted_val)
                    y_true_val = np.asarray(y_true_val)

                    # Save Scores to the .log file and visualize also with tensorboard

                    acc_val = Evaluation(y_predicted_val, y_true_val)

                logging.info("\n")
                logging.info("Evaluating model..")
                logging.info("Val loss was: " + str(sum(val_loss) / val_batches))
                logging.info("RESULTS AFTER EPOCH " + str(epoch) + ": \n")
                logging.info("Evaluation: " + str(acc_val))

                writer.add_scalars(
                    "Loss per epoch",
                    {
                        "Val loss": sum(val_loss) / val_batches,
                        "Train loss": sum(training_loss) / training_batches,
                    },
                    epoch,
                )

                writer.add_scalar(
                    "Precision/val macroPrec", acc_val["macroPrec"], epoch
                )
                writer.add_scalar(
                    "Precision/val microPrec", acc_val["microPrec"], epoch
                )
                writer.add_scalar(
                    "Precision/val weightPrec", acc_val["weightPrec"], epoch
                )
                writer.add_scalar("Recall/val macroRec", acc_val["macroRec"], epoch)
                writer.add_scalar("Recall/val microRec", acc_val["microRec"], epoch)
                writer.add_scalar("Recall/val weightRec", acc_val["weightRec"], epoch)
                writer.add_scalar("F1/val macroF1", acc_val["macroF1"], epoch)
                writer.add_scalar("F1/val microF1", acc_val["microF1"], epoch)
                writer.add_scalar("F1/val weightF1", acc_val["weightF1"], epoch)
                writer.add_scalar("IoU/val MacroIoU", acc_val["IoU"], epoch)

                if task_args["reduce_lr_on_plateau"] == 1:
                    scheduler.step(sum(val_loss) / val_batches)
                else:
                    scheduler.step()

                model.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
