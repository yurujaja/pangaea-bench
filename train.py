# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/gkakogeorgiou/mados
Modifications: support different datasets, models, and tasks
Authors: Yuru Jia, Valerio Marsocci
'''

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
from torchvision.transforms import functional as T

import ptflops

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'tasks'))
sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'models'))

from tasks import upernet_vit_base, cd_vit
import models
from models import prithvi_vit_base, spectral_gpt_vit_base, scale_mae_large, croma, remote_clip, ssl4eo_mae, ssl4eo_dino_small, ssl4eo_moco, ssl4eo_data2vec_small, gfm_swin_base, dofa_vit, satlasnet
from models import SATLASNetWeights

from models import adapt_gfm_pretrained
from utils.metrics import Evaluation
from models.pos_embed import interpolate_pos_embed
from utils.make_datasets import make_dataset


def load_config(args):
    cfg_path = args["run_config"]
    with open(cfg_path, "r") as file:
        train_config = yaml.safe_load(file)
    
    def load_specific_config(key):
        if args.get(key):
            with open(args[key], "r") as file:
                return yaml.safe_load(file)
        elif train_config.get(key):
            with open(train_config[key], "r") as file:
                return yaml.safe_load(file)
        else:
            raise ValueError(f"No configuration found for {key}")

    encoder_config = load_specific_config("encoder_config")
    dataset_config = load_specific_config("dataset_config") 
    task_config = load_specific_config("task_config")


    return train_config, encoder_config, dataset_config, task_config


def load_checkpoint(encoder, ckpt_path, model="prithvi"):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    logging.info("Load pre-trained checkpoint from: %s" % ckpt_path)

    # print(checkpoint.keys())

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
        logging.info("Loading pretrained model is already done when initializing the model.")

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

    models.utils.download_model(cfg)

    encoder_name = cfg["encoder_name"]
    if encoder_name not in encoders:
        raise ValueError(f"{encoder_name} is not yet supported.")
    encoder_weights = cfg["encoder_weights"]
    encoder_model_args = cfg["encoder_model_args"]
    
    if encoder_name in ["satlas_pretrain"]:
        satlas_weights_manager = SATLASNetWeights()
        encoder_model = satlas_weights_manager.get_pretrained_model(**encoder_model_args)
        # encoder_model_args["weights"] = torch.load(encoder_weights, map_location="cpu") if encoder_weights is not None else None
        # encoder_model = encoders[encoder_name](**encoder_model_args)
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


def create_task_model(task_cfg, encoder_cfg, encoder):
    models = {
        "upernet": upernet_vit_base,
        "cd_vit": cd_vit, 
    }
    model_name = task_cfg["task_model_name"]
    model_args = task_cfg["task_model_args"]
    if model_name not in models:
        raise ValueError(f"{model_name} is not yet supported.")

    if encoder_cfg["encoder_name"] in ['satlas_pretrain']:
        encoder_cfg["embed_dim"] = encoder.backbone.out_channels[0][1]

    model = models[model_name](encoder=encoder, **model_args)

    return model


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
    parser.add_argument("run_config", help="train config file path")
    parser.add_argument("--dataset_config", help="train config file path")
    parser.add_argument("--encoder_config", help="train config file path")
    parser.add_argument("--task_config", help="train config file path")
    parser.add_argument(
        "--workdir", default="./work-dir", help="the dir to save logs and models"
    )
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
    input,
    size,
    source_modal,
    target_modal,
    encoder_type="spectral_gpt",
    device=torch.device("cuda"),
):
    
    def adapt_input_tensor(
        tensor,
        size,
        source_bands,
        target_bands,
        encoder_type="spectral_gpt",
        device=torch.device("cuda"),
    ):   
              
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

        # Adapt from an arbitrary list of source bands to an arbitrary list of target bands
        # by moving the matching parts to the right place, and filling out the rest with zeros.
        if len(n_tensor.shape) == 4:
            zero_tensor = torch.zeros((Bs, 1, size, size)).to(device)
        elif len(n_tensor.shape) == 5:
            zero_tensor = torch.zeros((Bs, 1, Te, size, size)).to(
                device
            )

        source_band_indexes = [source_bands.index(t) if t in source_bands else None for t in target_bands]    
        out_tensors = [n_tensor[:, [i], ...] if i is not None else zero_tensor for i in source_band_indexes]
            
        return torch.concat(out_tensors, dim=1).to(device)
    
    # TODO: to support croma and dofa multi-modality

    tensor = input['optical'].to(device)
    source_bands = source_modal['optical']
    target_bands = target_modal['optical']
    return adapt_input_tensor(tensor, size, source_bands, target_bands, encoder_type, device)

    '''
    if encoder_type not in ["dofa", "croma"]:
        tensor = input['s2'].to(device)
        source_bands = source_modal['s2']
        target_bands = target_modal['s2']

        return adapt_input_tensor(tensor, size, source_bands, target_bands, encoder_type, device)
    else:
        output = []
        for modal in ["s1", "s2"]:
            # TODO: to support croma and dofa multi-modality
            if modal not in input:
                continue
            tensor = input[modal].to(device)
            source_bands = source_modal[modal]
            target_bands = target_modal[modal]
            input[modal] = adapt_input_tensor(tensor, size, source_bands, target_bands, encoder_type, device)
            output.append(input[modal])
        return output
    '''


def adapt_target(tensor, size, device=torch.device("cuda")):
    tensor = tensor.to(device)
    return T.resize(
        img=tensor, size=(size, size), interpolation=T.InterpolationMode.NEAREST
    ).squeeze(dim=1).long()


def main(args):
    # Reproducibility
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)


    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # Load Config file
    train_cfg, encoder_cfg, dataset_cfg, task_cfg = load_config(args)
    
    encoder_name = encoder_cfg["encoder_name"]
    dataset_name = dataset_cfg["dataset_name"]
    task_name = task_cfg["task_model_name"]

    exp_name = f"{encoder_name}-{dataset_name}-{task_name}-{timestamp}"
    exp_dir = osp.join(args["workdir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = osp.join(exp_dir, f"{exp_name}.log")
    # Checkpoints save path
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    # Tensorboard
    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard", timestamp))
    
    # Logging
    logging.basicConfig(
        filename=log_file,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("Parsed running parameters:")
    logging.info(json.dumps(args, indent=2))
    for name, config in {
        "training": train_cfg,
        "encoder": encoder_cfg,
        "dataset": dataset_cfg,
        "task": task_cfg,
    }.items():
        logging.info(f"Loaded {name} configuration:")
        logging.info(json.dumps(config, indent=2))   

    # Construct Data loader
    dataset_train, dataset_val, dataset_test = make_dataset(
                                            dataset_cfg["dataset_name"], 
                                            dataset_cfg["data_path"])
    dl_cfg = dataset_cfg["data_loader"]
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

    model = create_task_model(task_cfg, encoder_cfg, encoder)
    model.to(device)

    # input1 = torch.randn((2, 12, 128, 128)).to(device)
    # # input2 = torch.randn((2, 3, 128, 128)).to(device)
    # output = model(input1) #, input2)
    # print(output.shape)
    # sys.exit("FINE TEST")

    # Load model from specific epoch to continue the training or start the evaluation
    if args["resume_from"]:
        resume_file = args["resume_from"]
    elif train_cfg["resume_from"]:
        resume_file = train_cfg["resume_from"]
    else:
        resume_file = None

    if resume_file:
        checkpoint = torch.load(resume_file, map_location="cpu")
        model.load_state_dict(checkpoint)
        logging.info("Load model files from: %s" % resume_file)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Weighted Cross Entropy Loss & adam optimizer
    weight = gen_weights(class_distr, c=train_cfg["weight_param"])

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-1,
        reduction="mean",
        # weight=weight.to(device),
        # label_smoothing=task_args["label_smoothing"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["decay"]
    )

    # Learning Rate scheduler
    if train_cfg["reduce_lr_on_plateau"] == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, train_cfg["lr_steps"], gamma=0.1, verbose=True
        )

    # Start training
    start_epoch = 1
    if args["resume_from"] is not None:       
        start_epoch = int(osp.splitext(osp.basename(args["resume_from"]))[0]) + 1
    epochs = train_cfg["epochs"]
    eval_every = train_cfg["eval_every"]

    img_size = encoder_cfg["encoder_model_args"]["img_size"] if encoder_cfg["encoder_model_args"].get("img_size") else dataset_cfg["img_size"]

    # Write model-graph to Tensorboard
    if train_cfg["mode"] == "train":
        # Start Training!
        model.train()

        for epoch in range(start_epoch, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            training_loss = []
            training_batches = 0

            i_board = 0
            for it, data in enumerate(tqdm(train_loader, desc="training")):
                it = len(train_loader) * (epoch - 1) + it  # global training iteration
                image = data['image']
                target = data['target']

                image = adapt_input(
                    input=image,
                    size=img_size,
                    source_modal=dataset_cfg["bands"],
                    target_modal=encoder_cfg["input_bands"],
                    encoder_type=encoder_name,
                    device=device,
                )

                target = adapt_target(
                    tensor=target,
                    size=img_size,
                    device=device
                )   
                if epoch == start_epoch and it == 0:
                    macs, params = ptflops.get_model_complexity_info(
                        model=model,
                        input_res=tuple(image.size()[1:]),
                        as_strings=True, backend='pytorch',
                        verbose=True
                    )
                    logging.info(f"Model MACs: {macs}")
                    logging.info(f"Model Params: {params}")
                    print(f"Model MACs:{macs}")
                    print(f"Model Params:{params}")
                optimizer.zero_grad()

                logits = model(image)#.squeeze(dim=1)
                # print(logits.shape, target.shape)
                # print(logits.dtype, target.dtype)
                loss = criterion(logits, target)
                loss.backward()

                training_batches += target.shape[0]

                training_loss.append((loss.data * target.shape[0]).tolist())

                if train_cfg["clip_grad"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_cfg["clip_grad"]
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
                    for data in tqdm(val_loader, desc="validating"):
                        image = data['image']
                        target = data['target']

                        image = adapt_input(
                            input=image,
                            size=img_size,
                            source_modal=dataset_cfg["bands"],
                            target_modal=encoder_cfg["input_bands"],
                            encoder_type=encoder_name,
                            device=device,
                        )

                        target = adapt_target(
                            tensor=target,
                            size=img_size,
                            device=device
                        )

                        logits = model(image)
                        loss = criterion(logits, target)

                        # Accuracy metrics only on annotated pixels
                        logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
                        logits = logits.reshape((-1, dataset_cfg["num_classes"]))
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

                if train_cfg["reduce_lr_on_plateau"] == 1:
                    scheduler.step(sum(val_loss) / val_batches)
                else:
                    scheduler.step()

                model.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
