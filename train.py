# -*- coding: utf-8 -*-
''' 
Modifications: support different datasets, models, and tasks
Authors: Yuru Jia, Valerio Marsocci
'''

import sys
import os
import os.path as osp


from datasets.mados import MADOS, gen_weights, class_distr

import json
import random

import argparse
import numpy as np
import time 
from tqdm import tqdm
from os.path import dirname as up

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import functional as F


import ptflops

from utils.logger import setup_logger

from utils.metrics import Evaluation

from datasets.utils import make_dataset
from tasks.utils import make_task
from models.utils import make_encoder
from utils.configs import load_config
import pdb


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
    parser.add_argument(
        "--ckpt_path", type=str, help="load the checkpoint for evaluation"
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



def main(args):
    # Reproducibility
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)


    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    # Load Config file
    train_cfg, encoder_cfg, dataset_cfg, task_cfg = load_config(args)
    
    encoder_name = encoder_cfg["encoder_name"].split('.')[-1]
    dataset_name = dataset_cfg["dataset"].split('.')[-1]
    task_name = task_cfg["task"].split('.')[-1]

    exp_name = f"{encoder_name}-{dataset_name}-{task_name}-{train_cfg['mode']}-{timestamp}"
    exp_dir = osp.join(args["workdir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = osp.join(exp_dir, f"{exp_name}.log")
    # Checkpoints save path
    if train_cfg["mode"] == "train":
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True) 
    # Tensorboard
    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard", timestamp))
    
    # Setup Logging
    logger = setup_logger(log_file)

    logger.info("Parsed running parameters:")
    logger.info(json.dumps(args, indent=2))
    for name, config in {
        "training": train_cfg,
        "encoder": encoder_cfg,
        "dataset": dataset_cfg,
        "task": task_cfg,
    }.items():
        logger.info(f"Loaded {name} configuration:")
        logger.info(json.dumps(config, indent=2))   
    
    # Construct Data loader
    dataset_train, dataset_val, dataset_test = make_dataset(dataset_cfg, logger)
    dl_cfg = dataset_cfg["data_loader"]
    
    if train_cfg["mode"] == "train":
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
    elif train_cfg["mode"] == "test":
        test_loader = DataLoader(
            dataset_test,
            batch_size=dl_cfg["batch"],
            shuffle=False
        )

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device used: {device}")

    # Get the encoder
    encoder = make_encoder(encoder_cfg, logger, load_pretrained=True)
    encoder.to(device)

    # Build the task model with the encoder
    task = make_task(encoder, task_cfg, encoder_cfg, dataset_cfg, train_cfg)
   

    # Load model from specific epoch to continue the training or start the evaluation
    if args["resume_from"]:
        resume_file = args["resume_from"]
    elif train_cfg["resume_from"]:
        resume_file = train_cfg["resume_from"]
    else:
        resume_file = None

    if resume_file:
        checkpoint = torch.load(resume_file, map_location="cpu")
        task.load_model(checkpoint)
        logger.info("Load model files from: %s" % resume_file)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Weighted Cross Entropy Loss & adam optimizer
    # weight = gen_weights(class_distr, c=train_cfg["weight_param"])

    optimizer = torch.optim.Adam(
        task.head.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["decay"]
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

    # Write model-graph to Tensorboard
    if train_cfg["mode"] == "train":
        # Start Training!
        task.head.train()

        
        for epoch in range(start_epoch, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")

            task.train_one_epoch(train_loader, epoch, optimizer, device, logger, writer)  
            # Start Evaluation
            if epoch % train_cfg["eval_every"] == 0 or epoch == 1:
                seed_all(0)
                task.eval_after_epoch(val_loader, epoch, scheduler, device, logger, writer)

                
            # Save model
            if epoch % train_cfg["save_every"] == 0:
                ckpt_path = os.path.join(exp_dir, "checkpoints", f"{epoch}.pth")
                task.save_model(ckpt_path)
                logger.info(f"Model saved at {ckpt_path}")

    elif train_cfg["mode"] == "test":
        task.make_prediction(args["ckpt_path"], test_loader, device, logger)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
