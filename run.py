import os
import time
import argparse


import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


import foundation_models
import datasets
import segmentors
from engine import SegPreprocessor, SegEvaluator, SegTrainer
from foundation_models.utils import download_model


from utils.seed import fix_seed, get_generator, seed_worker
from utils.logger import init_logger
from utils.configs import load_config
from utils.registry import ENCODER_REGISTRY, SEGMENTOR_REGISTRY, DATASET_REGISTRY


parser = argparse.ArgumentParser(description="Train a downstreamtask with geospatial foundation models.")



parser.add_argument("--dataset_config", required=True,
                    help="train config file path")
parser.add_argument("--encoder_config", required=True,
                    help="train config file path")
parser.add_argument("--segmentor_config", required=True,
                    help="train config file path")
parser.add_argument("--finetune", action="store_true",
                    help="fine tune whole networks")
parser.add_argument("--test_only", action="store_true",
                    help="test a model only (to be done)")


parser.add_argument("--work_dir", default="./work-dir",
                    help="the dir to save logs and models")
parser.add_argument("--resume_path", type=str,
                    help="load model from previous epoch")


parser.add_argument("--seed", default=0, type=int,
                    help="random seed")
parser.add_argument("--num_workers", default=8, type=int,
                    help="number of data loading workers")
parser.add_argument("--batch_size", default=8, type=int,
                    help="batch_size")


parser.add_argument("--epochs", default=80, type=int,
                    help="number of data loading workers")
parser.add_argument("--lr", default=1e-4, type=int,
                    help="base learning rate")
parser.add_argument("--lr_milestones", default=[0.6, 0.9], type=float, nargs="+",
                    help="milestones in lr schedule")
parser.add_argument("--wd", default=0.05, type=int,
                    help="weight decay")

parser.add_argument("--fp16", action="store_true",
                    help="use float16 for mixed precision training")
parser.add_argument("--bf16", action="store_true",
                    help="use bfloat16 for mixed precision training")


parser.add_argument("--ckpt_interval", default=20, type=int,
                    help="checkpoint interval in epochs")
parser.add_argument("--eval_interval", default=5, type=int,
                    help="evaluate interval in epochs")
parser.add_argument("--log_interval", default=10, type=int,
                    help="log interval in iterations")


parser.add_argument('--rank', default=-1,
                    help='rank of current process')
parser.add_argument('--local_rank', default=-1,
                    help='local rank of current process')
parser.add_argument('--world_size', default=1,
                    help="world size")
parser.add_argument('--local_world_size', default=1,
                    help="local world size")
parser.add_argument('--init_method', default='tcp://localhost:10111',
                    help="url for distributed training")



if __name__ == "__main__":
    args = parser.parse_args()

    # fix all random seeds
    fix_seed(args.seed)

    # distributed training variables
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend='nccl')

    # load config
    encoder_cfg, dataset_cfg, segmentor_cfg = load_config(args)

    encoder_name = encoder_cfg["encoder_name"]
    dataset_name = dataset_cfg["dataset_name"]
    task_name = segmentor_cfg["task_name"]
    segmentor_name = segmentor_cfg["segmentor_name"]

    # setup a work directory and logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"{timestamp}-{encoder_name}-{segmentor_name}-{dataset_name}-{task_name}"
    exp_dir = os.path.join(args.work_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = init_logger(os.path.join(exp_dir, "train.log"), rank=args.rank)

    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment will be stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    # get datasets
    dataset = DATASET_REGISTRY.get(dataset_cfg['dataset_name'])
    dataset.download(dataset_cfg, silent=False)
    train_dataset, val_dataset, test_dataset = dataset.get_splits(dataset_cfg)

    # get train val data loaders
    train_loader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset),
        batch_size=args.batch_size,#dataset_cfg["batch"],
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=get_generator(args.seed),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=DistributedSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        #worker_init_fn=seed_worker,
        #generator=g,
        drop_last=False,
    )

    logger.info("Built {} dataset.".format(dataset_name))

    # prepare the foundation model
    download_model(encoder_cfg)
    encoder = ENCODER_REGISTRY.get(encoder_cfg['encoder_name'])(encoder_cfg, **encoder_cfg['encoder_model_args'])

    missing, incompatible_shape = encoder.load_encoder_weights(encoder_cfg['encoder_weights'])
    logger.info("Loaded encoder weight from {}.".format(encoder_cfg['encoder_weights']))
    if missing:
        logger.warning("Missing parameters:\n" + "\n".join("%s: %s" % (k, v) for k, v in sorted(missing.items())))
    if incompatible_shape:
        logger.warning("Incompatible parameters:\n" + "\n".join("%s: expected %s but found %s" % (k, v[0], v[1]) for k, v in sorted(incompatible_shape.items())))

    # prepare the segmentor
    model = SEGMENTOR_REGISTRY.get(segmentor_cfg['segmentor_name'])(args, segmentor_cfg, encoder).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    logger.info("Built {} for with {} encoder.".format(model.module.model_name, encoder.model_name))

    # build optimizer
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.wd)
    logger.info("Built {} optimizer.".format(str(type(optimizer))))

    # build scheduler
    total_iters = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [total_iters * r for r in args.lr_milestones], gamma=0.1)
    logger.info("Built {} scheduler.".format(str(type(scheduler))))

    # training: put all components into engines
    preprocessor = SegPreprocessor(args, encoder_cfg, dataset_cfg, logger)
    val_evaluator = SegEvaluator(args, preprocessor, val_loader, logger, exp_dir, device)
    trainer = SegTrainer(args, model, preprocessor, train_loader, optimizer, scheduler, val_evaluator, logger, exp_dir, device)
    trainer.train()

    # testing
    test_loader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )

    test_evaluator = SegEvaluator(args, preprocessor, test_loader, logger, exp_dir, device)
    test_evaluator.evaluate(model, 'final model')








