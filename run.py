import os
import time
import argparse


import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from engine import SegPreprocessor, SegEvaluator, SegTrainer
from foundation_models import prithviEncoderViT
from foundation_models.utils import download_model
from downstream_models import UPerNet, UNet

from datasets.utils import make_dataset


from utils.logger import init_logger
from utils.configs import load_config
#from utils.utils import fix_seed

parser = argparse.ArgumentParser(description="Train a downstreamtask with geospatial foundation models.")



parser.add_argument("--dataset_config", required=True,
                    help="train config file path")
parser.add_argument("--encoder_config", required=True,
                    help="train config file path")
parser.add_argument("--task_config", required=True,
                    help="train config file path")
parser.add_argument("--run_config",
                   help="read run config from file")
parser.add_argument("--test_only", action="store_true",
                    help="")



parser.add_argument("--work_dir", default="./work-dir",
                    help="the dir to save logs and models")
parser.add_argument("--resume_path", type=str,
                    help="load model from previous epoch")
#parser.add_argument("--ckpt_path", type=str,
#                    help="load the checkpoint for evaluation")


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
parser.add_argument("--wd", default=1e-4, type=int,
                    help="weight decay")

parser.add_argument("--fp16", action="store_true",
                    help="")
parser.add_argument("--bf16", action="store_true",
                    help="")


parser.add_argument("--ckpt_interval", default=20, type=int,
                    help="")
parser.add_argument("--eval_interval", default=5, type=int,
                    help="")
parser.add_argument("--log_interval", default=10, type=int,
                    help="")


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

    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])

    encoder_cfg, dataset_cfg, task_cfg = load_config(args)

    #print(encoder_cfg)
    mode = 'test' if args.test_only else 'train'
    encoder_name = encoder_cfg["encoder_name"]
    dataset_name = dataset_cfg["dataset"]
    task_name = task_cfg["task_model_name"]

    # setup a work directory
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"{encoder_name}-{dataset_name}-{task_name}-{mode}-{timestamp}"
    exp_dir = os.path.join(args.work_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = init_logger(os.path.join(exp_dir, "train.log"), rank=args.rank)

    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment will be stored in %s\n" % exp_dir)

    # initialize distributed training

    device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(device)
    logger.info(f"Device used: {device}")

    torch.distributed.init_process_group(backend='nccl')

    train_dataset, val_dataset, test_dataset = make_dataset(dataset_cfg)

    train_loader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset),
        batch_size=args.batch_size,#dataset_cfg["batch"],
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        #worker_init_fn=self.seed_worker,
        #generator=self.g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=DistributedSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        #worker_init_fn=self.seed_worker,
        #generator=self.g,
        drop_last=False,
    )
    total_iters = args.epochs * len(train_loader)
    logger.info("Built {} dataset.".format(dataset_name))

    download_model(encoder_cfg)

    encoder = prithviEncoderViT(encoder_cfg, **encoder_cfg["encoder_model_args"])

    missing, incompatible_shape = encoder.load_encoder_weights(encoder_cfg["encoder_weights"])
    logger.info("Loaded encoder weight from {}.".format(encoder_cfg["encoder_weights"]))
    if missing:
        logger.warning("Missing parameters:\n" + "\n".join("%s: %s" % (k, v) for k, v in sorted(missing.items())))
    if incompatible_shape:
        logger.warning("Incompatible parameters:\n" + "\n".join("%s: expected %s but found %s" % (k, v[0], v[1]) for k, v in sorted(incompatible_shape.items())))


    model = UPerNet(encoder).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    logger.info("Built {} for semantic segmentation with {} encoder.".format(model.module.model_name, encoder.model_name))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)
    logger.info("Built {} optimizer.".format(str(type(optimizer))))


    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [total_iters * r for r in args.lr_milestones], gamma=0.1)
    logger.info("Built {} scheduler.".format(str(type(scheduler))))


    preprocessor = SegPreprocessor(args, encoder_cfg, dataset_cfg, logger)
    evaluator = SegEvaluator(args, preprocessor, val_loader, logger, exp_dir, device)
    trainer = SegTrainer(args, model, preprocessor, train_loader, optimizer, scheduler, evaluator, logger, exp_dir, device)

    trainer.train()







