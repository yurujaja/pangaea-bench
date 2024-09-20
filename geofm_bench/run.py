import os as os
import pathlib
import pprint
import time

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from geofm_bench.utils.logger import init_logger
from geofm_bench.utils.utils import fix_seed


def get_exp_name(hydra_config: HydraConf) -> str:
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["foundation_model"]
    adaptor = choices["adaptor"]
    ds = choices["dataset"]
    return f"{timestamp}-{fm}-{adaptor}-{ds}"


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    exp_name = get_exp_name(HydraConfig.get())

    # fix all random seeds
    fix_seed(cfg.seed)

    # distributed training variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")

    if cfg.train:
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, config_log_dir / "config.yaml")
    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    # init wandb
    if cfg.use_wandb and rank == 0:
        import wandb

        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project="geofm-bench",
            name=exp_name,
            config=wandb_cfg,
            resume="allow",
            id=cfg.get("wandb_run_id"),
        )
        # TODO: add wandb_run_id to the saved config
        # cfg["wandb_run_id"] = wandb.run.id

    # get datasets
    dataset = instantiate(cfg.dataset, split="train")
    print(dataset)


#     dataset = DATASET_REGISTRY.get(cfg.dataset.dataset_name)
#     dataset.download(cfg.dataset, silent=False)
#     train_dataset, val_dataset, test_dataset = dataset.get_splits(cfg.dataset)
#
#     # Apply data processing to the datasets
#     for step in cfg.augmentation.train:
#         train_dataset = AUGMENTER_REGISTRY.get(step)(
#             train_dataset, cfg, cfg.augmentation.train[step]
#         )
#
#     for step in cfg.augmentation.test:
#         val_dataset = AUGMENTER_REGISTRY.get(step)(
#             val_dataset, cfg, cfg.augmentation.test[step]
#         )
#         test_dataset = AUGMENTER_REGISTRY.get(step)(
#             test_dataset, cfg, cfg.augmentation.test[step]
#         )
#
#     logger.info("Created processing pipelines:")
#     logger.info(f"   Training: {pprint.pformat([s for s in cfg.augmentation.train])}")
#     logger.info(f"   Evaluation: {pprint.pformat([s for s in cfg.augmentation.test])}")
#
#     logger.info("Built {} dataset.".format(dataset_name))
#
#     # prepare the foundation model
#     download_model(cfg.encoder)
#     encoder = ENCODER_REGISTRY.get(cfg.encoder.encoder_name)(
#         cfg.encoder, **cfg.encoder.encoder_model_args
#     )
#
#     missing, incompatible_shape = encoder.load_encoder_weights(
#         cfg.encoder.encoder_weights
#     )
#     logger.info("Loaded encoder weight from {}.".format(cfg.encoder.encoder_weights))
#     if missing:
#         logger.warning(
#             "Missing parameters:\n"
#             + "\n".join("%s: %s" % (k, v) for k, v in sorted(missing.items()))
#         )
#     if incompatible_shape:
#         logger.warning(
#             "Incompatible parameters:\n"
#             + "\n".join(
#                 "%s: expected %s but found %s" % (k, v[0], v[1])
#                 for k, v in sorted(incompatible_shape.items())
#             )
#         )
#
#     # prepare the segmentor
#     model = SEGMENTOR_REGISTRY.get(cfg.segmentor.segmentor_name)(
#         cfg, cfg.segmentor, encoder
#     ).to(device)
#     model = torch.nn.parallel.DistributedDataParallel(
#         model, device_ids=[cfg.local_rank], output_device=cfg.local_rank
#     )
#     logger.info(
#         "Built {} for with {} encoder.".format(
#             model.module.model_name, encoder.model_name
#         )
#     )
#     collate_fn = get_collate_fn(cfg)
#     # training
#     if not cfg.eval_dir:
#         if 0 < cfg.limited_label < 1:
#             indices = random.sample(
#                 range(len(train_dataset)), int(len(train_dataset) * cfg.limited_label)
#             )
#             train_dataset = Subset(train_dataset, indices)
#             perc = cfg.limited_label * 100
#             logger.info(
#                 f"Created a subset of the train dataset, with {perc}% of the labels available"
#             )
#         else:
#             logger.info("The entire train dataset will be used.")
#
#         # get train val data loaders
#         train_loader = DataLoader(
#             train_dataset,
#             sampler=DistributedSampler(train_dataset),
#             batch_size=cfg.batch_size,  # cfg.dataset["batch"],
#             num_workers=cfg.num_workers,
#             pin_memory=True,
#             # persistent_workers=True causes memory leak
#             persistent_workers=False,
#             worker_init_fn=seed_worker,
#             generator=get_generator(cfg.seed),
#             drop_last=True,
#             collate_fn=collate_fn,
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             sampler=DistributedSampler(val_dataset),
#             batch_size=cfg.batch_size,
#             num_workers=cfg.num_workers,
#             pin_memory=True,
#             persistent_workers=False,
#             worker_init_fn=seed_worker,
#             # generator=g,
#             drop_last=False,
#             collate_fn=collate_fn,
#         )
#
#         logger.info(
#             "Built {} dataset for training and evaluation.".format(dataset_name)
#         )
#
#         # flops calculator TODO: make it not hard coded
#         # TODO: Make this not drop the first training sample
#         if cfg.cal_flops:
#             train_features = next(iter(train_loader))
#             input_res = tuple(train_features["image"]["optical"].size())
#             macs, params = ptflops.get_model_complexity_info(
#                 model=model,
#                 input_res=input_res,
#                 input_constructor=prepare_input,
#                 as_strings=True,
#                 backend="pytorch",
#                 verbose=True,
#             )
#             logger.info(f"Model MACs: {macs}")
#             logger.info(f"Model Params: {params}")
#
#         # build loss
#         criterion = LOSS_REGISTRY.get(cfg.segmentor.loss.loss_name)(cfg.segmentor.loss)
#         logger.info("Built {} loss.".format(str(type(criterion))))
#
#         # build optimizer
#         optimizer = OPTIMIZER_REGISTRY.get(cfg.segmentor.optimizer.optimizer_name)(
#             model, cfg.segmentor.optimizer
#         )
#         logger.info("Built {} optimizer.".format(str(type(optimizer))))
#
#         # build scheduler
#         total_iters = cfg.epochs * len(train_loader)
#         scheduler = SCHEDULER_REGISTRY.get(cfg.segmentor.scheduler.scheduler_name)(
#             optimizer, total_iters, cfg.segmentor.scheduler
#         )
#         logger.info("Built {} scheduler.".format(str(type(scheduler))))
#
#         # training: put all components into engines
#         if task_name == "regression":
#             val_evaluator = RegEvaluator(cfg, val_loader, exp_dir, device)
#             trainer = RegTrainer(
#                 cfg,
#                 model,
#                 train_loader,
#                 criterion,
#                 optimizer,
#                 scheduler,
#                 val_evaluator,
#                 exp_dir,
#                 device,
#             )
#         else:
#             val_evaluator = SegEvaluator(cfg, val_loader, exp_dir, device)
#             trainer = SegTrainer(
#                 cfg,
#                 model,
#                 train_loader,
#                 criterion,
#                 optimizer,
#                 scheduler,
#                 val_evaluator,
#                 exp_dir,
#                 device,
#             )
#
#         # resume training if model_checkpoint is provided
#         if cfg.resume_from is not None:
#             trainer.load_model(cfg.resume_from)
#
#         trainer.train()
#
#     # Evaluation
#     else:
#         test_loader = DataLoader(
#             test_dataset,
#             sampler=DistributedSampler(test_dataset),
#             batch_size=cfg.batch_size,
#             num_workers=cfg.num_workers,
#             pin_memory=True,
#             persistent_workers=False,
#             drop_last=False,
#             collate_fn=collate_fn,
#         )
#
#         logger.info("Built {} dataset for evaluation.".format(dataset_name))
#
#         if task_name == "regression":
#             # TODO: This doesn't work atm
#             test_evaluator = RegEvaluator(cfg, test_loader, exp_dir, device)
#         else:
#             test_evaluator = SegEvaluator(cfg, test_loader, exp_dir, device)
#
#         model_ckpt_path = os.path.join(
#             exp_dir, next(f for f in os.listdir(exp_dir) if f.endswith("_best.pth"))
#         )
#         test_evaluator.evaluate(model, "best model", model_ckpt_path)
#
#     if cfg.use_wandb and cfg.rank == 0:
#         wandb.finish()
#
#
if __name__ == "__main__":
    main()
