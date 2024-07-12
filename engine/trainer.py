import os
import time

import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from utils.logger import AverageMeter, RunningAverageMeter, sec_to_hm


class Trainer():
    def __init__(self, args, model, preprocessor, train_loader, optimizer, lr_scheduler, evaluator, logger, exp_dir, device):
        #torch.set_num_threads(1)

        self.args = args
        self.rank = int(os.environ["RANK"])
        #self.train_cfg = train_cfg
        #self.dataset_cfg = dataset_cfg
        self.model = model
        self.preprocessor = preprocessor
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.logger = logger
        self.training_stats = {name: RunningAverageMeter(length=100) for name in ['loss', 'data_time', 'batch_time']}
        self.training_metrics ={}
        self.exp_dir = exp_dir
        self.device = device

        self.enable_mixed_precision = args.fp16 or args.bf16#train_cfg["mixed_precision"]
        self.precision = torch.float16 if args.fp16 else torch.bfloat16
        self.scaler = GradScaler(enabled=self.enable_mixed_precision)

        self.start_epoch = 0
        #self.eval_interval = args.eval_interval
        #self.ckpt_interval = args.ckpt_interval
        self.epochs = args.epochs
        if args.resume_path is not None:
            self.load_model(args.resume_path)


    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            # train the network for one epoch
            self.logger.info("============ Starting epoch %i ... ============" % epoch)
            # set sampler
            self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if (epoch + 1) % self.args.eval_interval == 0:
                self.evaluator(self.model)
            if self.rank == 0 and (epoch + 1) % self.args.ckpt_interval == 0:
                self.save_model(epoch)

        self.evaluator(self.model)
        if self.rank == 0:
            self.save_model(self.epochs, is_final=True)


    def train_one_epoch(self, epoch):
        self.model.train()

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            image, target = self.preprocessor(data)
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)
            self.training_stats['data_time'].update(time.time() - end_time)

            with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision, dtype=self.precision):
                logits = self.model(image, output_shape=(240, 240))
                loss = self.compute_loss(logits, target)
                self.compute_logging_metrics(logits.detach().clone(), target.detach().clone())

            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            self.training_stats['loss'].update(loss.item())
            if (batch_idx + 1) % self.args.log_interval == 0:
                self.log(batch_idx + 1, epoch)
            self.training_stats['batch_time'].update(time.time() - end_time)

            end_time = time.time()


    def save_model(self, epoch, is_final=False):
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "args": self.args,
        }
        checkpoint_path = os.path.join(self.exp_dir, "checkpoint_{}.pth".format('final' if is_final else f'epoch{epoch}'))
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}")


    def load_model(self, resume_path):
        model_dict = torch.load(resume_path, map_location=self.device)
        if 'model' in model_dict:
            self.model.module.load_state_dict(model_dict["model"])
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
            self.scaler.load_state_dict(model_dict["scaler"])
            self.start_epoch = model_dict["epoch"] + 1
        else:
            self.model.module.load_state_dict(model_dict)
            self.start_epoch = 0

        self.logger.info(f"Loaded model from {self.args.resume_path}. Start training from epoch {self.start_epoch}")

    def compute_loss(self, logits, target):
        pass

    @torch.no_grad()
    def compute_logging_metrics(self, logits, target):
        pass

    def log(self, batch_idx, epoch):

        #TO DO: upload to wandb
        left_batch_this_epoch = len(self.train_loader) - batch_idx
        left_batch_all = len(self.train_loader) * (self.epochs - epoch - 1) + left_batch_this_epoch
        left_time_this_epoch = sec_to_hm(left_batch_this_epoch * self.training_stats['batch_time'].avg)
        left_time_all = sec_to_hm(left_batch_all * self.training_stats['batch_time'].avg)

        basic_info = (
            "Epoch [{epoch}-{batch_idx}/{len_loader}]\t"
            "ETA [{left_time_all}|{left_time_this_epoch}]\t"
            "Time [{batch_time.avg:.3f}|{data_time.avg:.3f}]\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "lr {lr:.3e}"
            .format(
                epoch=epoch,
                len_loader=len(self.train_loader),
                batch_idx=batch_idx,
                left_time_this_epoch=left_time_this_epoch,
                left_time_all=left_time_all,
                batch_time=self.training_stats['batch_time'],
                data_time=self.training_stats['data_time'],
                loss=self.training_stats['loss'],
                lr=self.optimizer.param_groups[0]['lr']
            ))

        metrics_info = ['{} {:>7} ({:>7})'.format(k, '%.3f' % v.val, '%.3f' % v.avg) for k, v in self.training_metrics.items()]
        metrics_info = '\n'+'\t'.join(metrics_info)
        #extra_metrics_info = self.extra_info_template.format(**self.extra_info)
        log_info = basic_info + metrics_info

        self.logger.info(log_info)

    def reset_stats(self):
        for v in self.training_stats.values():
            v.reset()
        for v in self.training_metrics.values():
            v.reset()


class SegTrainer(Trainer):
    def __init__(self, args, model, preprocessor, train_loader, optimizer, scheduler, evaluator, logger, exp_dir, device):
        super().__init__(args, model, preprocessor, train_loader, optimizer, scheduler, evaluator, logger, exp_dir, device)

        self.training_metrics = {name: RunningAverageMeter(length=100) for name in ['Acc', 'mAcc', 'mIoU']}


    def compute_loss(self, logits, target):
        loss = F.cross_entropy(logits, target, ignore_index=-1)

        return loss

    @torch.no_grad()
    def compute_logging_metrics(self, logits, target):
        #logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear')
        pred = torch.argmax(logits, dim=1, keepdim=True)
        target = target.unsqueeze(1)
        ignore_mask = target == -1
        target[ignore_mask] = 0
        ignore_mask = ignore_mask.expand(-1, logits.shape[1], -1, -1)

        binary_pred = torch.zeros(logits.shape, dtype=bool, device=self.device)
        binary_target = torch.zeros(logits.shape, dtype=bool, device=self.device)
        binary_pred.scatter_(dim=1, index=pred, src=torch.ones_like(binary_pred))
        binary_target.scatter_(dim=1, index=target, src=torch.ones_like(binary_target))
        binary_pred[ignore_mask] = 0
        binary_target[ignore_mask] = 0

        intersection = torch.logical_and(binary_pred, binary_target)
        union = torch.logical_or(binary_pred, binary_target)

        acc = intersection.sum() / binary_target.sum() * 100
        macc = torch.nanmean(intersection.sum(dim=(0, 2, 3)) / binary_target.sum(dim=(0, 2, 3))) * 100
        miou = torch.nanmean(intersection.sum(dim=(0, 2, 3)) / union.sum(dim=(0, 2, 3))) * 100

        self.training_metrics['Acc'].update(acc.item())
        self.training_metrics['mAcc'].update(macc.item())
        self.training_metrics['mIoU'].update(miou.item())











