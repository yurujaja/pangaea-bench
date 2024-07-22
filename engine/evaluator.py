import torch
import torch.nn.functional as F

import time
from tqdm import tqdm
import numpy as np
import logging

class Evaluator():
    def __init__(self, args, val_loader, exp_dir, device):

        self.args = args
        self.val_loader = val_loader
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        #self.cls_name
        self.classes = self.val_loader.dataset.classes
        self.split = self.val_loader.dataset.split
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])

    def __call__(self, model):
        pass

    def compute_metrics(self):
        pass

    def log_metrics(self, metrics):
        pass


class SegEvaluator(Evaluator):
    def __init__(self, args, val_loader, exp_dir, device):
        super().__init__(args, val_loader, exp_dir, device)

    @torch.no_grad()
    def evaluate(self, model, model_name='model'):
        t = time.time()

        model.eval()

        tag = f'Evaluating {model_name} on {self.split} set'
        confusion_matrix = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image, target = data # TODO make this consistent with how data is passed around before the preprocessor
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            logits = model(image, output_shape=target.shape[-2:])
            pred = torch.argmax(logits, dim=1)
            valid_mask = target != -1
            pred, target = pred[valid_mask], target[valid_mask]
            count = torch.bincount((pred * self.num_classes + target), minlength=self.num_classes ** 2)
            confusion_matrix += count.view(self.num_classes, self.num_classes)

        torch.distributed.all_reduce(confusion_matrix, op=torch.distributed.ReduceOp.SUM)
        metrics = self.compute_metrics(confusion_matrix)
        self.log_metrics(metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name='model'):
        return self.evaluate(model, model_name)


    def compute_metrics(self, confusion_matrix):
        iou = torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - torch.diag(confusion_matrix)) * 100
        iou = iou.cpu()
        metrics = {'IoU': [iou[i].item() for i in range(self.num_classes)], 'mIoU': iou.mean().item()}

        return metrics

    def log_metrics(self, metrics):
        header = "------- IoU --------\n"
        iou = '\n'.join(c.ljust(self.max_name_len, ' ') + '\t{:>7}'.format('%.3f' % num) for c, num in zip(self.classes, metrics['IoU'])) + '\n'
        miou = "-------------------\n" + 'Mean'.ljust(self.max_name_len, ' ') + '\t{:>7}'.format('%.3f' % metrics['mIoU'])
        self.logger.info(header+iou+miou)