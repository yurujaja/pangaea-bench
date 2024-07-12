import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score
import matplotlib.pyplot as plt
import numpy as np

class Evaluator():
    def __init__(self, args, preprocessor, val_loader, logger, exp_dir, device):

        self.args = args
        self.preprocessor = preprocessor
        self.val_loader = val_loader
        self.logger = logger
        self.exp_dir = exp_dir
        self.device = device
        #self.cls_name
        self.class_name = self.val_loader.dataset.class_name
        self.num_classes = len(self.class_name)
        self.max_name_len = max([len(name) for name in self.class_name])

    def __call__(self, model):
        pass

    def compute_metrics(self):
        pass

    def log_metrics(self, metrics):
        pass


class SegEvaluator(Evaluator):
    def __init__(self, args, preprocessor, val_loader, logger, exp_dir, device):
        super().__init__(args, preprocessor, val_loader, logger, exp_dir, device)


    @torch.no_grad()
    def __call__(self, model):
        model.eval()

        confusion_matrix = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        for batch_idx, data in enumerate(self.val_loader):
            image, target = self.preprocessor(data)
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

    def compute_metrics(self, confusion_matrix):
        iou = torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - torch.diag(confusion_matrix)) * 100
        iou = iou.cpu()
        metrics = {'IoU': [iou[i].item() for i in range(self.num_classes)], 'mIoU': iou.mean().item()}

        return metrics

    def log_metrics(self, metrics):
        header = "------- IoU --------\n"
        iou = '\n'.join(c.ljust(self.max_name_len, ' ') + '\t{:>7}'.format('%.3f' % num) for c, num in zip(self.class_name, metrics['IoU'])) + '\n'
        miou = "-------------------\n" + 'Mean'.ljust(self.max_name_len, ' ') + '\t{:>7}'.format('%.3f' % metrics['mIoU'])
        self.logger.info(header+iou+miou)






