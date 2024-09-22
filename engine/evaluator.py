import os
import time
from tqdm import tqdm
import logging
import torch
import torch.nn.functional as F


class Evaluator:
    def __init__(self, args, val_loader, exp_dir, device):

        self.args = args
        self.val_loader = val_loader
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        # self.cls_name
        self.classes = self.val_loader.dataset.classes
        self.split = self.val_loader.dataset.split
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])

        if args.use_wandb:
            import wandb

            self.wandb = wandb

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
    def evaluate(self, model, model_name="model", model_ckpt_path=None):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded model from {model_ckpt_path} for evaluation")

        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )

        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image, target = data["image"], data["target"]
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            logits = model(image, output_shape=target.shape[-2:])
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
            else:
                pred = torch.argmax(logits, dim=1)
            valid_mask = target != -1
            pred, target = pred[valid_mask], target[valid_mask]
            count = torch.bincount(
                (pred * self.num_classes + target), minlength=self.num_classes**2
            )
            confusion_matrix += count.view(self.num_classes, self.num_classes)

        torch.distributed.all_reduce(
            confusion_matrix, op=torch.distributed.ReduceOp.SUM
        )
        metrics = self.compute_metrics(confusion_matrix)
        self.log_metrics(metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name="model", model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)

    def compute_metrics(self, confusion_matrix):
        # Calculate IoU for each class
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6)) * 100

        # Calculate precision and recall for each class
        precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
        recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        miou = iou.mean().item()
        mf1 = f1.mean().item()
        macc = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100

        # Convert metrics to CPU and to Python scalars
        iou = iou.cpu()
        f1 = f1.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        # Prepare the metrics dictionary
        metrics = {
            "IoU": [iou[i].item() for i in range(self.num_classes)],
            "mIoU": miou,
            "F1": [f1[i].item() for i in range(self.num_classes)],
            "mF1": mf1,
            "mAcc": macc,
            "Precision": [precision[i].item() for i in range(self.num_classes)],
            "Recall": [recall[i].item() for i in range(self.num_classes)],
        }

        return metrics

    def log_metrics(self, metrics):
        def format_metric(name, values, mean_value):
            header = f"------- {name} --------\n"
            metric_str = (
                "\n".join(
                    c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                    for c, num in zip(self.classes, values)
                )
                + "\n"
            )
            mean_str = (
                "-------------------\n"
                + "Mean".ljust(self.max_name_len, " ")
                + "\t{:>7}".format("%.3f" % mean_value)
            )
            return header + metric_str + mean_str

        iou_str = format_metric("IoU", metrics["IoU"], metrics["mIoU"])
        f1_str = format_metric("F1-score", metrics["F1"], metrics["mF1"])

        precision_mean = torch.tensor(metrics["Precision"]).mean().item()
        recall_mean = torch.tensor(metrics["Recall"]).mean().item()

        precision_str = format_metric("Precision", metrics["Precision"], precision_mean)
        recall_str = format_metric("Recall", metrics["Recall"], recall_mean)

        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"

        self.logger.info(iou_str)
        self.logger.info(f1_str)
        self.logger.info(precision_str)
        self.logger.info(recall_str)
        self.logger.info(macc_str)

        if self.args.use_wandb and self.args.rank == 0:
            self.wandb.log(
                {
                    "val_mIoU": metrics["mIoU"],
                    "val_mF1": metrics["mF1"],
                    "val_mAcc": metrics["mAcc"],
                    **{f"val_IoU_{c}": v for c, v in zip(self.classes, metrics["IoU"])},
                    **{f"val_F1_{c}": v for c, v in zip(self.classes, metrics["F1"])},
                    **{
                        f"val_Precision_{c}": v
                        for c, v in zip(self.classes, metrics["Precision"])
                    },
                    **{
                        f"val_Recall_{c}": v
                        for c, v in zip(self.classes, metrics["Recall"])
                    },
                }
            )


class RegEvaluator(Evaluator):
    def __init__(self, args, val_loader, exp_dir, device):
        super().__init__(args, val_loader, exp_dir, device)

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        # TODO: Rework this to allow evaluation only runs
        # Move common parts to parent class, and get loss function from the registry.
        t = time.time()
        
        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device)
            model_name = os.path.basename(model_ckpt_path).split('.')[0]
            if 'model' in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded model from {model_ckpt_path} for evaluation")

        model.eval()

        tag = f'Evaluating {model_name} on {self.split} set'
        # confusion_matrix = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image, target = data['image'], data['target']
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            logits = model(image, output_shape=target.shape[-2:]).squeeze(dim=1)
            mse = F.mse_loss(logits, target)

        # torch.distributed.all_reduce(confusion_matrix, op=torch.distributed.ReduceOp.SUM)
        metrics = {"MSE" : mse.item(), "RMSE" : torch.sqrt(mse).item()}
        self.log_metrics(metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name='model'):
        return self.evaluate(model, model_name)

    def log_metrics(self, metrics):
        header = "------- MSE and RMSE --------\n"
        mse = "-------------------\n" + 'MSE \t{:>7}'.format('%.3f' % metrics['MSE'])+'\n'
        rmse = "-------------------\n" + 'RMSE \t{:>7}'.format('%.3f' % metrics['RMSE'])
        self.logger.info(header+mse+rmse)

        if self.args.use_wandb and self.args.rank == 0:
            self.wandb.log({"val_MSE": metrics["MSE"], "val_RMSE": metrics["RMSE"]})
