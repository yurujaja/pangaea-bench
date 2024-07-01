import logging

import torch
from tqdm import tqdm
from utils.adaptation import adapt_input, adapt_target

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score
import sklearn.metrics as metr
import numpy as np


class SemanticSegmentationTask():
    '''
    Semantic segmentation task shead. 
    Args:
        head (class): head model
        img_size (int): size of the input image
        losses (list): list of loss functions
        encoder_cfg (dict): encoder configuration
        dataset_cfg (dict): dataset configuration
        train_cfg (dict): training configuration
    '''

    def __init__(self, head, img_size, losses, encoder_cfg, dataset_cfg, train_cfg):
        self.head = head
        self.img_size = img_size
        self.losses = losses
        self.encoder_cfg = encoder_cfg
        self.dataset_cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.encoder_name = encoder_cfg["encoder_name"]

        
    def compute_batch_losses(self, logits, target):
        loss = dict()
        for loss_fn in self.losses:
            if loss_fn == "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss(
                            ignore_index=-1,
                            reduction="mean",
                            # weight=weight.to(device),
                            # label_smoothing=task_args["label_smoothing"],
                        )
                loss["cross_entropy"] = criterion(logits, target)
            # TODO
            elif loss_fn == "dice":
                pass
            # TODO
            elif loss_fn == "focal":
                pass
            # TODO
            else:
                raise ValueError(f"Unknown loss function {loss_fn}")
        return loss


    # TODO: train_one_step is not complete!!
    def train_one_step(self, image, target, optimizer, clip_grad=None):
        self.head.train()
        optimizer.zero_grad()
        logits = self.head(image)
        
        loss = self.compute_losses(logits, target)

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                self.head.parameters(), clip_grad
            )
        optimizer.step()

        return loss
        
    
    def train_one_epoch(self, dataloader, epoch, optimizer, device, writer):
        self.head.train()

        training_batches = 0
        cumulative_losses = dict()
        i_board = 0
        for it, data in enumerate(tqdm(dataloader, desc="training")):
            it = len(dataloader) * (epoch - 1) + it  # global training iteration
            image = data['image']
            target = data['target']

            image = adapt_input(
                input=image,
                size=self.img_size,
                source_modal=self.dataset_cfg["bands"],
                target_modal=self.encoder_cfg["input_bands"],
                encoder_type=self.encoder_name,
                device=device,
            )

            target = adapt_target(
                tensor=target,
                size=self.img_size,
                device=device
            )   

            optimizer.zero_grad()      
            logits = self.head(image)

            loss = self.compute_batch_losses(logits, target)    

            for k, v in loss.items():
                if k in cumulative_losses:
                    # TODO: this only works for the cross entropy loss with mean reduction
                    cumulative_losses[k] += v.data * target.shape[0]
                else: 
                    cumulative_losses[k] = v.data * target.shape[0]
                writer.add_scalar(
                    f"training loss/{k}", v, (epoch - 1) * len(dataloader) + i_board)

            if self.train_cfg['clip_grad'] is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.head.parameters(), self.train_cfg['clip_grad']
                )

            optimizer.step()

            training_batches += target.shape[0]
            i_board += 1

        for loss_name, total_loss in cumulative_losses.items():
            logging.getLogger().info(f"The training loss {loss_name}: {total_loss/training_batches}")


    def evaluate_one_step(self, image, target):
        self.head.eval()
        with torch.no_grad():
            logits = self.head(image)
            loss = self.losses[0](logits, target)
        return loss
    

    def eval_after_epoch(self, dataloader, epoch, scheduler, device, writer):
        self.head.eval()
        
        val_loss = dict()
        val_batches = 0
        y_true_val = []
        y_predicted_val = []

        with torch.no_grad():
            for data in tqdm(dataloader, desc="validating"):
                image = data['image']
                target = data['target']

                image = adapt_input(
                    input=image,
                    size=self.img_size,
                    source_modal=self.dataset_cfg["bands"],
                    target_modal=self.encoder_cfg["input_bands"],
                    encoder_type=self.encoder_name,
                    device=device,
                )

                target = adapt_target(
                    tensor=target,
                    size=self.img_size,
                    device=device
                )

                logits = self.head(image)
                loss = self.compute_batch_losses(logits, target)    

                # Accuracy metrics only on annotated pixels
                logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
                logits = logits.reshape((-1, self.dataset_cfg["num_classes"]))
                target = target.reshape(-1)
                mask = target != -1
                logits = logits[mask]
                target = target[mask]

                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                target = target.cpu().numpy()

                val_batches += target.shape[0]

                for k, v in loss.items():
                    if k in val_loss:
                    # TODO: this only works for the cross entropy loss with mean reduction
                        val_loss[k] += v.data * target.shape[0]
                    else: 
                        val_loss[k] = v.data * target.shape[0]
                            

                y_predicted_val += probs.argmax(1).tolist()
                y_true_val += target.tolist()

            y_predicted_val = np.asarray(y_predicted_val)
            y_true_val = np.asarray(y_true_val)

            # Save Scores to the .log file and visualize also with tensorboard
            acc_val = self.evaluation_metrics(y_predicted_val, y_true_val)

        logger = logging.getLogger()
        logger.info("Evaluating model..")
        logger.info("RESULTS AFTER EPOCH " + str(epoch) + ": \n")
        for loss_name, total_loss in val_loss.items():
            logger.info(f"The val loss {loss_name}: {total_loss/val_batches}")
        logger.info(f"Evaluation: {acc_val}")


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

        if self.train_cfg["reduce_lr_on_plateau"] == 1:
            # TODO
            average_loss = sum(sum(val_loss[name]) / val_batches for name in val_loss) / len(val_loss)
            scheduler.step(average_loss)
        else:
            scheduler.step()


    def evaluation_metrics(self, y_predicted, y_true):
        micro_prec = precision_score(y_true, y_predicted, average='micro')
        macro_prec = precision_score(y_true, y_predicted, average='macro')
        weight_prec = precision_score(y_true, y_predicted, average='weighted')
        
        micro_rec = recall_score(y_true, y_predicted, average='micro')
        macro_rec = recall_score(y_true, y_predicted, average='macro')
        weight_rec = recall_score(y_true, y_predicted, average='weighted')
            
        macro_f1 = f1_score(y_true, y_predicted, average="macro")
        micro_f1 = f1_score(y_true, y_predicted, average="micro")
        weight_f1 = f1_score(y_true, y_predicted, average="weighted")
            
        subset_acc = accuracy_score(y_true, y_predicted)
        
        iou_acc = jaccard_score(y_true, y_predicted, average='macro')

        info = {
                "macroPrec" : macro_prec,
                "microPrec" : micro_prec,
                "weightPrec" : weight_prec,
                "macroRec" : macro_rec,
                "microRec" : micro_rec,
                "weightRec" : weight_rec,
                "macroF1" : macro_f1,
                "microF1" : micro_f1,
                "weightF1" : weight_f1,
                "subsetAcc" : subset_acc,
                "IoU": iou_acc
                }
        
        return info


    def make_prediction(self, ckpt_path, dataloader, device):
        self.load_model(ckpt_path)
        self.head.eval()

        y_true = []
        y_predicted = []

        with torch.no_grad():
            for data in tqdm(dataloader, desc="testing"):
                image = data['image']
                target = data['target']
                
    
                image = adapt_input(
                    input=image,
                    size=self.img_size,
                    source_modal=self.dataset_cfg["bands"],
                    target_modal=self.encoder_cfg["input_bands"],
                    encoder_type=self.encoder_name,
                    device=device,
                )

                target = adapt_target(
                    tensor=target,
                    size=self.img_size,
                    device=device
                )

                logits = self.head(image)

                probs = torch.nn.functional.softmax(logits, dim=1)
                predictions = probs.argmax(1)
                
                predictions = predictions.reshape(-1)
                target = target.reshape(-1)
                mask = target != -1
                
                predictions = predictions[mask].cpu().numpy()
                target = target[mask]
                
                target = target.cpu().numpy()
                
                y_predicted += predictions.tolist()
                y_true += target.tolist()


            # Save Scores to the .log file and visualize also with tensorboard
            acc = self.evaluation_metrics(y_predicted, y_true)
            logger = logging.getLogger()
            logger.info("\n")
            logger.info("STATISTICS: \n")
            logger.info("Evaluation: " + str(acc))
            print("Evaluation: " + str(acc))


    def save_model(self, path):
        torch.save(self.head.state_dict(), path)


    def load_model(self, path):
        checkpoint = torch.load(path)
        self.head.load_state_dict(checkpoint)
        
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()