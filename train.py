# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: train.py includes the training process for the
             pixel-level semantic segmentation.
'''
import sys
import os
import os.path as osp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets')))
try:
    from datasets.mados import *
except Exception as e:
    print("Failed to import from datasets.mados:", str(e))
    raise

import ast

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
# from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as T
from timm.utils import ModelEma

sys.path.append(up(os.path.abspath(__file__)))
# print(up(os.path.abspath(__file__)))
sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'tasks'))
sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'models'))
# from marinext_wrapper import MariNext

# from tasks.models_vit_tensor_CD_2 import *
from tasks import upernet_vit_base
from models import prithvi_vit_base, spectral_gpt_vit_base


from datasets.mados import MADOS, gen_weights, class_distr
from utils.metrics import Evaluation
from utils.utils import bool_flag, cosine_scheduler
from utils.pos_embed import interpolate_pos_embed
from utils.utils import bool_flag, cosine_scheduler



def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        return yaml.safe_load(file)


def load_checkpoint(encoder, ckpt_path, model="prithvi"):
    # TODO: cpu
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    logging.info("Load pre-trained checkpoint from: %s" % ckpt_path)

    if model == "prithvi":
        checkpoint_model = checkpoint
        del checkpoint_model["pos_embed"]
        del checkpoint_model["decoder_pos_embed"]
    elif model == "spectral_gpt":
        checkpoint_model = checkpoint['model']

    state_dict = encoder.state_dict()

    if model == "spectral_gpt":
        interpolate_pos_embed(encoder, checkpoint_model)
        for k in ['patch_embed.0.proj.weight', 'patch_embed.1.proj.weight', 'patch_embed.2.proj.weight',
                'patch_embed.2.proj.bias', 'head.weight', 'head.bias', 'pos_embed_spatial', 'pos_embed_temporal']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    msg = encoder.load_state_dict(checkpoint_model, strict=False)
    return msg 


def get_encoder_model(cfg, load_pretrained=True):
    # create model
    encoders = {
        'prithvi' : prithvi_vit_base,
        'spectral_gpt': spectral_gpt_vit_base,
    }
    encoder_name = cfg['encoder_name']
    if encoder_name not in encoders:     
        raise ValueError(f"{encoder_name} is not yet supported.")

    encoder_model_args = cfg["encoder_model_args"]
    encoder_model = encoders[encoder_name](**encoder_model_args)

    # load pretrained weights if there are any
    encoder_weights = cfg["encoder_weights"]    
    if encoder_weights is not None and load_pretrained:
        load_checkpoint(encoder_model, encoder_weights, encoder_name)
        
    return encoder_model



def create_task_model(cfg, encoder):
    models = {
        'upernet' : upernet_vit_base,
    }
    model_name = cfg['task_model_name']
    if model_name not in models:     
        raise ValueError(f"{model_name} is not yet supported.")
    model_args = cfg['task_model_args']
    model = models[model_name](encoder=encoder, **model_args)

    return model


def make_train_dataset(cfgs): 
    pass
'''   
    if cfgs.dataset_type == 'MADOS':
        train_dataset = 
    else:
        raise ValueError(f"{cfgs.dataset_type} is not yet supported.")
'''


def VSCP(image, target):
    
    n_augmented = image.shape[0]//2
    
    image_temp = image[:n_augmented*2,:,:,:].copy()
    target_temp = target[:n_augmented*2,:,:].copy()
    
    image_augmented = []
    target_augmented = []
    for i in range(n_augmented):

        image_temp[i,:,target_temp[i+n_augmented,:,:]!=-1] = image_temp[i+n_augmented,:,target_temp[i+n_augmented,:,:]!=-1]
        image_augmented.append(image_temp[i,:,:].copy())
        
        target_temp[i,target_temp[i+n_augmented,:,:]!=-1] = target_temp[i+n_augmented,target_temp[i+n_augmented,:,:]!=-1]
        target_augmented.append(target_temp[i,:,:].copy())
    
    image_augmented = np.stack(image_augmented)
    target_augmented = np.stack(target_augmented)
    
    return image_augmented, target_augmented


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
    parser = argparse.ArgumentParser(description='Train a downstreamtask with geospatial foundation models.')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', default='./work-dir', help='the dir to save logs and models')

    parser.add_argument('--path', help='Path of the images')
  
    parser.add_argument('--mode', default='train', help='select between train or test ')
    parser.add_argument('--epochs', default=80, type=int, help='Number of epochs to run')
    parser.add_argument('--batch', default=2, type=int, help='Batch size')
    parser.add_argument('--resume_from_epoch', default=0, type=int, help='load model from previous epoch')
    
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=15, type=int, help='Number of output classes')
    parser.add_argument('--weight_param', default=1.03, type=float, help='Weighting parameter for Loss Function')

    # Optimization
    parser.add_argument('--vscp',  type=bool_flag, default=True)
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Label smoothing')
    parser.add_argument('--clip_grad', default=None, type=float, help='Gradient Cliping')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='learning rate decay')
    parser.add_argument('--reduce_lr_on_plateau', default=0, type=int, help='reduce learning rate when no increase (0 or 1)')
    parser.add_argument('--lr_steps', default='[45,65]', type=str, help='Specify the steps that the lr will be reduced')

    # Evaluation/Checkpointing
    parser.add_argument('--checkpoint_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models'), help='folder to save checkpoints into (empty = this folder)')
    parser.add_argument('--eval_every', default=1, type=int, help='How frequently to run evaluation (epochs)')

    # # EMA related parameters
    parser.add_argument('--model_ema',  type=bool_flag, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.999, help='')
    parser.add_argument('--model_ema_eval',  type=bool_flag, default=True, help='Using ema to eval during training.')

    # misc
    parser.add_argument('--num_workers', default=0, type=int, help='How many cpus for loading data (0 is the main process)')
    parser.add_argument('--pin_memory', default=False, type=bool_flag, help='Use pinned memory or not')
    parser.add_argument('--prefetch_factor', default=2, type=int, help='Number of sample loaded in advance by each worker')
    parser.add_argument('--persistent_workers', default=False, type=bool_flag, help='This allows to maintain the workers Dataset instances alive.')
    parser.add_argument('--tensorboard', default='tsboard_segm', type=str, help='Name for tensorboard run')


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args = vars(args)  # convert to ordinary dict
    
    # lr_steps list or single float
    lr_steps = ast.literal_eval(args['lr_steps'])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise
        
    args['lr_steps'] = lr_steps
    

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)

    return args


def main(args):
    # Reproducibility
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)

    # setup logging, make one log on every process with the configuration for debugging.
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'{os.path.splitext(osp.basename(args.config))[0]}-{timestamp}'  
    exp_dir = osp.join(args.work_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = osp.join(exp_dir, f'{exp_name}.log')
    logging.basicConfig(
        filename=log_file,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging.info('Parsed task training parameters:')
    logging.info(json.dumps(args, indent = 2))

    # Load Config file
    model_config = load_config(args.config)
    encoder_name = model_config['encoder_name']
    logging.info(f"Loaded configuration: {config}")

    
    # Tensorboard
    writer = SummaryWriter(os.path.join(exp_dir, 'tensorboard', timestamp)) 


    splits_path = os.path.join(args['path'],'splits')
    
    # Construct Data loader
    dataset_train = MADOS(args['path'], splits_path, 'train')
    dataset_val = MADOS(args['path'], splits_path, 'val')
    
    train_loader = DataLoader(  dataset_train, 
                                batch_size = args['batch'], 
                                shuffle = True,
                                num_workers = args['num_workers'],
                                pin_memory = args['pin_memory'],
                                prefetch_factor = args['prefetch_factor'],
                                persistent_workers= args['persistent_workers'],
                                worker_init_fn=seed_worker,
                                generator=g,
                                drop_last=True)
    
    val_loader = DataLoader(   dataset_val, 
                                batch_size = args['batch'], 
                                shuffle = False,
                                num_workers = args['num_workers'],
                                pin_memory = args['pin_memory'],
                                prefetch_factor = args['prefetch_factor'],
                                persistent_workers= args['persistent_workers'],
                                worker_init_fn=seed_worker,     
                                generator=g)         
    
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # TODO
    device = torch.device("cpu")
    logging.info(f'Device used: {device}')

    # Get the encoder 
    encoder = get_encoder_model(model_config, load_pretrained=True)
    encoder.to(device)

    model = create_task_model(model_config, encoder)
    
    input1 = torch.rand(2, 6, 1, 224, 224)
    output = model(input1)#["out"]
    print((output.shape))

    # Load model from specific epoch to continue the training or start the evaluation
    if args['resume_from'] is not None:
        model_file = args['resume_from']
        logging.info('Loading model files from folder: %s' % model_file)

        checkpoint = torch.load(model_file, map_location = device)
        model.load_state_dict(checkpoint)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # Weighted Cross Entropy Loss & adam optimizer
    weight = gen_weights(class_distr, c = args['weight_param'])
    
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction= 'mean', weight=weight.to(device), label_smoothing=args['label_smoothing'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['decay'])

    # Learning Rate scheduler
    if args['reduce_lr_on_plateau']==1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args['lr_steps'], gamma=0.1, verbose=True)

    # Start training
    start_epoch = 1
    if args['resume_from'] is not None:
        start_epoch = int(osp.splitext(osp.basename(args['resume_from']))[0]) + 1
    epochs = args['epochs']
    eval_every = args['eval_every']

    # Write model-graph to Tensorboard
    if args['mode']=='train':        
        
        # Start Training!                                            
        model.train()
        
        for epoch in range(start_epoch, epochs+1):

            training_loss = []
            training_batches = 0
            
            i_board = 0
            for it, (image, target) in enumerate(tqdm(train_loader, desc="training")):
                
                it = len(train_loader) * (epoch-1) + it  # global training iteration
                
                # TODO
                image = T.resize(img = image, size = (128, 128), interpolation = T.InterpolationMode.BILINEAR).squeeze().cuda().float()
                target = T.resize(img = target, size = (128, 128), interpolation = T.InterpolationMode.NEAREST).squeeze().long().cuda()

                # print("batch after resize", image.shape)

                optimizer.zero_grad()

                output = model(image)["out"]
                # print(output.shape)
                logits = F.upsample(input=output, 
                                     size=image.size()[2:4], mode='bilinear')
                
                loss = criterion(logits, target)

                loss.backward()
    
                training_batches += target.shape[0]
    
                training_loss.append((loss.data*target.shape[0]).tolist())
                
                if args['clip_grad'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad'])
                
                optimizer.step()
                
                
                # Write running loss
                writer.add_scalar('training loss', loss , (epoch - 1) * len(train_loader)+i_board)
                i_board+=1
            
            logging.info("Training loss was: " + str(sum(training_loss) / training_batches))
            
            ckpt_path = os.path.join(exp_dir, 'checkpoints', f'{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Save models to {ckpt_path}")
            
           
            # Start Evaluation                                         
            if epoch % eval_every == 0 or epoch==1:
                model.eval()
    
                val_loss = []
                val_batches = 0
                y_true_val = []
                y_predicted_val = []
                
                seed_all(0)
                
                with torch.no_grad():
                    for (image, target) in tqdm(val_loader, desc="validating"):
    
                        image = image.to(device)
                        target = target.to(device)

                        # TODO
                        image = T.resize(img = image, size = (128, 128), interpolation =  T.InterpolationMode.BILINEAR).cuda().float()
                        target = T.resize(img = target, size = (128, 128), interpolation =  T.InterpolationMode.NEAREST).long().cuda()
    
                        logits = model(image)["out"]
                        logits = F.upsample(input=logits, size=(
                        target.shape[-2], target.shape[-1]), mode='bilinear')
                        
                        
                        loss = criterion(logits, target)
                                    
                        # Accuracy metrics only on annotated pixels
                        logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                        logits = logits.reshape((-1,args['output_channels']))
                        target = target.reshape(-1)
                        mask = target != -1
                        logits = logits[mask]
                        target = target[mask]
                        
                        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                        target = target.cpu().numpy()
                        
                        val_batches += target.shape[0]
                        val_loss.append((loss.data*target.shape[0]).tolist())
                        y_predicted_val += probs.argmax(1).tolist()
                        y_true_val += target.tolist()
                            
                        
                    y_predicted_val = np.asarray(y_predicted_val)
                    y_true_val = np.asarray(y_true_val)
                    
                    
                    # Save Scores to the .log file and visualize also with tensorboard 
                    
                    acc_val = Evaluation(y_predicted_val, y_true_val)
                    
                logging.info("\n")
                logging.info("Evaluating model..")
                logging.info("Val loss was: " + str(sum(val_loss) / val_batches))
                logging.info("RESULTS AFTER EPOCH " +str(epoch) + ": \n")
                logging.info("Evaluation: " + str(acc_val))

                writer.add_scalars('Loss per epoch', {'Val loss':sum(val_loss) / val_batches, 
                                                      'Train loss':sum(training_loss) / training_batches}, 
                                   epoch)
                
                writer.add_scalar('Precision/val macroPrec', acc_val["macroPrec"] , epoch)
                writer.add_scalar('Precision/val microPrec', acc_val["microPrec"] , epoch)
                writer.add_scalar('Precision/val weightPrec', acc_val["weightPrec"] , epoch)
                writer.add_scalar('Recall/val macroRec', acc_val["macroRec"] , epoch)
                writer.add_scalar('Recall/val microRec', acc_val["microRec"] , epoch)
                writer.add_scalar('Recall/val weightRec', acc_val["weightRec"] , epoch)
                writer.add_scalar('F1/val macroF1', acc_val["macroF1"] , epoch)
                writer.add_scalar('F1/val microF1', acc_val["microF1"] , epoch)
                writer.add_scalar('F1/val weightF1', acc_val["weightF1"] , epoch)
                writer.add_scalar('IoU/val MacroIoU', acc_val["IoU"] , epoch)
    
                if args['reduce_lr_on_plateau'] == 1:
                    scheduler.step(sum(val_loss) / val_batches)
                else:
                    scheduler.step()
                    
                    
                model.train()

           
if __name__ == "__main__":
    args = parse_args()
    main(args)

