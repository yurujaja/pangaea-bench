# Dataset Guide

This document provides a detailed overview of the datasets used in this repository. For each dataset, you will find instructions on how to prepare the data, along with command-line examples for running models. 

###  Sen1Floods11
- The code supports automatic downloading of the dataset into `./data` folder. 
- The basic experiment uses mean and std values for nomralization and applies random cropping to align images with the size used for GFMs pretraining.
   Below is a CLI example for running the experiment with the Prithvi pretrained encoder:
  ```
  torchrun --nnodes=1 --nproc_per_node=1 run.py \
      --config "configs/run/default.yaml" \
      --encoder_config "configs/foundation_models/prithvi.yaml" \
      --dataset_config "configs/datasets/sen1floods11.yaml" \
      --segmentor_config "configs/segmentors/upernet.yaml" \
      --augmentation_config "configs/augmentations/segmentation_default.yaml" \
      --use_wandb
  ```
