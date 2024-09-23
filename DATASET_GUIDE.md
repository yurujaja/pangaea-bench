# Dataset Guide

This document provides a detailed overview of the datasets used in this repository. For each dataset, you will find instructions on how to prepare the data, along with command-line examples for running models. 

###  Sen1Floods11
- The code supports automatic downloading of the dataset into `./data` folder. 
- The basic experiment uses mean and std values for nomralization and applies random cropping to align images with the size used for GFMs pretraining.
   Below is a CLI example for running the experiment with the Prithvi pretrained encoder and UperNet segmentation decoder:

  ```
  torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --config "configs/run/default.yaml" \
  --encoder_config "configs/foundation_models/prithvi.yaml" \
  --dataset_config "configs/datasets/sen1floods11.yaml" \
  --segmentor_config "configs/segmentors/upernet.yaml" \
  --augmentation_config "configs/augmentations/segmentation_default.yaml" \
  --use_wandb
  ```
### xView2
- The dataset needs to be downloaded manually from the official website. This requires a registration and accepting the terms and conditions. On the download page, we need the datasets under `Datasets from the Challenge`, excluding the holdout set. Extract the datasets in the `./data/xView2/` folder, such that it contains e.g. `./data/xView2/tier3/images/...`.
- The `tier3` set does not come up labels in the form of images, so we first need to create them from the respective JSON data. We create a `masks` folder on the level of the `images` folder by running:

  ```
   python datasets/xView2_create_masks.py
   ```
- The basic experimental setup for this dataset is a change detection task. Two images showing the same location are encoded using a foundation model as encoder. A smaller UPerNet model is trained to compute the 5-class segmentation mask from these encodings. Below is a CLI example for running the experiment with the Prithvi pretrained encoder:
   ```
   torchrun --nnodes=1 --nproc_per_node=1 run.py  \
   --config configs/run/default.yaml  \
   --encoder_config configs/foundation_models/prithvi.yaml  \
   --dataset_config configs/datasets/xview2.yaml   \
   --segmentor_config configs/segmentors/siamdiffupernet.yaml \
   --augmentation_config configs/augmentations/segmentation_oversampling.yaml  \
   --use_wandb
   ```
###  Crop Type Mapping (South Sudan)
- The code supports automatic downloading of the dataset into `./data` folder. 
- The most frequent 4 classes are considered and the others are ignored. The dataset contains varied length of timeseries data from different sensor types, we select the most latest six images with the corresponding date metadata. For models that don't support multi-temporal data, each time frame is processed separately for feature extraction and then mapped into a single representation. Below is a CLI example for running the experiment with the CROMA pretrained encoder which jointly process optical and sar information:

  ```
  
  ```