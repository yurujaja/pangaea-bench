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
- The original dataset contains corrupted files, which are skipped during the experiment. We follow the dataset paper to use the most frequent 4 classes and the others are ignored.
- The basic experimental setup for this dataset is a multi-temporal multi-modal semantic segmentation task. For models that don't support multi-temporal data, each time frame is processed separately for feature extraction and then mapped into a single representation. This setup requires the configuration file `configs/segmentors/upernet_mt.yaml`. Additionally, in the dataset configuration, specify the number of time frames, for example, `multi_temporal: 6`, where the latest six images are selected for both optical and SAR data. Below is a CLI example for running the experiment using the CROMA pretrained encoder, which jointly processes optical and SAR information:

  ```
###  AI4SmallFarms
- The code supports automatic downloading of the dataset into `./data` folder.
- The original dataset contains vector files as well as Google Maps (GM) files, which are skipped during the experiment. Only the .tif Sentinel-2 images and delineation labels are kept after downloading.
- The dataset is uni-temporal, and the labels contain only two classes (farm boundary or background). For training using the Prithvi encoder, the following command should be used:
  ```
  torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --config "configs/run/default.yaml" \
  --encoder_config "configs/foundation_models/prithvi.yaml" \
  --dataset_config "configs/datasets/ai4smallfarms.yaml" \
  --segmentor_config "configs/segmentors/upernet_binary.yaml" \
  --augmentation_config "configs/augmentations/ai4smallfarms.yaml" \
  --use_wandb
  ```

  ```
