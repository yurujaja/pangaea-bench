[![Tests](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml)

## Introduction
(TBD)

### engines
In engines, basic modules in the training pipeline are defined including data_preprocessor, trainer and evaluator.
1. data_preprocessor replaced the previous adaptation.py, i.e., selects the bands needed by an encoder and pads unavailable bands with zeros, and different augmentations.
2. trainer now support mixed precision/distributed training and print training stats and metrics in real time.
3. evaluator can be called independently and evaluate a model also in distributed way and compute per class metrics.
4. see run.py for how to assemble these modules and concatenate them

### datasets
1. The implementations are simplified and standardized (I try my best).
2. Dataset metas are read from configs, including newly added classes (name), ignore_index, and so on.
3.Mados, sen1floods, hlsburnscars, xView2, biomasster are supported by this branch currently.
4. To add (register) a new dataset implementation, use the decorator @DATASET_REGISTRY.register().

### foundation_models
1. Remove all irrelevant modules and functions used in pre-training. Only keep the essential modules in encoders for extracting features.
2. Support multi-stage output that may be needed by segmentors, specified by output layers in encoder config.
3. All the encoder should work properly.
4. To add (register) a new encoder implementation, use the decorator @ENCODER_REGISTRY.register().

### segmentors
1. Now the UperNet implementation is based on mmsegmentation, which is more likely correct: https://github.com/open-mmlab/mmsegmentation/tree/main
2. We can copypaste more segmentors later.
3. To add (register) a new encoder implementation, use the decorator @SEGMENTOR_REGISTRY.register().
4. So far, we have UPerNet for unitemporal semantic segmentation, UPerNetCD for change detection and MTUPerNet for multitemporal semantic segmentation
5. for multi-temporal, L-TAE and linear projection are supported

All of these parameters can also be set in the run config file.

To use more gpus or nodes, set `--nnodes` and `--nproc_per_node` correspondingly, see:
https://pytorch.org/docs/stable/elastic/run.html

To use mixed precision training, specify either `--fp16` for float16 and or `--bf16` for bfloat16

For fine-tuning instead of linear probing, specify `--finetune`.

## 🛠️ Setup
Clone the repository:
```
git clone git@github.com:yurujaja/geofm-bench.git
cd geofm-bench
```

**Dependencies**

Use either Conda or Mamba:
```
conda env create -f environment.yaml
conda activate geofm-bench8
```

Optional: install [Mamba](https://github.com/conda-forge/miniforge/releases/) for faster resolution times
```
wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh
./Mambaforge-24.3.0-0-Linux-x86_64.sh

mamba env create -f environment.yaml
mamba activate geofm-bench8
```

## 🏋️ Training
There are 5 basic component types in our config system:
- `config`: Information of training settings such as batch size, epochs, use wandb. `limited_label` is to indicate the percentage of dataset used for training, for example, `-1` means the full training dataset is used while `0.5` means 50% used. 
- `encoder_config`: GFM encoder related parameters. `output_layers` is used for which layers are used for Upernet decoder. 
- `dataset_config`: Information of downstream datasets such as image size, band_statistics, etc. 
- `segmentor_config`: Downstream task decoder fine-tuning related parameters, including the head type, loss, optimizer, scheduler, etc.
- `augmentation_config`: Both preprocessing and augmentations steps required for the dataset, such as bands adaptation, normalization, resize/crop.

We provide several examples of command lines to initilize different training tasks on single gpu.
### 💻 Decoder Finetuning
**Single Temporal Semantic Segmentation** 

Take MADOS dataset, Prithvi Encoder and Upernet Decoder as example:
```
torchrun --nnodes=1 --nproc_per_node=1 run.py  \
--config configs/run/default.yaml  \
--encoder_config configs/foundation_models/prithvi.yaml  \
--dataset_config configs/datasets/mados.yaml   \
--segmentor_config configs/segmentors/upernet.yaml \
--augmentation_config configs/augmentations/segmentation_default.yaml  \
--num_workers 4 --eval_interval 1  --use_wandb
```

**Multi Temporal Semantic Segmentation**

Multi-temporal model `configs/segmentors/upernet_mt.yaml` should be used. In addition, in the dataset config, indicate the number of time frames, e.g., `multi_temporal: 6`
```
torchrun --nnodes=1 --nproc_per_node=1 run.py  \
--config configs/run/default.yaml  \
--encoder_config configs/foundation_models/prithvi.yaml  \
--dataset_config configs/datasets/croptypemapping.yaml   \
--segmentor_config configs/segmentors/upernet_mt.yaml \
--augmentation_config configs/augmentations/ctm.yaml  \
--num_workers 4 --eval_interval 1 --use_wandb
```

**Multi Temporal Change Detection** 
```
torchrun ...
```

**Multi Temporal Regression** 
```
torchrun ...
```

### 💻 Fully Supervised Training
**Single Temporal Change Detection** 
```
torchrun ...
```
## 🏃 Evaluation 
Indicate the `eval_dir` where the checkpoints and configurations are stored.
```
torchrun --nnodes=1 --nproc_per_node=1 run.py --batch_size 1 --eval_dir work-dir/the-folder-where-your-exp-is-saved
```


## ✏️ Contributing
We appreciate all contributions to improve xxx. Please refer to [Contributing Guidelines](.github/CONTRIBUTING.md)




## Some numbers

| Encoder | Dataset      | Epochs | mIoU   |
|---------|--------------|--------|--------|
| Prithvi | MADOS        | 80     | 53.455 |
| Prithvi | HLSBurnScars | 80     | 86.208 |
| Prithvi | Sen1Floods11 | 80     | 87.217 |

## 💡 Acknowledgements
