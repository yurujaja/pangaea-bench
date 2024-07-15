## What is New
In general, the architecture of the whole codebase is refactored and a few bugs and errors are fixed by the way.


### engines
In engines, basic modules in the training pipeline are defined including data_preprocessor, trainer and evaluator.
1. data_preprocessor replaced the previous adaptation.py, i.e., selects the bands needed by an encoder and pads unavailable bands with zeros.
2. trainer now support mixed precision/distributed training and print training stats and metrics in real time.
3. evaluator can be called independently and evaluate a model also in distributed way and compute per class metrics (Only IoU currently).
4. see run.py for how to assemble these modules and concatenate them


### datasets
1. The implementations are simplified and standardized (I try my best).
2. Dataset metas are read from configs, including newly added classes (name), ignore_index, and so on.
3. Only mados, sen1floods, hlsburnscars are supported by this branch currently. Others are to be completed.
4. To add (register) a new dataset implementation, use the decorator @DATASET_REGISTRY.register().

### foundation_models
1. Remove all irrelevant modules and functions used in pre-training. Only keep the essential modules in encoders for extracting features.
2. Support multi-stage output that may be needed by segmentors, specified by output layers in encoder config.
3. Only Prithvi, ScaleMAE, RemoteCLIP, CROMA are supported by this branch currently. Others are to be completed.
4. To add (register) a new encoder implementation, use the decorator @ENCODER_REGISTRY.register().

### segmentors
1. Now the UperNet implementation is based on mmsegmentation, which is more likely correct: https://github.com/open-mmlab/mmsegmentation/tree/main
2. We can copypaste more segmentors later.
3. To add (register) a new encoder implementation, use the decorator @SEGMENTOR_REGISTRY.register().

## What is still missing
1. Add the other datasets and foundation models following the existing examples in this codebase. Meanwhile, check the correctness of the original datasets before copypasting. 
2. Data augmentation need to be done. We can wrap the dataset class by a configurable augmentor to perform both data preprocessing and augmentation. In this way we avoid preprocessing data in the main process, which is slow.
3. The structure of all the config files should be discussed and standardized. The way to load them can be improved as well.
4. Add more options of optimizer, scheduler and so on. Add more evaluation metrics.  
5. Implement test only mode.
6. Possibly we will add an option of using wandb, which help track all the experiments and share numbers.
Any other suggestions will be highly appreciated


## Setup
Should be the same as the old version of code, maybe some dependencies can be removed


## Example
### Training
Now running config is given in terminal directly for fast hyperparameter tuning, we may add run config file back later
Set `encoder_config`, `dataset_config`, and  `segmentor_config` and start the training process on single gpu:
```
torchrun --nnodes=1 --nproc_per_node=1 run.py  \
--encoder_config configs/foundation_models/prithvi.yaml  \
--dataset_config configs/datasets/mados.yaml   \
--segmentor_config configs/segmentors/upernet.yaml \
--num_workers 4 --eval_interval 1
```

To use more gpus or nodes, set and --nnodes and --nproc_per_node correspondingly, see:
https://pytorch.org/docs/stable/elastic/run.html

To use mixed precision training, specify either --fp16 for float16 and or --bf16 for bfloat16

For fine-tuning instead of linear probing, specify --finetune.

## Some numbers

| Encoder | Dataset      | Epochs | mIoU   |
|---------|--------------|--------|--------|
| Prithvi | MADOS        | 80     | 53.455 |
| Prithvi | HLSBurnScars | 80     | 86.208 |
| Prithvi | Sen1Floods11 | 80     | 87.217 |


