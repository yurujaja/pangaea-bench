[![Tests](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml)

## 📚 Introduction

While geospatial foundation models (GFMs) have proliferated rapidly, their evaluations remain inconsistent and narrow. Existing works often utilize suboptimal downstream datasets (e.g., EuroSAT) and tasks (e.g., land cover classification), which constrain comparability and real-world usability. Additionally, a lack of diversity in evaluation protocols, including image resolution and sensor types, further complicates the extensive assessments of GFM performance. To bridge this gap, we propose a standardized evaluation protocol that incorporates a wide-ranging selection of datasets, tasks, resolutions, and sensor types, establishing a robust and widely applicable benchmark for GFMs.

In this repo, you can find the code to benchmark GFMs. For the moment we included several GFMs that present different approach. We look forward to adding new models and datasets.

For the moment, we support the following **models**:

|             | Paper | GitHub | Keywords |
|:-----------:|:-----:|:------:|:--------:|
|  SSL4EOS12  |       |        |          |
|  Scale-MAE  |       |        |          |
|  SatlasNet  |       |        |          |
|     GFM     |       |        |          |
| SpectralGPT |       |        |          |
|     DOFA    |       |        |          |
|    CROMA    |       |        |          |
|   Prithvi   |       |        |          |
|  RemoteCLIP |       |        |          |

And the following **datasets**:

|                     | Paper | Download | Domain | Task | Sensors | Location |
|:-------------------:|:-----:|:--------:|:------:|:----:|---------|----------|
|    HLS Burn Scars   |       |          |        |      |         |          |
|        MADOS        |       |          |        |      |         |          |
|        PASTIS       |       |          |        |      |         |          |
|     Sen1Floods11    |       |          |        |      |         |          |
|        xView2       |       |          |        |      |         |          |
| Five Billion Pixels |       |          |        |      |         |          |
|   DynamicEarthNet   |       |          |        |      |         |          |
|   CropTypeMapping   |       |          |        |      |         |          |
|      SpaceNet7      |       |          |        |      |         |          |
|    AI4SmallFarms    |       |          |        |      |         |          |
|     BioMassters     |       |          |        |      |         |          |
  
The repository supports the following **tasks**:
 - unitemporal semantic segmentation
 - multi-temporal semantic segmentation
 - unitemporal regression
 - multi-temporal regression
 - change detection

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

**Change Detection** 
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

To use more gpus or nodes, set `--nnodes` and `--nproc_per_node` correspondingly, see:
https://pytorch.org/docs/stable/elastic/run.html

To use mixed precision training, specify either `--fp16` for float16 and or `--bf16` for bfloat16

## 🏃 Evaluation 
Indicate the `eval_dir` where the checkpoints and configurations are stored.
```
torchrun --nnodes=1 --nproc_per_node=1 run.py --batch_size 1 --eval_dir work-dir/the-folder-where-your-exp-is-saved
```

## ✏️ Contributing
We appreciate all contributions to improve xxx. Please refer to [Contributing Guidelines](.github/CONTRIBUTING.md)

## ⚠️ Warnings

Some features are under construction:
 - the automatic download is working for all the datasets and models' weights but, respectively, **Five Billion Pixels** and **GFM**.


## 🧮 Some first results

A pre-print is coming soon... Stay tuned to read it

| Encoder | Dataset      | Epochs | mIoU   |
|---------|--------------|--------|--------|
| Prithvi | MADOS        | 80     | 53.455 |
| Prithvi | HLSBurnScars | 80     | 86.208 |
| Prithvi | Sen1Floods11 | 80     | 87.217 |

Please note: 

## 💡 Acknowledgements

##  ©️ License

## 📝 Citing
