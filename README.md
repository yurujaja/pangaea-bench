[![Tests](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml)

# TITLE 

## 📚 Introduction

While geospatial foundation models (GFMs) have proliferated rapidly, their evaluations remain inconsistent and narrow. Existing works often utilize suboptimal downstream datasets (e.g., EuroSAT) and tasks (e.g., land cover classification), which constrain comparability and real-world usability. Additionally, a lack of diversity in evaluation protocols, including image resolution and sensor types, further complicates the extensive assessments of GFM performance. To bridge this gap, we propose a standardized evaluation protocol that incorporates a wide-ranging selection of datasets, tasks, resolutions, and sensor types, establishing a robust and widely applicable benchmark for GFMs.

In this repo, you can find the code to benchmark GFMs. For the moment we included several GFMs that present different approach. We look forward to adding new models and datasets.

For the moment, we support the following **models**:

|             | Paper | GitHub | Keywords |
|:-----------:|:-----:|:------:|:--------:|
|  [SSL4EOS12](https://arxiv.org/abs/2211.07044)  | SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal <br> Dataset for Self-Supervised Learning in Earth Observation      | [link](https://github.com/zhu-xlab/SSL4EO-S12) | DINO, MAE, DATA2VEC, MOCO|
|  [Scale-MAE](https://arxiv.org/pdf/2212.14532)  | Scale-MAE: Scalable Masked Autoencoders for Self-Supervised Learning on Climate Datasets      | [link](https://github.com/bair-climate-initiative/scale-mae) | Masked Autoencoders, Multiscale|
|  [SatlasNet](https://arxiv.org/pdf/2211.15660)  | SatlasNet: A Spatio-Temporal Atlas for Global Mapping from Satellite Images | [link](https://github.com/allenai/satlas/tree/main) | Supervised, Multi-temporal |
|  [GFM](https://arxiv.org/pdf/2404.01260)        | GFM: Generalized Foundation Models for Climate Science | [link](https://github.com/mmendiet/GFM) | |
|  [SpectralGPT](https://arxiv.org/abs/2311.07113) | SpectralGPT: Generative Pretrained Transformer for Hyperspectral Image Analysis      | [link](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT) | MAE, Multi-spectral |
|  [DOFA](https://arxiv.org/pdf/2403.15356)       | DOFA: Dynamic Object Feature Aggregation for Self-Supervised Learning in Satellite Data      | [link](https://github.com/zhu-xlab/DOFA) | MAE, Dynamic bands |
|  [CROMA](https://arxiv.org/pdf/2311.00566)      | CROMA: Cross-Modal Alignment for Satellite Image Analysis      | [link](https://github.com/antofuller/CROMA) | Contrastive Learning, MAE |
|  [Prithvi](https://arxiv.org/pdf/2310.18660)    | Prithvi: Foundation Models for Earth Observation Data      | [link](https://github.com/NASA-IMPACT/hls-foundation-os) | MAE, Multi-temporal |
|  [RemoteCLIP](https://arxiv.org/pdf/2306.11029) | RemoteCLIP: A Contrastive Language-Image Pretraining Model for Remote Sensing      | [link](https://github.com/ChenDelong1999/RemoteCLIP) | Contrastive Learning |


And the following **datasets**:

|                     | Download | Domain | Task | Sensors | Location |
|:-------------------:|:--------:|:------:|:----:|:-------:|:--------:|
|    HLS Burn Scars   |          |        |      |         | Global   |
|        MADOS        |          |        |      |         | Global   |
|        PASTIS       |          |        |      |         | France   |
|     [Sen1Floods11](http://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html)    | [link](https://github.com/cloudtostreet/Sen1Floods11) |  Flood |Semantic Segmentation  | S1, S2 | Global |
|        [xView2](https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html)       | [link](https://xview2.org/dataset) | HADR | Semantic Segmentation | Maxar | Global   |
| Five Billion Pixels |          |        |      |         | China    |
|   DynamicEarthNet   |          |        |      |         | Global   |
|   [CropTypeMapping](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.pdf) |   [link](https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_type_mapping_ghana-ss.html#download) | Agriculture |Semantic Segmentation |S1, S2, Planet|South Sudan|
|      SpaceNet7      |          |        |      |         | Global   |
|    AI4SmallFarms    |          |        |      |         | Cambodia/Vietnam |
|     [BioMassters](https://papers.nips.cc/paper_files/paper/2023/file/40daf2a00278c4bea1b26cd4c8a654f8-Paper-Datasets_and_Benchmarks.pdf)     |   [link](https://huggingface.co/datasets/nascetti-a/BioMassters)       | Forest       | Regression   |  S1, S2 | Finland   |


Please refer to [**Dataset Guide**](DATASET_GUIDE.md) to understand the processing requirements and commands specific to each dataset.

The repository supports the following **tasks** using GFMs:
 - [single temporal semantic segmentation](#single-temporal-semantic-segmentation)
 - [multi-temporal semantic segmentation](#multi-temporal-semantic-segmentation)
 - [change detection](#change-detection)
 - [single temporal regression](#single-temporal-regression)
 - [multi-temporal regression](#multi-temporal-regression)

It is possible also to train some [supervised baselines](#-fully-supervised-training), based on UNet.

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

- `config`: Information of training settings such as batch size, epochs, use wandb. `limited_label` is to indicate the percentage of dataset used for training, for example, `-1` means the full training dataset is used while `0.5` means 50% used. #strategy used
- `encoder_config`: GFM encoder related parameters. `output_layers` is used for which layers are used for Upernet decoder. 
- `dataset_config`: Information of downstream datasets such as image size, band_statistics, etc. 
- `segmentor_config`: Downstream task decoder fine-tuning related parameters, including the head type, loss, optimizer, scheduler, etc.
- `augmentation_config`: Both preprocessing and augmentations steps required for the dataset, such as bands adaptation, normalization, resize/crop.

We provide several examples of command lines to initilize different training tasks on single GPU.

Please note:
 - Command line's parameters have the priority on the parameters in the config files. So, if you want to change e.g. the `batch size`, without changing the `config`, you can just add `--batch size n` to the command line
 - To use more gpus or nodes, set `--nnodes` and `--nproc_per_node` correspondingly, see:
https://pytorch.org/docs/stable/elastic/run.html
 - To use mixed precision training, specify either `--fp16` for float16 and or `--bf16` for bfloat16

### 💻 Decoder Finetuning
#### Single Temporal Semantic Segmentation

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

#### Multi-Temporal Semantic Segmentation

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

#### Change Detection
```
torchrun --nnodes=1 --nproc_per_node=1 run.py  \
--config configs/run/default.yaml  \
--encoder_config configs/foundation_models/prithvi.yaml  \
--dataset_config configs/datasets/xview2.yaml   \
--segmentor_config configs/segmentors/siamdiffupernet.yaml \
--augmentation_config configs/augmentations/segmentation_oversampling.yaml  \
--num_workers 4 --eval_interval 1 --use_wandb
```
#### Single Temporal Regression
```
torchrun --nnodes=1 --nproc_per_node=1 run.py  \
--config configs/run/default.yaml  \
--encoder_config configs/foundation_models/prithvi.yaml  \
--dataset_config configs/datasets/biomassters.yaml   \
--segmentor_config configs/segmentors/reg_upernet.yaml \
--augmentation_config configs/augmentations/regression_default.yaml  \
--num_workers 4 --eval_interval 1 --use_wandb
```

#### Multi-Temporal Regression
```
torchrun --nnodes=1 --nproc_per_node=1 run.py  \
--config configs/run/default.yaml  \
--encoder_config configs/foundation_models/prithvi.yaml  \
--dataset_config configs/datasets/biomassters.yaml   \
--segmentor_config configs/segmentors/reg_upernet_mt.yaml \
--augmentation_config configs/augmentations/regression_default.yaml  \
--num_workers 4 --eval_interval 1 --use_wandb
```

### 💻 Fully Supervised Training
#### Single Temporal Semantic Segmentation
```
torchrun ...
```
In general


## 🏃 Evaluation 
Indicate the `eval_dir` where the checkpoints and configurations are stored.

```
torchrun --nnodes=1 --nproc_per_node=1 run.py --batch_size 1 --eval_dir work-dir/the-folder-where-your-exp-is-saved
```

## ✏️ Contributing
We appreciate all contributions. Please refer to [Contributing Guidelines](.github/CONTRIBUTING.md)

## ⚠️ Warnings

Some features are under construction:
 - the automatic download is working for all the datasets and models' weights but, respectively, **Five Billion Pixels**, **BioMassters**, and **GFM**.


## 🧮 Some first results

A pre-print is coming soon... Stay tuned!

| Encoder | Dataset      | Epochs | mIoU   |
|---------|--------------|--------|--------|
| Prithvi | MADOS        | 80     | 53.455 |
| Prithvi | HLSBurnScars | 80     | 86.208 |
| Prithvi | Sen1Floods11 | 80     | 87.217 |

Please note: #add different conditions 

## 💡 Acknowledgements

##  ©️ License

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE

## 📝 Citing

If you use this software in your work, please cite:

```
@misc{pangaea,
  author = {},
  title = {Pangaea},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yurujaja/geofm-bench}},
}
```
