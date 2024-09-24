[![Tests](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml)

# TITLE 

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
We provide several ways to install the dependencies.

1. Using either Conda or Mamba:
    ```
    conda env create -f environment.yaml
    conda activate geofm-bench
    ```

    Optional: install [Mamba](https://github.com/conda-forge/miniforge/releases/) for faster resolution times
    ```
    wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh
    sh ./Mambaforge-24.3.0-0-Linux-x86_64.sh

    mamba env create -f environment.yaml
    mamba activate geofm-bench
    ```

2. Using pip, create a Python native virtual environment and install dependencies into it:
    ```
    export GFMBENCH_PATH=/path/to/venv/geofm-bench  # change this
    python3 -m venv ${GFMBENCH_PATH}
    source ${GFMBENCH_PATH}/bin/activate

    pip install -r requirements.txt
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
torchrun ...
```
#### Single Temporal Regression
```
torchrun ...
```

#### Multi-Temporal Regression
```
torchrun ...
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

| Encoder | Dataset       | Epochs | mIoU   |
|---------|---------------|--------|--------|
| Prithvi | MADOS         | 80     | 53.455 |
| Prithvi | HLSBurnScars  | 80     | 86.208 |
| Prithvi | Sen1Floods11  | 80     | 87.217 |
| Prithvi | AI4SmallFarms | 80     | 33.796 |

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
