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
| [HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) | [link](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) | Wildfire | Semantic Segmentation | HLS (Harmonized Landsat Sentinel-2) | Global |
|        MADOS        |          |        |      |         | Global   |
|        PASTIS       |          |        |      |         | France   |
|     [Sen1Floods11](http://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html)    | [link](https://github.com/cloudtostreet/Sen1Floods11) |  Flood |Semantic Segmentation  | S1, S2 | Global |
|        [xView2](https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html)       | [link](https://xview2.org/dataset) | HADR | Semantic Segmentation | Maxar | Global   |
| Five Billion Pixels |          |        |      |         | China    |
|   DynamicEarthNet   |          |        |      |         | Global   |
|   [CropTypeMapping](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.pdf) |   [link](https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_type_mapping_ghana-ss.html#download) | Agriculture |Semantic Segmentation |S1, S2, Planet|South Sudan|
|      [SpaceNet 7](https://openaccess.thecvf.com/content/CVPR2021/papers/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.pdf)      |    [link](https://spacenet.ai/sn7-challenge/)      |    Urban    |   Change detection   |     Planet    | Global   |
|    [AI4SmallFarms](https://ieeexplore.ieee.org/document/10278130)  | [link](https://doi.org/10.17026/dans-xy6-ngg6)  |  Agriculture     |  Semantic segmentation  |   S2   | Cambodia/Vietnam |
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

We provide several ways to install the dependencies.

1. **Using either Conda or Mamba**:
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

2. **Using pip**, create a Python native virtual environment and install dependencies into it:
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

## 🔧 Customization

### Using Your Own Dataset

We have designed the repo to allow for using your own datasets with minimal effort. Follow the steps below to integrate your dataset:

1. **Create a Dataset Configuration File**:

   - Navigate to `configs/datasets/` and create a new YAML file named after your dataset (e.g., `my_dataset.yaml`).
   - Define all necessary dataset parameters such as `dataset_name`, `root_path`, `img_size`, `bands`, `data_mean`, `data_std`, `num_classes`, and class labels.
   - **Example**:

     ```yaml
     dataset_name: MyDataset
     root_path: ./data/my_dataset
     auto_download: False

     img_size: 256
     multi_temporal: False
     multi_modal: False

     ignore_index: -1
     num_classes: 3
     classes:
       - Class1
       - Class2
       - Class3

     bands:
       optical:
         - B1
         - B2
         - B3

     data_mean:
       optical:
         - 0.485
         - 0.456
         - 0.406

     data_std:
       optical:
         - 0.229
         - 0.224
         - 0.225
     ```

2. **Implement a Dataset Class**:

   - In the `datasets/` directory, create a new Python file named after your dataset (e.g., `my_dataset.py`).
   - Implement a class that inherits from `torch.utils.data.Dataset`.
   - Register your dataset with the `@DATASET_REGISTRY.register()` decorator.
   - Implement the required methods: `__init__`, `__len__`, `__getitem__`, `get_splits`, and `download` (if applicable).
   - **Example**:

     ```python
     import torch
     from utils.registry import DATASET_REGISTRY

     @DATASET_REGISTRY.register()
     class MyDataset(torch.utils.data.Dataset):
         def __init__(self, cfg, split):
             self.root_path = cfg['root_path']
             self.split = split
             # Initialize file lists or data structures here

         def __len__(self):
             # Return the total number of samples
             return len(self.file_list)

         def __getitem__(self, index):
             # Load your data and labels here
             image = ...  # Load image
             target = ...  # Load target label or mask

             # Convert to tensors
             image = torch.tensor(image, dtype=torch.float32)
             target = torch.tensor(target, dtype=torch.long)

             return {
                 'image': {'optical': image},
                 'target': target,
                 'metadata': {}
             }

         @staticmethod
         def get_splits(dataset_config):
             train_dataset = MyDataset(cfg=dataset_config, split="train")
             val_dataset = MyDataset(cfg=dataset_config, split="val")
             test_dataset = MyDataset(cfg=dataset_config, split="test")
             return train_dataset, val_dataset, test_dataset

         @staticmethod
         def download(dataset_config, silent=False):
             # Implement if your dataset requires downloading
             pass
     ```

3. **Adjust the Augmentation Pipeline**:

   - If your dataset requires specific preprocessing or augmentation, create or modify an augmentation configuration file in `configs/augmentations/`.
   - Ensure that all preprocessing steps (e.g., normalization, resizing) match your dataset's requirements.

4. **Run Training**:

   - Use the `run.py` script with your dataset and augmentation configurations.
   - **Example Command**:

     ```bash
     torchrun --nnodes=1 --nproc_per_node=1 run.py \
     --config configs/run/default.yaml \
     --encoder_config configs/foundation_models/prithvi.yaml \
     --dataset_config configs/datasets/my_dataset.yaml \
     --segmentor_config configs/segmentors/upernet.yaml \
     --augmentation_config configs/augmentations/segmentation_default.yaml \
     --num_workers 4 --eval_interval 1 --use_wandb
     ```

### Using Your Own Model

To benchmark your own foundation model, follow these steps:

1. **Create an Encoder Configuration File**:

   - In `configs/foundation_models/`, create a new YAML file named after your model (e.g., `my_model.yaml`).
   - Define model-specific parameters, including `encoder_name`, `foundation_model_name`, `encoder_weights`, `input_bands`, and any model architecture arguments.
   - **Example**:

     ```yaml
     encoder_name: MyModel_Encoder
     foundation_model_name: MyModel
     encoder_weights: ./pretrained_models/my_model_weights.pth
     download_url: https://path.to.your.model/weights.pth
     temporal_input: False

     encoder_model_args:
       img_size: 224
       in_chans: 3
       embed_dim: 768
       patch_size: 16
       num_heads: 12
       depth: 12
       mlp_ratio: 4

     input_bands:
       optical:
         - B1
         - B2
         - B3

     output_layers:
       - 3
       - 5
       - 7
       - 11
     ```

2. **Implement an Encoder Class**:

   - In `foundation_models/`, create a new Python file named after your model (e.g., `my_model_encoder.py`).
   - Implement a class that inherits from `nn.Module`.
   - Register your encoder with the `@ENCODER_REGISTRY.register()` decorator.
   - Implement the required methods: `__init__`, `load_encoder_weights`, and `forward`.
   - **Example**:

     ```python
     import torch.nn as nn
     from utils.registry import ENCODER_REGISTRY

     @ENCODER_REGISTRY.register()
     class MyModel_Encoder(nn.Module):
         def __init__(self, cfg, **kwargs):
             super().__init__()
             self.model_name = 'MyModel'
             # Initialize your model architecture here
             # For example:
             self.backbone = nn.Sequential(
                 nn.Conv2d(cfg.encoder_model_args['in_chans'], 64, kernel_size=3, padding=1),
                 nn.ReLU(),
                 # Add more layers as needed
             )
             # Specify output layers if applicable
             self.output_layers = cfg['output_layers']

         def load_encoder_weights(self, pretrained_path):
             # Load pretrained weights
             state_dict = torch.load(pretrained_path, map_location='cpu')
             self.load_state_dict(state_dict, strict=False)
             print(f"Loaded encoder weights from {pretrained_path}")

         def forward(self, image):
             x = image['optical']
             outputs = []
             # Forward pass through the model
             for idx, layer in enumerate(self.backbone):
                 x = layer(x)
                 if idx in self.output_layers:
                     outputs.append(x)
             return outputs
     ```

3. **Adjust Band Mapping**:

   - Ensure that the `input_bands` in your encoder configuration match the bands used in your model.
   - Update the `input_bands` section in your encoder YAML file if necessary.

4. **Run Training with Your Model**:

   - Use the `run.py` script with your encoder configuration.
   - **Example Command**:

     ```bash
     torchrun --nnodes=1 --nproc_per_node=1 run.py \
     --config configs/run/default.yaml \
     --encoder_config configs/foundation_models/my_model.yaml \
     --dataset_config configs/datasets/mados.yaml \
     --segmentor_config configs/segmentors/upernet.yaml \
     --augmentation_config configs/augmentations/segmentation_default.yaml \
     --num_workers 4 --eval_interval 1 --use_wandb
     ```

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
