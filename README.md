
### Progress
Currently supported foundation models:
- Prithvi
- SpectralGPT
- Scale-MAE
- RemoteCLIP 
- SSL4EO (data2vec, MoCov3, DINO and MAE)
- DOFA
- GFM
- CROMA
- SatlasNet

Currently supported tasks:
- Upernet for semantic segmentation (also multitemporal)
- Change Detection (bitemporal)

## Setup
Clone the repository:
```
git clone git@github.com:yurujaja/geofm-bench.git
cd geofm-bench
```

Dependencies:
```
conda env create -f environment.yaml
conda activate geofm-bench3
```

Optional: install Mamba (https://github.com/conda-forge/miniforge/releases/) for faster resolution times
```
wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh
./Mambaforge-24.3.0-0-Linux-x86_64.sh

mamba env create -f environment.yaml
mamba activate geofm-bench3
```

### Download pre-trained weights
For using GFM please download pretrained weights into the `pretrained_models` folder manually.
```
mkdir pretrained_models
cd pretrained_models

# GFM
You can find the links in their official repository 
https://github.com/boranhan/Geospatial_Foundation_Models?tab=readme-ov-file#geopile-and-gfm-pretrained-model

```

## Tests
To run our unit tests, simply run
```
python -m unittest
```

Warning: This will download all pretrained model files, and all datsets.

You can also choose to run subsets of tests:
```
# Run a test module:
python -m unittest tests.test_models

# Run a test collection:
python -m unittest tests.test_datasets.testDatasetSetup
```

## Pipeline - demo
To quickly get started, utilize [MADOS dataset](https://zenodo.org/records/10664073) to establish the complete pipeline for semantic segmentation.
### Training
**Option1**: Configure all your pipeline params in `configs/run.yaml`, set `encoder_config`, `dataset_config`, and  `task_config`. Then, start the training process by running:
```
python train.py configs/run.yaml
```

**Option2**: Specify your configuration directly through command-line arguments as follows:
```
python train.py configs/run.yaml \
    --encoder_config configs/models_config/prithvi.yaml \
    --dataset_config configs/datasets_config/mados.yaml \
    --task_config configs/tasks_config/upernet.yaml
```

### Evaluate
Provide the checkpoint of the trained decoder for inference. Change `mode` to `test` in `configs/run.yaml` and provide the checkpoint path through the argument:
```
python train.py configs/run.yaml  --ckpt_path work-dir/your_exp/your_checkpoint_id.pth
```


#### Note:
- **Configurations**: The current configurations include parameters related to foundation model encoders and downstream task models. Future updates will aim to enhance configuration files to support additional tasks. To support multitemporal, please modify the `num_frames` parameter in the config. Consider that in all the configs, it appears in the `task` parameters. For Prithvi it appears also in the `encoder` parameter.
- **Logging**: By default, logs and checkpoints are stored in the `work_dir`.
- **The Mados dataset** in use is a simple example that only iterates over the first few data items. To do so, we added the following line 156 in `datasets/mados.py`. 
    ```
    if crop_name in self.ROIs_split[:2]:
    ```
- **Design Choices**: to make the comparison fairer we have implemented (so far) the two following solutions: 
    - L-TAE is the choice for combining the multitemporal information not in a vanilla way (a linear layer is used in this case). Please consider that Prithvi and SatlasNet have their own multitemporal encoding choices, so in those cases L-TAE (and linear) are not used
    - We inserted a FLOPs/MACs computation. In fact, different models can have different sizes and hyperparameters, so comparing the performances without considering the needed computation would be a limit. For example, Prithvi pretrained model has 1/4 of GFLOPs w.r.t. SpectralGPT pretrained model (e.g. SpectralGPT uses a patch size of 8 w.r.t. Prithvi that uses 16). We can also consider adding inference times when we will develop the test.py
    
###  How to Contribute

#### New code
- **Datasets**: Add your dataset code within the `datasets` folder. 
    - In the `__getitem__` function-, the output should have a dict structure like below:
        ```
        output = {
                'image': {
                    'optical':optical_image,
                    'sar': sar_image
                },
                'target': target,
                'metadata': {}
            }
        return output
        ```
    - Add a config file in `configs/datasets_config`.
    - Your dataset should implement a `get_splits(dataset_config)` static method, that returns three dataset splits: train, validation, and test. For examples see the existing datasets.
    - It is also highly advised that your dataset implements a `download(dataset_config)` static method, that automates dataset download. This might not be required, eg. if your dataset is streamed from an online source.


#### Existing code

TODO: here are some aspects that should be improved:
- new tasks:

    - support multimodality for croma and dofa (should be easy)
    - support pixel level regression (should be easy, changing the loss when using upernet)

- fix some model bugs:
    - **WARNINGS**
    - `dofa`: I. check the match of checkpoints II. remove the hard coded `wave_list` in `upernet.py`
    - `scale_mae`: remove hard coding for `input_res` (spatial resolution of input)
    - `croma`: just support 12 channels for optical (`in_chans` parameters is not used. We have to figure out if we should change or leave it like that)
    - `ssl4eo_mae`, `gfm`: check the match of the checkpoints (i.e. missing keys and unexpected keys)
    - for multitemporal, `prithvi` config's `num_frames` parameter needs to be updated. This is redundant with the task config
    - **CODING STYLE**
    - in `ssl4eo_moco` and `ssl4eo_dino` there are almost the same function for the ViT, we can move in a common file
    - `dofa` and `scale_mae` are not using positional embedding functions from pos_embed.py because of dtype issues when importing these functions (Double vs Float)

- the training loop (`train.py`) should be improved to support also change detection (should be easy)

- improve the `adapt_input` function (in `train.py`), which is used to adapt the input shape of the data to be processed into the models 
    - At the moment, it supports just the mentioned models (and dataset) -> NEW MODELS/ Datasets TO BE ADDED
    - Moreover, for selecting the correct number of bands, just Sentinel-2 is supported -> TO SHAPE IT ALSO FOR OTHER MODALITIES