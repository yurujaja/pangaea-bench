
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
- SatlasNet (SwinB-backbone)

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
conda env create -f environment.yml
conda activate geofm-bench
```
### Download pre-trained weights
Please download pretrained weights into the `pretrained` folder.
```
mkdir pretrained
# Prithvi
wget https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt?download=true -O pretrained/Prithvi_100M.pt

# SpectralGPT+ 
wget https://zenodo.org/records/8412455/files/SpectralGPT+.pth -O pretrained/SpectralGPT+.pth
# or SpectralGTP
wget https://zenodo.org/records/8412455/files/SpectralGPT.pth -O pretrained/SpectralGPT.pth

# Scale-MAE
wget https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth

# RemoteCLIP Base
wget https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-B-32.pt
# or RemoteCLIP Large
wget https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-L-14.pt

# CROMA Base
wget https://huggingface.co/antofuller/CROMA/blob/main/CROMA_base.pt
# or CROMA Large
wget https://huggingface.co/antofuller/CROMA/blob/main/CROMA_large.pt

# DOFA Base
wget https://huggingface.co/XShadow/DOFA/blob/main/DOFA_ViT_base_e100.pth
# or DOFA Large
wget https://huggingface.co/XShadow/DOFA/blob/main/DOFA_ViT_large_e100.pth

# SSL4EO
You can find all the links in their official repository https://github.com/zhu-xlab/SSL4EO-S12/tree/main

# GFM
You can find the links in their official repository 
https://github.com/boranhan/Geospatial_Foundation_Models?tab=readme-ov-file#geopile-and-gfm-pretrained-model

# SatlasPretrain (currently only support Sentinel2_SwinB_SI_RGB)
You can find the links in their official repository 
https://github.com/allenai/satlaspretrain_models/

```
### Download Data
- Please download [MADOS](https://zenodo.org/records/10664073)  into the `./data/MADOS` folder.
- Please download [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)   into the `./data/Sen1Floods11` folder.


## Pipeline -demo
To quickly get started, utilize [MADOS dataset](https://zenodo.org/records/10664073) to establish the complete pipeline for semantic segmentation.

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

#### Note:
- **Configurations**: The current configurations include parameters related to foundation model encoders and downstream task models. Future updates will aim to enhance configuration files to support additional tasks. To support multitemporal, please modify the `num_frames` parameter in the config. Consider that in all the configs, it appears in the `task` parameters. For Prithvi it appears also in the `encoder` parameter.
- **Logging**: By default, logs and checkpoints are stored in the `work_dir`.
- **The Mados dataset** in use is a simple example that only iterates over the first few data items. To do so, we added the following line 126 in `datasets/mados.py`. Also, the validation dataloder is set to be the same as the train dataloader (line 323 in `train.py`).
    ```
    self.tiles = self.tiles[:2]
    ```
- **Design Choices**: to make the comparison fairer we have implemented (so far) the two following solutions: 
    - So far, the multitemporal mechanism is a simple linear layer (L-TAE is suggested to be implemented)
    - We inserted a FLOPs/MACs computation. In fact, different models can have different sizes and hyperparameters, so comparing the performances without considering the needed computation would be a limit. For example, Prithvi pretrained model has 1/4 of GFLOPs w.r.t. SpectralGPT pretrained model (e.g. SpectralGPT uses a patch size of 8 w.r.t. Prithvi that uses 16). We can also consider adding inference times when we will develop the test.py
    
###  How to Contribute

#### New code
- **Datasets**: Add your dataset code within the `datasets` folder.
- **Foundation Models**: Integrate new foundation model code under the `models` folder. (done)
  - [X] SSL4EO-S12
  - [X] CROMA
  - [X] Scale-MAE
  - [X] SatlasNet
  - [X] Prithvi
  - [X] DOFA
  - [X] SpectralGPT
  - [X] RemoteCLIP
  - [X] GFM (msGFM's weights are not released)
- **Downstream Tasks**: Insert the code for downstream tasks (i.e. change detection) within the `tasks` folder. This may also necessitate modifications to `train.py` to accommodate new tasks. The tasks to be supported are i) multitemporal change detection and ii) pixel-level regression.
- **Add the Test**: Create a `test.py`, following a similar structure of `train.py`

#### Existing code

TODO: here are some aspects that should be improved:
- new tasks:
    - support multitemporality for change detection (should be easy, if following what we did for upernet)
    - support pixel level regression (should be easy, changing the loss when using upernet)

- improve the `adapt_input` function (in `train.py`), which is used to adapt the input shape of the data to be processed into the models **to do it through the config (e.g. pass the list of bands through the config)** 
    - At the moment, it supports just the mentioned models (and dataset) -> NEW MODELS TO BE ADDED
    - Moreover, for selecting the correct number of bands, just Sentinel-2 is supported -> TO SHAPE IT ALSO FOR OTHER MODALITIES
    - When a model needs more bands than the data have we are adding zero channels at the end of the available bands. We should change it to padding the missing channels. -> TO FIX THIS ERROR
    - When just RGB are used, we should reorder them -> TO FIX THIS ERROR

