
### Progress
Currently supported foundation models:
- Prithvi
- SpectralGPT

Currently supported tasks:
- Upernet for semantic segmentation

### Setup
Clone the repository:
```
git clone https://github.com/yurujaja/geofm-bench.git
cd geofm-bench
```
Dependencies:
```
conda create -n <env_name> python=3.8.12  # change <env_name> 
conda activate <env_name> 
conda install -c conda-forge gdal==3.3.2 

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

conda install pytables==3.7.0
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
```

### Pipeline -demo
To quickly get started, utilize [MADOS dataset](https://zenodo.org/records/10664073) to establish the complete pipeline for semantic segmentation:
```
python train.py configs/Prithvi_100M_config.yaml --path /your/datapath
```
#### Note:
- **Configurations**: The current configurations include parameters related to foundation model encoders and downstream task models. Future updates will aim to enhance configuration files to support additional tasks.
- **Logging**: By default, logs and checkpoints are stored in the `work_dir`.
- **The Mados dataset** in use is a simple example that only iterates over the first few data items. To do so, we added the following line 126 in `datasets/mados.py`. Also, the validation dataloder is set to be the same as train dataloader (line 323 in `train.py`).
    ```
    self.tiles = self.tiles[:2]
    ```
    
###  How to Contribute

#### New code
- **Datasets**: Add your dataset code within the `datasets` folder.
- **Foundation Models**: Integrate new foundation model code under the `models` folder.
- **Downstream Tasks**: Insert the code for downstream tasks within the `tasks` folder. This may also necessitate modifications to `training.py` to accommodate new tasks.

#### Existing code

TODO: here are some aspects that should be improved:
- config file: we should uniform the task parameters and the encoder parameters (some of them are redundant). Moreover, we should remove all the argparse from the training loop but the one about the paths and the training strategies (e.g. GPUs)
- add a strategy to combine multitemporal input data: some encoders should already support multitemporal data (e.g. Prithvi), for some others we should add a strategy to combine them (e.g. U-TAE)
- improve the `adapt_input` function (in `train.py`), which is used to adapt the input shape of the data to be processed into the models. At the moments, it supports just the mentioned models. Moreover, for selecting the correct number of bands, just Sentinel-2 is supported. When a model needs more bands than the data have we are zero padding the missing channels.

