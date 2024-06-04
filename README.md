
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
conda create -n mados python=3.8.12
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

# SpectralGPT
wget
```

### Pipeline -demo
To quickly get started, utilize the simple MADOS dataset to establish the complete pipeline for semantic segmentation:
```
python train.py configs/Prithvi_100M_config.yaml
```
#### Note:
- **Configurations**: The current configurations include parameters related to foundation model encoders and downstream task models. Future updates will aim to enhance configuration files to support additional tasks.
- **Logging**: By default, logs and checkpoints are stored in the `work_dir`.
- **The Mados dataset** in use is a simple example that only iterates over the first few data items.
###  How to Contribute

- **Datasets**: Add your dataset code within the `datasets` folder.
- **Foundation Models**: Integrate new foundation model code under the `models` folder.
- **Downstream Tasks**: Insert the code for downstream tasks within the `tasks` folder. This may also necessitate modifications to `training.py` to accommodate new tasks.
