
### Progress
Currently supported foundation models:
- Prithvi
- SpectralGPT
- Scale-MAE

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

# Scale-MAE
wget https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth
```

### Pipeline -demo
To quickly get started, utilize [MADOS dataset](https://zenodo.org/records/10664073) to establish the complete pipeline for semantic segmentation:
```
python train.py configs/Prithvi_100M_config.yaml --path /your/datapath
```
#### Note:
- **Configurations**: The current configurations include parameters related to foundation model encoders and downstream task models. Future updates will aim to enhance configuration files to support additional tasks.
- **Logging**: By default, logs and checkpoints are stored in the `work_dir`.
- **The Mados dataset** in use is a simple example that only iterates over the first few data items. To do so, we added the following line 126 in `datasets/mados.py`. Also, the validation dataloder is set to be the same as the train dataloader (line 323 in `train.py`).
    ```
    self.tiles = self.tiles[:2]
    ```
- **Design Choices**: to make the comparison fairer we have implemented (so far) the two following solutions: 
    - For the UperNet, SpectralGPT uses a small linear projector to adjust the spectral dimension. We left this projection also when it's not strictly needed (e.g. Prithvi) to make the comparison uniform
    - We inserted a FLOPs/MACs computation. In fact, different models can have different sizes and hyperparameters, so comparing the performances without considering the needed computation would be a limit. For example, Prithvi pretrained model has 1/4 of GFLOPs w.r.t. SpectralGPT pretrained model (e.g. SpectralGPT uses a patch size of 8 w.r.t. Prithvi that uses 16). We can also consider adding inference times when we will develop the test.py
    
###  How to Contribute

#### New code
- **Datasets**: Add your dataset code within the `datasets` folder.
- **Foundation Models**: Integrate new foundation model code under the `models` folder.
- **Downstream Tasks**: Insert the code for downstream tasks (i.e. change detection, multi-temporal sem-seg) within the `tasks` folder. This may also necessitate modifications to `train.py` to accommodate new tasks.
- **Add the Test**: Create a `test.py`, following a similar structure to `train.py`

#### Existing code

TODO: here are some aspects that should be improved:
- config file: we should uniform the task parameters and the encoder parameters (some of them are redundant). Moreover, we should remove all the argparse from the training loop but the one about the paths and the training strategies (e.g. GPUs)
- add a strategy to combine multitemporal input data: some encoders should already support multitemporal data (e.g. Prithvi), for some others we should add a strategy to combine them (e.g. [L-TAE](https://github.com/VSainteuf/utae-paps/tree/main))
- improve the `adapt_input` function (in `train.py`), which is used to adapt the input shape of the data to be processed into the models. 
    - At the moment, it supports just the mentioned models (and dataset) -> NEW MODELS TO BE ADDED
    - Moreover, for selecting the correct number of bands, just Sentinel-2 is supported -> TO SHAPE IT ALSO FOR OTHER MODALITIES
    - When a model needs more bands than the data have we are adding zero channels at the end of the available bands. We should change it to padding the missing channels. -> TO FIX THIS ERROR
    - When just RGB are used, we should reorder them -> TO FIX THIS ERROR

