# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/gkakogeorgiou/mados
'''

import os
import time
import pathlib
import urllib.request
import urllib.error
import zipfile
import tqdm
from glob import glob

import torch
import torch.utils.data
import random
import rasterio
import numpy as np
import gdal

# Pixel-Level class distribution (total sum equals 1.0)
class_distr = torch.Tensor([0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452, 
 0.34775, 0.20638, 0.00062, 0.1169, 0.09188, 0.01309, 0.00917, 0.00176, 0.00963])

bands_mean = np.array([0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902, 0.03680401,
 0.03999107, 0.03566642, 0.03965081, 0.0267993,  0.01978944]).astype('float32')

bands_std = np.array([0.03240627, 0.03432253, 0.0354812,  0.0375769,  0.03785412, 0.04992323,
 0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115]).astype('float32')

mados_cat_mapping =  {'Marine Debris': 1,
					  'Dense Sargassum': 2,
					  'Sparse Floating Algae': 3,
					  'Natural Organic Material': 4,
					  'Ship': 5,
					  'Oil Spill': 6,
					  'Marine Water': 7,
					  'Sediment-Laden Water': 8,
					  'Foam': 9,
					  'Turbid Water': 10,
					  'Shallow Water': 11,
					  'Waves & Wakes': 12,
					  'Oil Platform': 13,
					  'Jellyfish': 14,
					  'Sea snot': 15}
				   
mados_color_mapping =  { 'Marine Debris': 'red',
						 'Dense Sargassum': 'green',
						 'Sparse Floating Algae': 'limegreen',
						 'Natural Organic Material': 'brown',
						 'Ship': 'orange',
						 'Oil Spill': 'thistle',
						 'Marine Water': 'navy',
						 'Sediment-Laden Water': 'gold',
						 'Foam': 'purple',
						 'Turbid Water': 'darkkhaki',
						 'Shallow Water': 'darkturquoise',
						 'Waves & Wakes': 'bisque',
						 'Oil Platform': 'dimgrey',
						 'Jellyfish': 'hotpink',
						 'Sea snot': 'yellow'}

labels = ['Marine Debris', 'Dense Sargassum', 'Sparse Floating Algae', 'Natural Organic Material', 
'Ship', 'Oil Spill', 'Marine Water', 'Sediment-Laden Water', 'Foam', 
'Turbid Water', 'Shallow Water', 'Waves & Wakes', 'Oil Platform', 'Jellyfish', 'Sea snot']

s2_mapping = {'nm440': 0,
              'nm490': 1,
              'nm560': 2,
              'nm665': 3,
              'nm705': 4,
              'nm740': 5,
              'nm783': 6,
              'nm842': 7,
              'nm865': 8,
              'nm1600': 9,
              'nm2200': 10,
              'Class': 11,
              'Confidence': 12,
              'Report': 13}

conf_mapping = {'High': 1,
                'Moderate': 2,
                'Low': 3}

report_mapping = {'Very close': 1,
                  'Away': 2,
                  'No': 3}

def cat_map(x):
    return mados_cat_mapping[x]

cat_mapping_vec = np.vectorize(cat_map)

# Utility progress bar handler for urlretrieve
# Should go in a utils folder if we use it elsewhere
class DownloadProgressBar:
    def __init__(self):
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm.tqdm(desc="Downloading...", total=total_size, unit="b", unit_scale=True, unit_divisor=1024)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None

###############################################################
# MADOS DATASET                                               #
###############################################################
def get_band(path):
    return int(path.split('_')[-2])

class MADOS(torch.utils.data.Dataset):
    def __init__(self, path, splits=None, mode = 'train'):

        self.download_dataset(pathlib.Path(path), silent=True)

        #Default splits dir
        if not splits:
            splits = pathlib.Path(path) / "splits"
        
        if mode=='train':
            self.ROIs_split = np.genfromtxt(os.path.join(splits, 'train_X.txt'),dtype='str')
                
        elif mode=='test':
            self.ROIs_split = np.genfromtxt(os.path.join(splits, 'test_X.txt'),dtype='str')
                
        elif mode=='val':
            self.ROIs_split = np.genfromtxt(os.path.join(splits, 'val_X.txt'),dtype='str')
            
        else:
            raise
        self.X = []           # Loaded Images
        self.y = []           # Loaded Output masks
            
        self.tiles = glob(os.path.join(path,'*'))
        self.tiles = self.tiles[:2]

        for tile in tqdm(self.tiles, desc = 'Load '+mode+' set to memory'):

                # Get the number of different crops for the specific tile
                splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]
                
                for crop in splits:
                    crop_name = os.path.basename(tile)+'_'+crop.split('.tif')[0]
                    
                    if crop_name in self.ROIs_split:
    
                        # Load Input Images
                        # Get the bands for the specific crop 
                        all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_'+crop))
                        all_bands = sorted(all_bands, key=get_band)
            
                        ################################
                        # Upsample the bands #
                        ################################
                        current_image = []
                        for c, band in enumerate(all_bands, 1):
                            upscale_factor = int(os.path.basename(os.path.dirname(band)))//10
            
                            with rasterio.open(band, mode ='r') as src:
                                current_image.append(src.read(1,
                                                                out_shape=(
                                                                    int(src.height * upscale_factor),
                                                                    int(src.width * upscale_factor)
                                                                ),
                                                                resampling=rasterio.enums.Resampling.nearest
                                                              ).copy()
                                                  )
                        
                        stacked_image = np.stack(current_image)
                        self.X.append(stacked_image)
                        

                        # Load Classsification Mask
                        cl_path = os.path.join(tile,'10',os.path.basename(tile)+'_L2R_cl_'+crop)
                        ds = gdal.Open(cl_path)
                        temp = np.copy(ds.ReadAsArray().astype(np.int64))
                        
                        ds=None                   # Close file
                        self.y.append(temp)
            
        # print(self.X)
        self.X = np.stack(self.X)
        self.y = np.stack(self.y)
        
        # Categories from 1 to 0
        self.y = self.y - 1

        self.impute_nan = np.tile(bands_mean, (self.X.shape[-1],self.X.shape[-2],1))
        self.mode = mode
        self.length = len(self.y)
        self.path = path

    def __len__(self):
        return self.length
    
    def getnames(self):
        return self.ROIs_split
    
    def __getitem__(self, index):
        image = self.X[index]
        target = self.y[index]

        image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype('float32')       # CxWxH to WxHxC
        
        nan_mask = np.isnan(image)
        image[nan_mask] = self.impute_nan[nan_mask]
        
        # print(target.shape)
        target = target[:,:,np.newaxis]
        # print(target.shape)

        image = ((image.astype(np.float32).transpose(2, 0, 1).copy() - bands_mean.reshape(-1,1,1))/ bands_std.reshape(-1,1,1)).squeeze()
        target = target.squeeze()

        # print("dataloader", image.shape)
        
        return image.copy(), target.copy()

    @staticmethod
    def download_dataset(output_path:pathlib.Path, silent=False, mados_url="https://zenodo.org/records/10664073/files/MADOS.zip?download=1"):
        try:
            os.makedirs(output_path, exist_ok=False)
        except FileExistsError:
            if not silent:
                print("MADOS Dataset folder exists, skipping downloading dataset.")
            return

        temp_file_name = f"temp_{hex(int(time.time()))}_MADOS.zip"
        pbar = DownloadProgressBar()

        try:
            urllib.request.urlretrieve(mados_url, output_path / temp_file_name, pbar)
        except urllib.error.HTTPError as e:
            print('Error while downloading dataset: The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
            return
        except urllib.error.URLError as e:
            print('Error while downloading dataset: Failed to reach a server.')
            print('Reason: ', e.reason)
            return

        with zipfile.ZipFile(output_path / temp_file_name, 'r') as zip_ref:
            print(f"Extracting to {output_path} ...")
            # Remove top-level dir in ZIP file for nicer data dir structure
            members = []
            for zipinfo in zip_ref.infolist():
                new_path = os.path.join(*(zipinfo.filename.split(os.path.sep)[1:]))
                zipinfo.filename = str(new_path)
                members.append(zipinfo)

            zip_ref.extractall(output_path, members)
            print("done.")

        os.remove(output_path / temp_file_name)

###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)

if __name__ == "__main__":
    mados_path = "data/mados"
    ds = MADOS(mados_path)
    print(next(iter(ds)))