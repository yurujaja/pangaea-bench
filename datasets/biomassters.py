import numpy as np
import torch
import pandas as pd
import pathlib
import rasterio
from tifffile import imread
from os.path import join as opj
from .utils import read_tif
from utils.registry import DATASET_REGISTRY

def read_imgs(multi_temporal, temp , fname, data_dir, img_size):
    imgs_s1, imgs_s2, mask = [], [], []
    if multi_temporal:
        month_list = list(range(12))
    else:
        month_list = [temp]
    
    for month in month_list:
        
        s1_fname = '%s_%s_%02d.tif' % (str.split(fname, '_')[0], 'S1', month)
        s2_fname = '%s_%s_%02d.tif' % (str.split(fname, '_')[0], 'S2', month)

        s1_filepath = data_dir.joinpath(s1_fname)
        if s1_filepath.exists():
            img_s1 = imread(s1_filepath)
            m = img_s1 == -9999
            img_s1 = img_s1.astype('float32')
            img_s1 = np.where(m, 0, img_s1)
        else:            
            img_s1 = np.zeros((img_size, img_size) + (4,), dtype='float32')
        
        s2_filepath = data_dir.joinpath(s2_fname)
        if s2_filepath.exists():
            img_s2 = imread(s2_filepath)
            img_s2 = img_s2.astype('float32')
        else:            
            img_s2 = np.zeros((img_size, img_size) + (11,), dtype='float32')
        
        img_s1 = np.transpose(img_s1, (2, 0, 1))
        img_s2 = np.transpose(img_s2, (2, 0, 1))
        imgs_s1.append(img_s1)
        imgs_s2.append(img_s2)
        mask.append(False)

    mask = np.array(mask)

    imgs_s1 = np.stack(imgs_s1, axis=1)
    imgs_s2 = np.stack(imgs_s2, axis=1)

    return imgs_s1, imgs_s2, mask

@DATASET_REGISTRY.register()
class BioMassters(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        
        self.root_path = cfg['root_path']
        self.data_min = cfg['data_min']
        self.data_max = cfg['data_max']
        self.multi_temporal = cfg['multi_temporal']
        self.temp = cfg['temporal']
        self.split = split
        self.img_size = cfg['img_size']
        
        self.data_path = pathlib.Path(self.root_path).joinpath(f"{split}_Data_list.csv")
        self.id_list = pd.read_csv(self.data_path)['chip_id']
        
        self.split_path = 'train' if split == 'val' else split
        self.dir_features = pathlib.Path(self.root_path).joinpath(f'{self.split_path}_features')
        self.dir_labels = pathlib.Path(self.root_path).joinpath( f'{self.split_path}_agbm')

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):

        chip_id = self.id_list.iloc[index]
        fname = str(chip_id)+'_agbm.tif'
        
        imgs_s1, imgs_s2, mask = read_imgs(self.multi_temporal, self.temp, fname, self.dir_features, self.img_size)
        with rasterio.open(self.dir_labels.joinpath(fname)) as lbl:
            target = lbl.read(1)
        target = np.nan_to_num(target)

        imgs_s1 = torch.from_numpy(imgs_s1).float()
        imgs_s2 = torch.from_numpy(imgs_s2).float()
        target = torch.from_numpy(target).float()

        return {
            'image': {
                    'optical': imgs_s2,
                    'sar' : imgs_s1,
                    },
            'target': target,  
            'metadata': {'masks':mask}
        }

    @staticmethod
    def get_splits(dataset_config):
        dataset_train = BioMassters(cfg=dataset_config, split='train')
        dataset_val = BioMassters(cfg=dataset_config, split='val')
        dataset_test = BioMassters(cfg=dataset_config, split='test')
        return dataset_train, dataset_val, dataset_test
    
    @staticmethod
    def download(dataset_config:dict, silent=False):
        pass
