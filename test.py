from datasets import CropTypeMappingSouthSudan
from omegaconf import OmegaConf
import pdb
from collections import Counter
import numpy as np
import os
import tarfile
import shutil 

ds = CropTypeMappingSouthSudan
cfg = OmegaConf.load('configs/datasets/croptypemapping.yaml')

ds_train, ds_val, ds_test = ds.get_splits(dataset_config=cfg)

s2_sum = []
s1_sum = []
for i in range(len(ds_train)):
    data = ds_train[i]
    s2_sum.append(data['image']['optical'])
    s1_sum.append(data['image']['sar'])
    # print(data['image']['sar'].shape)
    # print(data['target'].shape)
    # print(data['metadata']['s2'].shape)
    # print(data['metadata']['s1'].shape)
    # print(i)
    # print(data['image']['optical'].shape)
    # print(data['image']['sar'].shape)
    # print(data['target'].shape)
    # print(data['target'].unique())
    # print(data['metadata']['s2'].shape)
    # print(data['metadata']['s1'].shape)
s2_sum = np.concatenate(s2_sum, axis=1)
s1_sum = np.concatenate(s1_sum, axis=1)
print(s2_sum.shape)
print(s1_sum.shape)

s2_mean = np.mean(s2_sum, axis=(1,2,3))
s2_std = np.std(s2_sum, axis=(1,2,3))
s1_mean = np.mean(s1_sum, axis=(1,2,3))
s1_std = np.std(s1_sum, axis=(1,2,3))
print(s2_mean)
print(s2_std)
print(s1_mean)
print(s1_std)
# target_sum = np.concatenate(target_sum)

# # Get unique values and their counts
# unique_values, counts = np.unique(target_sum, return_counts=True)

# # Print the counts
# sum = 0
# for value, count in zip(unique_values, counts):
#     if value == 0:
#         continue
#     sum += count
#     print(f"Value {value}: {count} times")

# for value, count in zip(unique_values, counts):
#     if value == 0:
#         continue
#     print(f"Value {value}: {count/sum*100:.2f}%")