import glob
import numpy as np
import os
import rasterio


split_file = os.path.join("data/Sen1Floods11", f"splits/flood_handlabeled/flood_train_data.csv")
data_root = os.path.join("data/Sen1Floods11", "data/flood_events/HandLabeled/")
with open(split_file) as f:
    file_list = f.readlines()

file_list = [f.rstrip().split(",") for f in file_list]

path = [os.path.join(data_root, 'S2Hand', f[0].replace('S1Hand', 'S2Hand')) for f in file_list]


#path = sorted(glob.glob('./data/Sen1Floods11/data/flood_events/HandLabeled/S2Hand/*tif'))

sum = np.zeros(13).astype(np.float64)
sum_sq = np.zeros(13).astype(np.float64)
data_list = []
for i, img in enumerate(path[:]):
    with rasterio.open(img) as src:
        data = src.read()

    data = data.reshape((13, -1))#.astype(np.float64)
    data_list.append(data)

    #sum = sum + np.sum(data, axis=1)
    #sum_sq = sum_sq + np.sum(data * data, axis=1)
    #print(i, sum, sum_sq)

data_list = np.concatenate(data_list, axis=1)
std = np.std(data_list, axis=1)
mean = np.mean(data_list, axis=1)

# n = (len(path) * 512 * 512)
# mean = sum / n
# tmp = sum_sq - sum * sum / n
# var = tmp / (n - 1)
# std = np.sqrt(var)

print(mean)
print(std)



