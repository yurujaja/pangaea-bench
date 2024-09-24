import glob
import numpy as np
import os
import tifffile


split_file = os.path.join("data/sen1floods11_v1.1/v1.1", f"splits/flood_handlabeled/flood_train_data.csv")
data_root = os.path.join("data/sen1floods11_v1.1/v1.1", "data/flood_events/HandLabeled/")
with open(split_file) as f:
    file_list = f.readlines()

file_list = [f.rstrip().split(",") for f in file_list]

# path = [os.path.join(data_root, 'S2Hand', f[0].replace('S1Hand', 'S2Hand')) for f in file_list]
path = [os.path.join(data_root, 'S1Hand', f[0]) for f in file_list]

#path = sorted(glob.glob('./data/Sen1Floods11/data/flood_events/HandLabeled/S2Hand/*tif'))

sum = np.zeros(2).astype(np.float64)
sum_sq = np.zeros(2).astype(np.float64)
data_list = []
for i, img in enumerate(path[:]):
    data = tifffile.imread(img)
    data = np.nan_to_num(data)

    data = data.reshape((2, -1))#.astype(np.float64)
    data_list.append(data)

    #sum = sum + np.sum(data, axis=1)
    #sum_sq = sum_sq + np.sum(data * data, axis=1)
    #print(i, sum, sum_sq)

data_list = np.concatenate(data_list, axis=1)
std = np.std(data_list, axis=1)
mean = np.mean(data_list, axis=1)

max = np.max(data_list, axis=1)
min = np.min(data_list, axis=1)

# n = (len(path) * 512 * 512)
# mean = sum / n
# tmp = sum_sq - sum * sum / n
# var = tmp / (n - 1)
# std = np.sqrt(var)

print(max)
print(min)



