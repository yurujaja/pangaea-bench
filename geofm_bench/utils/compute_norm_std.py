import glob
import numpy as np
import os
import rasterio


def compute_norm_std(split_file_path, data_root_path):
    """
    Computes the mean, standard deviation, maximum, and minimum values for a set of raster images.

    Parameters:
    split_file_path (str): Path to the CSV file containing the list of image files.
    data_root_path (str): Root directory where the image files are stored.

    Returns:
    tuple: A tuple containing the mean, standard deviation, maximum, and minimum values.
    """

    with open(split_file_path) as f:
        file_list = f.readlines()

    file_list = [f.rstrip().split(",") for f in file_list]

    # Construct the full paths to the image files
    path = [os.path.join(data_root_path, 'S1Hand', f[0]) for f in file_list]

    data_list = []
    for img in path:
        with rasterio.open(img) as src:
            data = src.read()
            data = np.nan_to_num(data)

        data = data.reshape((2, -1))
        data_list.append(data)

    data_list = np.concatenate(data_list, axis=1)
    std = np.std(data_list, axis=1)
    mean = np.mean(data_list, axis=1)
    max_val = np.max(data_list, axis=1)
    min_val = np.min(data_list, axis=1)

    return mean, std, max_val, min_val

# Example usage
split_file = os.path.join("data/sen1floods11_v1.1/v1.1", f"splits/flood_handlabeled/flood_train_data.csv")
data_root = os.path.join("data/sen1floods11_v1.1/v1.1", "data/flood_events/HandLabeled/")
mean, std, max_val, min_val = compute_norm_std(split_file, data_root)

print("Max values:", max_val)
print("Min values:", min_val)



