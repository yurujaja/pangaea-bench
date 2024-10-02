from tqdm import tqdm
import numpy as np
from torch.utils.data import Subset, DataLoader

# Function to calculate class distributions with a progress bar
def calculate_class_distributions(dataset, num_classes):
    class_distributions = []
    
    # Adding a progress bar for dataset processing
    for idx in tqdm(range(len(dataset)), desc="Calculating Class Distributions"):
        target = dataset[idx]['target']
        total_pixels = target.numel()
        class_counts = [(target == i).sum().item() for i in range(num_classes)]
        class_ratios = [count / total_pixels for count in class_counts]
        class_distributions.append(class_ratios)
    
    return np.array(class_distributions)

# Function to bin class distributions with a progress bar
def bin_class_distributions(class_distributions, num_bins=3, logger=None):

    logger.info("Binning class distributions...")
    # Adding a progress bar for binning class distributions
    binned_distributions = np.digitize(class_distributions, np.linspace(0, 1, num_bins+1)) - 1
    return binned_distributions

# Function to perform stratification and return only the indices
def stratify_single_dataset_indices(dataset, num_classes, label_fraction=1.0, num_bins=3, logger=None):
    
    logger.info("Starting stratification...")

    # Step 1: Calculate class distributions with progress tracking
    class_distributions = calculate_class_distributions(dataset, num_classes)

    # Step 2: Bin the class distributions
    binned_distributions = bin_class_distributions(class_distributions, num_bins=num_bins, logger=logger)

    # Step 3: Combine the bins to use for stratification
    combined_bins = np.apply_along_axis(lambda row: ''.join(map(str, row)), axis=1, arr=binned_distributions)

    # Step 4: Select a subset of labeled data with progress tracking
    num_labeled = int(len(dataset) * label_fraction)
    logger.info(f"Selecting {label_fraction * 100:.0f}% labeled data from {len(dataset)} samples...")

    # Shuffle and take the labeled part of the dataset based on the binned distributions
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    # Sort the indices based on combined bins to preserve class distribution
    sorted_indices = np.argsort(combined_bins)
    labeled_idx = sorted_indices[:num_labeled]
    unlabeled_idx = sorted_indices[num_labeled:]

    logger.info("Stratification complete.")
    return labeled_idx, unlabeled_idx
