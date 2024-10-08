import random
from tqdm import tqdm
import numpy as np


# Function to calculate class distributions for classification with a progress bar
def calculate_class_distributions(dataset, num_classes):
    class_distributions = []

    # Adding a progress bar for dataset processing
    for idx in tqdm(range(len(dataset)), desc="Calculating class distributions per sample"):
        target = dataset[idx]['target']
        total_pixels = target.numel()
        class_counts = [(target == i).sum().item() for i in range(num_classes)]
        class_ratios = [count / total_pixels for count in class_counts]
        class_distributions.append(class_ratios)

    return np.array(class_distributions)


# Function to calculate distribution metrics for regression
def calculate_regression_distributions(dataset):
    distributions = []

    # Adding a progress bar for dataset processing
    for idx in tqdm(range(len(dataset)), desc="Calculating regression distributions per sample"):
        target = dataset[idx]['target']
        mean_value = target.mean().item()  # Example for mean; adjust as needed for other metrics
        distributions.append(mean_value)

    return np.array(distributions)


# Function to bin class distributions with a progress bar
def bin_class_distributions(class_distributions, num_bins=3, logger=None):
    logger.info(f"Class distributions are being binned into {num_bins} categories")
    # Adding a progress bar for binning class distributions
    binned_distributions = np.digitize(class_distributions, np.linspace(0, 1, num_bins+1)) - 1
    return binned_distributions


# Function to bin regression distributions with a progress bar
def bin_regression_distributions(regression_distributions, num_bins=3, logger=None):
    logger.info(f"Regression distributions are being binned into {num_bins} categories")
    # Define the range for binning based on minimum and maximum values in regression distributions
    binned_distributions = np.digitize(
        regression_distributions, 
        np.linspace(regression_distributions.min(), regression_distributions.max(), num_bins + 1)
    ) - 1
    return binned_distributions


# Function to perform stratification for classification and return only the indices
def stratify_classification_dataset_indices(dataset, num_classes, label_fraction=1.0, num_bins=3, logger=None):
    # Step 1: Calculate class distributions with progress tracking
    class_distributions = calculate_class_distributions(dataset, num_classes)

    # Step 2: Bin the class distributions
    binned_distributions = bin_class_distributions(class_distributions, num_bins=num_bins, logger=logger)
    
    # Step 3: Combine the bins to use for stratification
    combined_bins = np.apply_along_axis(lambda row: ''.join(map(str, row)), axis=1, arr=binned_distributions)

    # Step 4: Select a subset of labeled data with progress tracking
    num_labeled = int(len(dataset) * label_fraction)

    # Sort the indices based on combined bins to preserve class distribution
    sorted_indices = np.argsort(combined_bins)
    labeled_idx = sorted_indices[:num_labeled]
    unlabeled_idx = sorted_indices[num_labeled:]

    return labeled_idx, unlabeled_idx


# Function to perform stratification for regression and return only the indices
def stratify_regression_dataset_indices(dataset, label_fraction=1.0, num_bins=3, logger=None):
    # Step 1: Calculate regression distributions with progress tracking
    regression_distributions = calculate_regression_distributions(dataset)

    # Step 2: Bin the regression distributions
    binned_distributions = bin_regression_distributions(regression_distributions, num_bins=num_bins, logger=logger)
    
    # Step 3: Sort the indices based on binned distributions for stratification
    sorted_indices = np.argsort(binned_distributions)
    
    # Step 4: Select a subset of labeled data with progress tracking
    num_labeled = int(len(dataset) * label_fraction)
    labeled_idx = sorted_indices[:num_labeled]
    unlabeled_idx = sorted_indices[num_labeled:]

    return labeled_idx, unlabeled_idx


# Function to get subset indices based on the strategy, supporting both classification and regression
def get_subset_indices(dataset, strategy="random", label_fraction=0.5, num_bins=3, logger=None):
    logger.info(
        f"Creating a subset of the {dataset.split} dataset using {strategy} strategy, with {label_fraction * 100}% of labels utilized."
    )
    if strategy == "stratified_classification":
        indices, _ = stratify_classification_dataset_indices(
            dataset, num_classes=dataset.num_classes, label_fraction=label_fraction, num_bins=num_bins, logger=logger
        )
    elif strategy == "stratified_regression":
        indices, _ = stratify_regression_dataset_indices(
            dataset, label_fraction=label_fraction, num_bins=num_bins, logger=logger
        )
    else:  # Default to random sampling
        n_samples = len(dataset)
        indices = random.sample(
            range(n_samples), int(n_samples * label_fraction)
        )
    
    return indices
