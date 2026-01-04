"""
Data download utilities
"""

import kagglehub


def download_dataset(dataset_name):
    """
    Download dataset from Kaggle using kagglehub
    
    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., "user/dataset-name")
    
    Returns:
        str: Path to downloaded dataset
    """
    print(f"Downloading dataset: {dataset_name}")
    path = kagglehub.dataset_download(dataset_name)
    print(f"✓ Dataset downloaded to: {path}")
    return path
