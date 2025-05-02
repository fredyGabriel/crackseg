"""Provides functionality for splitting datasets into train/val/test sets."""

# import os
# import random
import warnings
import math
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path
# import numpy as np
# import math
from torch.utils.data import Dataset
from omegaconf import DictConfig


def split_indices(
    num_samples: int,
    ratios: Dict[str, float],
    seed: Optional[int] = None,
    shuffle: bool = True
) -> Dict[str, List[int]]:
    """Split indices into train/val/test sets based on provided ratios.

    Args:
        num_samples (int): Total number of samples to split.
        ratios (Dict[str, float]): Dictionary with keys for split names and
            values as ratios. Valid keys are 'train', 'val', 'test'.
            If 'test' is missing, it will be inferred as 1.0 minus the sum of
            other ratios.
        seed (Optional[int]): Random seed for reproducibility. If None, no seed
            is set and results may vary between runs.
        shuffle (bool): Whether to shuffle indices before splitting.

    Returns:
        Dict[str, List[int]]: Dictionary with keys for split names and values
        as lists of indices.

    Raises:
        ValueError: If ratios have invalid keys, values, or sum to > 1.0.
    """
    # Validate ratio keys
    valid_keys = {'train', 'val', 'test'}
    if not all(k in valid_keys for k in ratios.keys()):
        raise ValueError(
            f"Ratio keys must be one of {valid_keys}, got {ratios.keys()}"
        )

    # Validate ratio values
    if not all(0 <= v <= 1.0 for v in ratios.values()):
        raise ValueError(
            "Ratio values must be between 0 and 1.0"
        )

    # Handle case where test is not provided
    if 'test' not in ratios:
        sum_provided = sum(ratios.values())
        if sum_provided > 1.0:
            raise ValueError(
                "Sum of provided ratios cannot exceed 1.0 when inferring "
                "test ratio"
            )
        ratios = ratios.copy()  # Avoid modifying the input
        ratios['test'] = 1.0 - sum_provided

    # Validate sum of ratios
    ratio_sum = sum(ratios.values())
    if not (0.99 <= ratio_sum <= 1.01):  # Allow small floating point error
        raise ValueError(
            f"Sum of ratios must be close to 1.0, got {ratio_sum}"
        )

    # Generate indices
    indices = list(range(num_samples))

    # Shuffle if requested
    if shuffle and seed is not None:
        random.seed(seed)
    if shuffle:
        random.shuffle(indices)

    # Calculate split sizes
    train_size = math.floor(num_samples * ratios['train'])
    val_size = math.floor(num_samples * ratios['val'])
    # Ensure no rounding issues

    # Split the indices
    result = {
        'train': indices[:train_size],
        'val': indices[train_size:train_size + val_size],
        'test': indices[train_size + val_size:]
    }

    return result


# --- Updated function ---

def get_all_samples(data_root: str) -> List[Tuple[str, str]]:
    """Scans data_root to find all image/mask pairs.

    Assumes a structure like:
        data_root/
        ├── images/
        │   ├── img1.png
        │   ├── img2.jpg
        │   └── ...
        └── masks/
            ├── img1.png
            ├── img2.png  (Masks must be .png)
            └── ...

    Args:
        data_root (str): The root directory containing 'images' and 'masks'
        folders.

    Returns:
        List[Tuple[str, str]]: A list of (image_path, mask_path) tuples.

    Raises:
        FileNotFoundError: If 'images' or 'masks' directory doesn't exist.
    """
    # Ensure data_root is an absolute path
    data_root_path = Path(data_root).resolve()
    images_dir = data_root_path / "images"
    masks_dir = data_root_path / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    all_samples = []
    image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    # Iterate through image files
    image_files = sorted([p for p in images_dir.glob("*")
                          if p.suffix.lower() in image_extensions])

    for img_path in image_files:
        # Expect mask to have the same stem but .png extension
        mask_path = masks_dir / f"{img_path.stem}.png"

        if mask_path.is_file():
            all_samples.append((str(img_path), str(mask_path)))
        else:
            warnings.warn(
                f"Mask not found for image: {img_path.name} "
                f"(expected at {mask_path})"
            )

    if not all_samples:
        warnings.warn(f"No matching image/mask pairs found in {data_root}")

    return all_samples


# --- Final Implementation ---

def create_split_datasets(
    data_root: str,
    transform_cfg: DictConfig,
    seed: Optional[int] = None,
    cache_flag: bool = False,
    dataset_cls: type = None  # Pass the actual Dataset class type
) -> Dict[str, Dataset]:  # Return type updated
    """Creates split datasets (train, val, test) from existing folders.

    Finds samples within data_root/train, data_root/val, data_root/test
    and instantiates the provided dataset_cls for each split.

    Args:
        data_root (str): Root directory containing 'train', 'val', 'test'
                         subdirectories, each with 'images' and 'masks'.
        transform_cfg (DictConfig): Configuration for dataset transformations,
                                expected to have keys 'train', 'val', 'test'.
        seed (Optional[int]): Random seed for dataset init.
        cache_flag (bool): Whether to enable in-memory caching in datasets.
        dataset_cls (type): The Dataset class to instantiate (e.g.,
                            CrackSegmentationDataset).

    Returns:
        Dict[str, Dataset]: Dictionary mapping split name ('train', 'val',
                           'test') to the instantiated Dataset object.

    Raises:
        ValueError: If dataset_cls is not provided.
        FileNotFoundError: If data_root subdirectories are missing.
        RuntimeError: If no samples are found in data_root.
    """
    if dataset_cls is None:
        raise ValueError("dataset_cls must be provided.")

    datasets: Dict[str, Dataset] = {}
    # Iterate through expected split names
    for split_name in ['train', 'val', 'test']:
        # Construct path to the specific split directory
        split_data_root = Path(data_root) / split_name

        # Get samples specifically for this split
        try:
            split_samples = get_all_samples(str(split_data_root))
        except FileNotFoundError:
            warnings.warn(
                f"Directory or subdirs not found for split '{split_name}' "
                f"at {split_data_root}. Skipping split."
            )
            continue  # Skip this split if dirs not found

        if not split_samples:
            warnings.warn(
                f"No samples found for split '{split_name}' "
                f"at {split_data_root}. Creating empty dataset if possible."
            )
            # Continue to create dataset, it might handle empty list

        # Get transform config for this specific split
        if split_name not in transform_cfg:
            raise ValueError(
                f"Transform config missing for split: {split_name}"
            )
        split_transform_config = transform_cfg[split_name]

        try:
            # Instantiate the provided dataset class for this split
            datasets[split_name] = dataset_cls(
                mode=split_name,
                samples_list=split_samples,  # Pass the split-specific list
                seed=seed,
                in_memory_cache=cache_flag,
                config_transform=split_transform_config
            )
            print(
                f"Created dataset for '{split_name}' "
                f"with {len(datasets[split_name])} samples."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate dataset for split '{split_name}': {e}"
            ) from e

    # Check if we actually created datasets (especially train/val needed)
    if 'train' not in datasets or 'val' not in datasets:
        warnings.warn(
            "Train or Val dataset could not be created. "
            "Check data paths and structure."
        )

    return datasets
