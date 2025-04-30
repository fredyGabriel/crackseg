"""Provides functionality for splitting datasets into train/val/test sets."""

# import os
import random
import warnings
from typing import List, Tuple, Dict, Optional
from pathlib import Path
# import numpy as np
import math
from torch.utils.data import Dataset

# Placeholder for now, might need CrackSegmentationDataset


def split_indices(
    num_samples: int,
    ratios: Dict[str, float],
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Dict[str, List[int]]:
    """Splits a range of indices into train, validation, and test sets.

    Args:
        num_samples (int): The total number of samples.
        ratios (Dict[str, float]): Split ratios (e.g., {'train': 0.7,
                                                        'val': 0.15}).
            Keys must be 'train', 'val', 'test'. Values should sum to 1.0.
            'test' ratio is inferred if not provided.
        seed (Optional[int]): Random seed for shuffling. Defaults to None.
        shuffle (bool): Whether to shuffle indices. Defaults to True.

    Returns:
        Dict[str, List[int]]: Map from split name to list of indices.

    Raises:
        ValueError: If ratios are invalid.
    """
    if not all(k in ['train', 'val', 'test'] for k in ratios):
        raise ValueError("Ratio keys must be 'train', 'val', or 'test'.")
    if not all(0.0 <= v <= 1.0 for v in ratios.values()):
        raise ValueError("Ratio values must be between 0.0 and 1.0.")

    internal_ratios = ratios.copy()
    provided_ratio_sum = sum(internal_ratios.values())

    if 'test' not in internal_ratios:
        if provided_ratio_sum > 1.0:
            raise ValueError("Sum of provided ratios cannot exceed 1.0")
        internal_ratios['test'] = 1.0 - provided_ratio_sum
    elif not math.isclose(sum(internal_ratios.values()), 1.0):
        raise ValueError("Sum of ratios must be close to 1.0")

    indices = list(range(num_samples))
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)

    split_indices_dict: Dict[str, List[int]] = {'train': [], 'val': [],
                                                'test': []}
    current_idx = 0

    # Calculate counts using floor to avoid exceeding total samples initially
    train_count = math.floor(num_samples * internal_ratios['train'])
    val_count = math.floor(num_samples * internal_ratios['val'])
    # Calculate test count accurately based on the remainder
    test_count = num_samples - train_count - val_count

    if test_count < 0:
        # This case should ideally not happen with floor, but as safety:
        test_count = 0
        val_count = num_samples - train_count
        if val_count < 0:
            val_count = 0
            train_count = num_samples  # All to train

    split_indices_dict['train'] = indices[current_idx:
                                          current_idx + train_count]
    current_idx += train_count
    split_indices_dict['val'] = indices[current_idx:
                                        current_idx + val_count]
    current_idx += val_count
    split_indices_dict['test'] = indices[current_idx:
                                         current_idx + test_count]

    # Final check
    total_assigned = sum(len(v) for v in split_indices_dict.values())
    if total_assigned != num_samples:
        warnings.warn(
            f"Final counts sum to {total_assigned}, expected {num_samples}. "
            f"Check ratios/sample size."
        )

    return split_indices_dict


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
    data_root_path = Path(data_root)
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
    image_size: Tuple[int, int],
    ratios: Dict[str, float],
    seed: Optional[int] = None,
    cache_flag: bool = False,
    dataset_cls: type = None  # Pass the actual Dataset class type
) -> Dict[str, Dataset]:  # Return type updated
    """Creates split datasets (train, val, test).

    Finds all samples in data_root, splits them according to ratios,
    and instantiates the provided dataset_cls for each split.

    Args:
        data_root (str): Root directory containing 'images' and 'masks'.
        image_size (Tuple[int, int]): Target size for resizing.
        ratios (Dict[str, float]): Split ratios (e.g., {'train': 0.7}).
        seed (Optional[int]): Random seed for splitting and dataset init.
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

    all_samples = get_all_samples(data_root)
    if not all_samples:
        # get_all_samples raises FileNotFoundError if dirs are missing
        # It warns if no pairs are found, but we raise here if list is empty
        raise RuntimeError(f"No valid image/mask pairs found in {data_root}.")

    num_samples = len(all_samples)
    indices_map = split_indices(num_samples, ratios, seed, shuffle=True)

    datasets: Dict[str, Dataset] = {}
    for split_name, indices in indices_map.items():
        split_samples = [all_samples[i] for i in indices]

        if not split_samples and ratios.get(split_name, 0) > 0:
            # Only warn if the ratio was non-zero but we got 0 samples
            # (e.g., very few total samples and a small ratio)
            warnings.warn(
                f"Split '{split_name}' resulted in 0 samples despite ratio "
                f"{ratios.get(split_name)}. Check total samples vs ratios."
            )
            # Still create an empty dataset if the class supports it

        try:
            # Instantiate the provided dataset class
            datasets[split_name] = dataset_cls(
                mode=split_name,
                image_size=image_size,
                samples_list=split_samples,  # Pass the filtered list
                seed=seed,                   # Pass seed for reproducibility
                in_memory_cache=cache_flag
            )
            # Log dataset size
            print(f"Created dataset for '{split_name}' with \
{len(datasets[split_name])} samples.")
        except Exception as e:
            # Log the error and potentially stop, or just warn and continue?
            # For now, re-raise as it indicates a fundamental problem.
            raise RuntimeError(
                f"Failed to instantiate dataset for split '{split_name}': {e}"
            ) from e

    return datasets
