"""
Data Statistics Utilities
This module provides functions for calculating and retrieving statistics
about the project's datasets.
"""

from pathlib import Path


def get_dataset_image_counts(data_root: Path) -> dict[str, int]:
    """
    Counts the number of images in train, val, and test directories.

    Args:
        data_root: The root directory of the 'data' folder.

    Returns:
        A dictionary with counts for 'train', 'val', and 'test' sets.
    """
    counts = {}
    for split in ["train", "val", "test"]:
        image_dir = data_root / split / "images"
        if image_dir.is_dir():
            # Count common image file types
            image_files = (
                list(image_dir.glob("*.jpg"))
                + list(image_dir.glob("*.jpeg"))
                + list(image_dir.glob("*.png"))
            )
            counts[split] = len(image_files)
        else:
            counts[split] = 0
    return counts
