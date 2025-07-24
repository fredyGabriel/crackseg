#!/usr/bin/env python3
"""
Custom collate functions for crack segmentation datasets.

This module provides collate functions that properly handle dictionary-based
datasets like CrackSegmentationDataset, ensuring that batches are correctly
formed with proper tensor stacking.
"""

from typing import Any

import torch
from torch.utils.data import default_collate


def dict_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Custom collate function for dictionary-based datasets.

    This function properly handles batches from CrackSegmentationDataset
    which returns dictionaries with 'image' and 'mask' keys.

    Args:
        batch: List of dictionaries, each containing 'image' and 'mask' tensors

    Returns:
        Dictionary with batched tensors:
        - 'image': torch.Tensor of shape (B, C, H, W)
        - 'mask': torch.Tensor of shape (B, 1, H, W)

    Example:
        >>> dataset = CrackSegmentationDataset(...)
        >>> dataloader = DataLoader(
        ...     dataset, batch_size=4, collate_fn=dict_collate_fn
        ... )
        >>> for batch in dataloader:
        ...     images = batch['image']  # (4, 3, 256, 256)
        ...     masks = batch['mask']    # (4, 1, 256, 256)
    """
    if not batch:
        raise ValueError("Empty batch provided to dict_collate_fn")

    # Verify all samples have the same keys
    first_keys = set(batch[0].keys())
    for i, sample in enumerate(batch):
        if set(sample.keys()) != first_keys:
            raise ValueError(
                f"Sample {i} has different keys: {set(sample.keys())} vs "
                f"{first_keys}"
            )

    # Stack tensors for each key
    result = {}
    for key in first_keys:
        tensors = [sample[key] for sample in batch]
        result[key] = torch.stack(tensors, dim=0)

    return result


def mixed_collate_fn(batch: list[Any]) -> Any:
    """
    Collate function that handles both dictionary and tuple batches.

    This function automatically detects the batch type and applies the
    appropriate collation method:
    - For dictionary batches: Uses dict_collate_fn
    - For tuple batches: Uses default_collate

    Args:
        batch: List of samples (either dict or tuple)

    Returns:
        Batched data in the same format as input samples
    """
    if not batch:
        raise ValueError("Empty batch provided to mixed_collate_fn")

    # Check if all samples are dictionaries
    if all(isinstance(sample, dict) for sample in batch):
        return dict_collate_fn(batch)

    # Check if all samples are tuples
    if all(isinstance(sample, tuple) for sample in batch):
        return default_collate(batch)

    # Mixed types - use default collate
    return default_collate(batch)
