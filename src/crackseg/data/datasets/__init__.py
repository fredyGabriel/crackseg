"""Dataset implementations for crack segmentation.

This package contains dataset classes and utilities for crack segmentation,
including the main CrackSegmentationDataset and supporting components.
"""

from .base_dataset import CrackSegmentationDataset
from .cache_manager import CacheManager
from .dataset import CrackSegmentationDataset as SimpleDataset
from .loaders import ImageLoader, MaskLoader
from .types import CacheItemType, SourceType

__all__ = [
    "CrackSegmentationDataset",
    "SimpleDataset",
    "CacheManager",
    "ImageLoader",
    "MaskLoader",
    "CacheItemType",
    "SourceType",
]
