"""Crack segmentation dataset implementation with configurable data loading.

This module provides the core dataset class for pavement crack segmentation
tasks. It supports flexible data loading strategies, in-memory caching for
performance, and extensive configuration through Hydra integration.

Key Features:
- Multiple data source support (file paths, PIL Images, numpy arrays)
- In-memory caching for improved training speed
- Robust error handling and fallback mechanisms
- Configurable data augmentation through Albumentations
- Sample limiting for development and testing
- Comprehensive validation and integrity checking

The module follows PyTorch Dataset conventions while providing additional
functionality specific to segmentation tasks:

- Binary mask validation and normalization
- Automatic image format conversion and color space handling
- Deterministic sample selection with seed control
- Memory-efficient caching strategies

Examples:
    Basic dataset creation with samples list:
    ```python
    from crackseg.data.dataset import CrackSegmentationDataset

    # Define sample pairs
    samples = [
        ("data/train/images/crack_001.jpg", "data/train/masks/crack_001.png"),
        ("data/train/images/crack_002.jpg", "data/train/masks/crack_002.png"),
    ]

    # Create dataset
    dataset = CrackSegmentationDataset(
        mode="train",
        samples_list=samples,
        image_size=(512, 512),
        in_memory_cache=True
    )

    # Access data
    sample = dataset[0]
    image, mask = sample["image"], sample["mask"]
    ```

    Factory-based creation with configuration:
    ```python
    from crackseg.data.dataset import create_crackseg_dataset
    from omegaconf import OmegaConf

    # Load configurations
    data_cfg = OmegaConf.load("configs/data/default.yaml")
    transform_cfg = OmegaConf.load(
        "configs/data/transform/augmentations.yaml"
    )

    # Create dataset via factory
    dataset = create_crackseg_dataset(
        data_cfg=data_cfg,
        transform_cfg=transform_cfg,
        mode="train",
        samples_list=samples,
        max_samples=1000  # Limit for fast development
    )
    ```

Performance Considerations:
- In-memory caching significantly speeds up training but requires sufficient
  RAM
- Sample limiting is useful for development and prototyping
- Error handling ensures robust operation with corrupted or missing files
- Transform caching at PIL level before applying Albumentations transformations

Integration:
- Designed for use with PyTorch DataLoader
- Integrates with Hydra configuration system
- Compatible with distributed training setups
- Supports custom sampling strategies

See Also:
    - src.data.transforms: Transform pipeline implementation
    - src.data.validation: Data validation utilities
    - src.data.factory: DataLoader creation utilities
    - configs/data/: Configuration examples and templates
"""

# Import the main components from specialized modules
from .base_dataset import CrackSegmentationDataset
from .dataset_factory import create_crackseg_dataset
from .dataset_utils import CacheItemType, SourceType

# Re-export the main classes and functions for backward compatibility
__all__ = [
    "CrackSegmentationDataset",
    "create_crackseg_dataset",
    "SourceType",
    "CacheItemType",
]
