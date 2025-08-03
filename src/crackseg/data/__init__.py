"""Crack segmentation data pipeline.

This module provides comprehensive data loading and processing capabilities
for crack segmentation tasks. It includes dataset implementations,
transformation pipelines, DataLoader configurations, and factory functions
for creating complete data pipelines.

Main Components:
    - Datasets: CrackSegmentationDataset and related dataset classes
    - Transforms: Image transformation pipelines and utilities
    - DataLoaders: Optimized DataLoader creation with memory management
    - Factory: Factory functions for creating complete data pipelines
    - Validation: Configuration validation utilities
    - Utils: General utility functions for data processing

Key Features:
    - Modular dataset architecture with caching support
    - Comprehensive transformation pipeline with Albumentations integration
    - Memory-optimized DataLoader creation with distributed training support
    - Factory functions for complete pipeline orchestration
    - Configuration validation and error handling
    - Utility functions for data splitting and processing

Common Usage:
    # Create complete data pipeline
    from crackseg.data import create_dataloaders_from_config

    data_pipeline = create_dataloaders_from_config(
        data_config=cfg.data,
        transform_config=cfg.transforms,
        dataloader_config=cfg.dataloader
    )

    # Access components
    train_loader = data_pipeline["train"]["dataloader"]
    val_dataset = data_pipeline["val"]["dataset"]

    # Create individual components
    from crackseg.data import CrackSegmentationDataset, create_dataloader

    dataset = CrackSegmentationDataset(data_root="/path/to/data")
    dataloader = create_dataloader(dataset, batch_size=16)

Integration:
    - Seamless integration with Hydra configuration system
    - Compatible with PyTorch distributed training
    - Supports custom dataset classes and sampling strategies
    - Integrates with experiment tracking and checkpointing

Performance Features:
    - Memory-optimized data loading with adaptive batch sizing
    - Multi-processing support with configurable workers
    - Pin memory optimization for fast GPU transfers
    - Prefetch factor tuning for maximum throughput
    - Smart caching strategies for small datasets

Configuration Support:
    - Flexible configuration formats (dict, DictConfig)
    - Automatic parameter validation and correction
    - Intelligent defaults for missing parameters
    - Comprehensive error reporting and debugging

References:
    - Configuration: configs/data/, configs/transforms/, configs/dataloader/
    - Documentation: docs/guides/data_pipeline.md
    - Examples: scripts/examples/data_loading_examples.py
"""

from crackseg.data.datasets import CrackSegmentationDataset
from crackseg.data.factory import create_dataloaders_from_config
from crackseg.data.loaders import create_dataloader

__all__ = [
    "CrackSegmentationDataset",
    "create_dataloader",
    "create_dataloaders_from_config",
]
