"""DataLoader factory functions for creating optimized data pipelines.

This module provides factory functions for creating DataLoaders from
configuration with comprehensive optimization features including distributed
training support, memory management, and performance tuning.

Key Features:
    - Main factory function for complete data pipeline creation
    - Distributed training support with automatic detection
    - Memory optimization with adaptive batch sizing
    - Performance tuning with worker optimization
    - Comprehensive error handling and validation

Core Functions:
    - create_dataloaders_from_config(): Main factory function for complete
      data pipeline

Configuration Support:
    - Hydra configuration system integration
    - Flexible configuration formats (dict, DictConfig)
    - Automatic parameter validation and correction
    - Intelligent defaults for missing parameters

Distributed Training:
    - Automatic PyTorch distributed environment detection
    - Proper sampler configuration for multi-GPU training
    - Rank-aware shuffling and sampling
    - Memory optimization across multiple processes

Memory Management:
    - Adaptive batch sizing to prevent OOM errors
    - FP16 data loading for GPU memory optimization
    - Configurable memory limits and optimization strategies
    - Smart caching for small datasets

Performance Optimizations:
    - Worker process optimization based on system resources
    - Pin memory configuration for fast GPU transfers
    - Prefetch factor tuning for maximum throughput
    - Efficient data splitting algorithms

Common Usage:
    # Complete pipeline creation from Hydra config
    data_pipeline = create_dataloaders_from_config(
        data_config=cfg.data,
        transform_config=cfg.transforms,
        dataloader_config=cfg.dataloader
    )

    # Access components
    train_loader = data_pipeline["train"]["dataloader"]
    val_dataset = data_pipeline["val"]["dataset"]

Integration:
    - Used by training pipelines for data loading setup
    - Compatible with PyTorch distributed training
    - Integrates with Hydra configuration system
    - Supports custom dataset classes and sampling strategies

Error Handling:
    - Comprehensive validation of configuration parameters
    - Clear error messages for debugging
    - Automatic correction of common configuration issues
    - Graceful fallback for missing parameters

References:
    - Dataset: src.data.dataset.CrackSegmentationDataset
    - DataLoader: src.data.loaders.create_dataloader
    - Validation: src.data.validation.validate_data_config
    - Configuration: configs/data/ and configs/dataloader/
"""

from typing import Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from crackseg.data.datasets.base_dataset import CrackSegmentationDataset
from crackseg.data.loaders import create_dataloader
from crackseg.data.validation import validate_data_config

from .config_processor import (
    _prepare_dataloader_params,
    validate_factory_config,
)
from .dataset_creator import _create_or_load_split_datasets


def create_dataloaders_from_config(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataloader_config: DictConfig,
    dataset_class: type[CrackSegmentationDataset] = CrackSegmentationDataset,
) -> dict[str, dict[str, Dataset[Any] | DataLoader[Any]]]:
    """Create complete data pipeline with datasets and dataloaders from
    configuration.

    This is the main factory function that orchestrates the entire data
    pipeline creation process. It provides a one-stop solution for converting
    Hydra configurations into ready-to-use PyTorch datasets and dataloaders
    with comprehensive error handling and optimization features.

    Args:
        data_config: Data configuration containing data paths, split ratios,
            and dataset-specific parameters.
        transform_config: Transform configuration defining augmentation
            pipelines for each split.
        dataloader_config: DataLoader configuration with batch size, workers,
            memory settings, and distributed training parameters.
        dataset_class: Dataset class to use for creating datasets.
            Defaults to CrackSegmentationDataset.

    Returns:
        Dictionary containing complete data pipeline with structure:
        {
            "train": {"dataset": train_dataset, "dataloader": train_loader},
            "val": {"dataset": val_dataset, "dataloader": val_loader},
            "test": {"dataset": test_dataset, "dataloader": test_loader}
        }

    Example:
        >>> # Complete pipeline creation
        >>> pipeline = create_dataloaders_from_config(
        ...     data_config=cfg.data,
        ...     transform_config=cfg.transforms,
        ...     dataloader_config=cfg.dataloader
        ... )
        >>>
        >>> # Access components
        >>> train_loader = pipeline["train"]["dataloader"]
        >>> val_dataset = pipeline["val"]["dataset"]

    Raises:
        ValueError: If configuration is invalid or required parameters
            are missing.
        RuntimeError: If dataset creation fails or DataLoader
            initialization errors occur.

    Note:
        This function automatically handles distributed training setup,
        memory optimization, and performance tuning based on the provided
        configuration. It validates all parameters and provides intelligent
        defaults for missing values.
    """
    # Validate configuration
    validate_factory_config(data_config, transform_config, dataloader_config)
    validate_data_config(data_config)

    # Prepare DataLoader parameters
    loader_config, _rank, _world_size, _is_distributed, batch_size = (
        _prepare_dataloader_params(dataloader_config, data_config)
    )

    # Create split datasets
    split_datasets = _create_or_load_split_datasets(
        data_config=data_config,
        transform_config=transform_config,
        dataloader_config=dataloader_config,
        dataset_class=dataset_class,
    )

    # Create DataLoaders for each split
    result = {}
    for split_name, dataset in split_datasets.items():
        # Create DataLoader
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            config=loader_config,
        )

        result[split_name] = {
            "dataset": dataset,
            "dataloader": dataloader,
        }

    return result
