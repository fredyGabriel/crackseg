"""Configuration processing utilities for data loader factory.

This module provides utilities for processing and validating DataLoader
configuration parameters with support for distributed training and memory
optimization.
"""

import warnings

import torch
from omegaconf import DictConfig

from crackseg.data.loaders import DataLoaderConfig
from crackseg.data.utils import get_rank, get_world_size


def _prepare_dataloader_params(
    dataloader_config: DictConfig, data_config: DictConfig
) -> tuple[DataLoaderConfig, int, int, bool, int]:
    """Prepare comprehensive DataLoader configuration with distributed
    training support.

    Processes and validates all dataloader parameters, automatically detecting
    distributed training environment and configuring appropriate sampling
    strategies. Handles memory optimization, worker processes, and advanced
    PyTorch DataLoader features with intelligent defaults.

    Args:
        dataloader_config: DataLoader configuration containing batch size,
            workers, distributed settings, memory options, and sampling
            configurations.
        data_config: Data configuration providing fallback values and
            additional context for dataloader parameter resolution.

    Returns:
        tuple containing:
            - DataLoaderConfig: Comprehensive configuration object for
            dataloader creation
            - int: Process rank for distributed training (0 for single-process)
            - int: World size for distributed training (1 for single-process)
            - bool: Whether distributed training is enabled and properly
            configured
            - int: Batch size to use for all dataloaders
    """
    # Detect distributed training environment
    is_distributed = torch.distributed.is_initialized()
    rank = get_rank()
    world_size = get_world_size()

    # Extract batch size with validation
    batch_size = dataloader_config.get("batch_size", 16)
    if batch_size <= 0:
        warnings.warn(
            f"Invalid batch_size: {batch_size}. Using default: 16",
            stacklevel=2,
        )
        batch_size = 16

    # Create DataLoaderConfig with distributed support
    loader_config = DataLoaderConfig(
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=dataloader_config.get("shuffle", True),
        pin_memory=dataloader_config.get("pin_memory", True),
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
    )

    # Configure distributed settings if needed
    if is_distributed:
        loader_config.distributed = True
        loader_config.rank = rank
        loader_config.world_size = world_size

    return loader_config, rank, world_size, is_distributed, batch_size


def validate_factory_config(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataloader_config: DictConfig,
) -> None:
    """Validate factory configuration parameters.

    Args:
        data_config: Data configuration to validate.
        transform_config: Transform configuration to validate.
        dataloader_config: DataLoader configuration to validate.

    Raises:
        ValueError: If configuration is invalid.
    """
    # Validate data configuration
    if not hasattr(data_config, "data_root"):
        raise ValueError("data_config must contain 'data_root'")

    # Validate transform configuration
    if not hasattr(transform_config, "train"):
        raise ValueError("transform_config must contain 'train' section")

    # Validate dataloader configuration
    if not hasattr(dataloader_config, "batch_size"):
        warnings.warn(
            "dataloader_config missing 'batch_size', using default: 16",
            stacklevel=2,
        )
