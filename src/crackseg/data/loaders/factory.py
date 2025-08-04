"""DataLoader factory for crack segmentation.

This module provides the main DataLoader creation functionality with
comprehensive configuration and optimization capabilities.
"""

import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..utils.collate import mixed_collate_fn
from ..utils.sampler import SamplerFactoryArgs, sampler_factory
from .config import DataLoaderConfig
from .memory import calculate_adaptive_batch_size
from .validation import validate_dataloader_params
from .workers import configure_num_workers


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int = 32,
    config: DataLoaderConfig | None = None,
) -> DataLoader[Any]:
    """
    Creates and configures a PyTorch DataLoader with sensible defaults.

    This function provides a comprehensive DataLoader creation interface
    with intelligent defaults, memory optimization, and distributed training
    support. It automatically handles parameter validation, memory management,
    and sampler configuration.

    Args:
        dataset: The dataset from which to load the data.
        batch_size: How many samples per batch to load. Default: 32.
        config: Configuration object for the dataloader.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.

    Raises:
        ValueError: If batch_size or prefetch_factor are not positive,
            or if num_workers is less than -1.
        ValueError: If both shuffle and sampler are set (PyTorch limitation).

    Examples:
        Basic DataLoader creation:
        ```python
        from crackseg.data.loaders import create_dataloader, DataLoaderConfig

        # Simple configuration
        dataloader = create_dataloader(
            dataset=train_dataset,
            batch_size=32
        )

        # Advanced configuration with memory management
        config = DataLoaderConfig(
            num_workers=8,
            pin_memory=True,
            adaptive_batch_size=True,
            max_memory_mb=8000
        )

        dataloader = create_dataloader(
            dataset=train_dataset,
            batch_size=64,
            config=config
        )
        ```

        Distributed training setup:
        ```python
        config = DataLoaderConfig(
            sampler_config={
                "kind": "distributed",
                "shuffle": True
            },
            rank=local_rank,
            world_size=world_size
        )

        dataloader = create_dataloader(dataset, batch_size=32, config=config)
        ```

    Note:
        If using DistributedSampler, you must call `set_epoch(epoch)` on
        the sampler at the beginning of each epoch to ensure correct
        shuffling across processes.

        When using fp16=True, you should wrap your training loop with
        torch.cuda.amp.autocast() context manager.
    """
    if config is None:
        config = DataLoaderConfig()

    # --- Parameter Validation ---
    validate_dataloader_params(
        batch_size, config.prefetch_factor, config.num_workers
    )

    # --- Memory Optimization (if requested) ---
    actual_batch_size = calculate_adaptive_batch_size(
        current_batch_size=batch_size,
        adaptive_enabled=config.adaptive_batch_size,
        max_memory_limit_mb=config.max_memory_mb,
    )

    # --- Mixed Precision ---
    if config.fp16 and not torch.cuda.is_available():
        warnings.warn(
            "Mixed precision (fp16) requested but CUDA not available. "
            "Falling back to standard precision.",
            stacklevel=2,
        )

    # --- Determine num_workers ---
    actual_num_workers = configure_num_workers(config.num_workers)

    # --- Determine pin_memory ---
    can_pin_memory = config.pin_memory and torch.cuda.is_available()
    if config.pin_memory and not can_pin_memory:
        warnings.warn(
            "pin_memory=True requires CUDA availability. "
            "Setting pin_memory=False.",
            stacklevel=2,
        )

    # --- Sampler logic ---
    sampler_instance, current_shuffle_status = _create_sampler_from_config(
        sampler_config_param=config.sampler_config,
        dataset_param=dataset,
        world_size_param=config.world_size,
        rank_param=config.rank,
        shuffle_param=config.shuffle,
    )

    # --- Create DataLoader ---
    actual_prefetch_factor: int | None = (
        config.prefetch_factor if actual_num_workers > 0 else None
    )
    dataloader_kwargs = (
        config.dataloader_extra_kwargs
        if config.dataloader_extra_kwargs
        else {}
    )

    dataloader_instance = DataLoader(
        dataset=dataset,
        batch_size=actual_batch_size,
        shuffle=current_shuffle_status if sampler_instance is None else False,
        sampler=sampler_instance,
        num_workers=actual_num_workers,
        pin_memory=can_pin_memory,
        prefetch_factor=actual_prefetch_factor,
        persistent_workers=True if actual_num_workers > 0 else False,
        collate_fn=mixed_collate_fn,
        **dataloader_kwargs,
    )

    return dataloader_instance


def _create_sampler_from_config(
    sampler_config_param: dict[str, Any] | None,
    dataset_param: Dataset[Any],
    world_size_param: int | None,
    rank_param: int | None,
    shuffle_param: bool,
) -> tuple[Sampler[Any] | None, bool]:
    """
    Create sampler from configuration dictionary.

    Args:
        sampler_config_param: Sampler configuration dictionary.
        dataset_param: Dataset to create sampler for.
        world_size_param: World size for distributed training.
        rank_param: Rank for distributed training.
        shuffle_param: Whether to shuffle data.

    Returns:
        tuple: (sampler_instance, shuffle_status)

    Note:
        This function handles the complex logic of sampler creation
        and ensures compatibility between different sampler types.
    """
    if sampler_config_param is None:
        return None, shuffle_param

    # Extract sampler configuration
    sampler_kind = sampler_config_param.get("kind", "random")
    sampler_args = {
        k: v for k, v in sampler_config_param.items() if k != "kind"
    }

    # Create sampler factory arguments
    factory_args = SamplerFactoryArgs(
        shuffle=shuffle_param,
        rank=rank_param,
        num_replicas=world_size_param,
        **sampler_args,
    )

    try:
        # Create sampler using factory
        sampler_instance = sampler_factory(
            sampler_kind, dataset_param, factory_args
        )

        # Update shuffle status based on sampler
        if sampler_instance is not None:
            # When using custom samplers, let the sampler handle shuffling
            current_shuffle_status = False
        else:
            current_shuffle_status = shuffle_param

        return sampler_instance, current_shuffle_status

    except Exception as e:
        warnings.warn(
            f"Failed to create sampler '{sampler_kind}': {e}. "
            "Falling back to default behavior.",
            stacklevel=2,
        )
        return None, shuffle_param


def create_dataloader_with_validation(
    dataset: Dataset[Any],
    batch_size: int = 32,
    config: DataLoaderConfig | None = None,
    validate_dataset: bool = True,
) -> DataLoader[Any]:
    """
    Create DataLoader with additional validation.

    Args:
        dataset: The dataset from which to load the data.
        batch_size: How many samples per batch to load.
        config: Configuration object for the dataloader.
        validate_dataset: Whether to validate dataset properties.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.

    Examples:
        ```python
        dataloader = create_dataloader_with_validation(
            dataset=train_dataset,
            batch_size=32,
            validate_dataset=True
        )
        ```
    """
    # Create the dataloader
    dataloader = create_dataloader(dataset, batch_size, config)

    # Additional validation if requested
    if validate_dataset:
        _validate_dataset_compatibility(dataset, dataloader)

    return dataloader


def _validate_dataset_compatibility(
    dataset: Dataset[Any], dataloader: DataLoader[Any]
) -> None:
    """
    Validate dataset compatibility with DataLoader configuration.

    Args:
        dataset: The dataset to validate.
        dataloader: The created DataLoader instance.

    Note:
        This function performs additional validation to ensure
        the dataset works well with the DataLoader configuration.
    """
    # Check dataset length
    if hasattr(dataset, "__len__"):
        try:
            dataset_length = len(dataset)  # type: ignore
            batch_size = dataloader.batch_size

            if batch_size is not None and dataset_length < batch_size:
                warnings.warn(
                    f"Dataset length ({dataset_length}) is smaller than "
                    f"batch_size ({batch_size}). This may cause issues.",
                    stacklevel=2,
                )
        except (TypeError, AttributeError):
            # Dataset doesn't support len() or has issues
            pass

    # Check for potential memory issues
    if hasattr(dataset, "get_sample_size"):
        try:
            sample_size = dataset.get_sample_size()
            if sample_size > 100 * 1024 * 1024:  # 100MB
                warnings.warn(
                    f"Large sample size detected ({sample_size / (1024 * 1024):.1f}MB). "
                    "Consider reducing batch_size or using memory optimization.",
                    stacklevel=2,
                )
        except Exception:
            pass  # Sample size estimation not available
