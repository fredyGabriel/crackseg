"""High-level factory module for creating datasets and dataloaders from
configuration.

This module provides a unified interface for creating complete data pipelines
from Hydra configuration files. It handles the complex orchestration of dataset
creation, data splitting, distributed training setup, and dataloader
configuration with intelligent fallback mechanisms and robust error handling.

Key Features:
    - Configuration-driven dataset and dataloader creation
    - Automatic train/validation/test splitting with configurable ratios
    - Distributed training support with automatic rank detection
    - Memory-efficient loading with adaptive batch sizing
    - Robust fallback mechanisms for edge cases
    - Integration with custom sampling strategies

Core Components:
    - create_dataloaders_from_config(): Main factory function for complete
    data pipeline
    - _create_or_load_split_datasets(): Dataset splitting with fallback logic
    - _prepare_dataloader_params(): DataLoader configuration preparation

Architecture:
    Configuration -> Dataset Creation -> Data Splitting -> DataLoader Setup ->
    Complete Pipeline

Common Usage:
    # Standard configuration-based setup
    data_pipeline = create_dataloaders_from_config(
        data_config=cfg.data,
        transform_config=cfg.transforms,
        dataloader_config=cfg.dataloader
    )

    # Access components
    train_loader = data_pipeline["train"]["dataloader"]
    val_dataset = data_pipeline["val"]["dataset"]

    # Distributed training ready
    for batch in train_loader:
        # Automatically handles distributed sampling

Configuration Structure:
    data:
        data_root: "/path/to/dataset"
        train_split: 0.7
        val_split: 0.2
        test_split: 0.1
        in_memory_cache: false

    transforms:
        train: {...}    # Training augmentations
        val: {...}      # Validation transforms
        test: {...}     # Test transforms

    dataloader:
        batch_size: 16
        num_workers: 4
        distributed:
            enabled: true
        memory:
            adaptive_batch_size: true
            max_memory_mb: 8000

Error Handling:
    - Graceful fallback from optimized to standard dataset creation
    - Automatic distributed training detection and configuration
    - Memory management with adaptive batch sizing
    - Comprehensive validation of configuration parameters

Integration:
    - Works seamlessly with Hydra configuration system
    - Compatible with PyTorch distributed training
    - Supports custom dataset classes and sampling strategies
    - Integrates with experiment tracking and checkpointing

Performance Optimizations:
    - In-memory caching for small datasets
    - Adaptive batch sizing based on available memory
    - Optimized data splitting with efficient indexing
    - Multi-processing data loading with configurable workers

References:
    - Dataset: src.data.dataset.CrackSegmentationDataset
    - DataLoader: src.data.dataloader.create_dataloader
    - Splitting: src.data.splitting.create_split_datasets
    - Validation: src.data.validation.validate_data_config
"""

import warnings
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data.dataloader import DataLoaderConfig, create_dataloader
from src.data.dataset import create_crackseg_dataset
from src.data.validation import validate_data_config

from .dataset import CrackSegmentationDataset
from .distributed import get_rank, get_world_size
from .splitting import DatasetCreationConfig, create_split_datasets


def _create_or_load_split_datasets(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataloader_config: DictConfig,
    dataset_class: type[CrackSegmentationDataset],
) -> dict[str, Dataset[Any]]:
    """Create or load split datasets with intelligent fallback mechanism.

    Attempts to create datasets using the optimized splitting functionality,
    with automatic fallback to manual splitting if the optimized approach
    fails. This ensures robust dataset creation across different data
    configurations and file system states.

    Args:
        data_config: Data configuration containing paths, splits, and caching
            options. Must include 'data_root', split ratios, and optional
            caching settings.
        transform_config: Transform configurations for each split
            (train/val/test). Each split should have corresponding transform
            specifications.
        dataloader_config: DataLoader configuration including sample limits and
            memory settings that affect dataset creation.
        dataset_class: Dataset class to instantiate. Must be compatible with
            CrackSegmentationDataset interface and factory functions.

    Returns:
        dict[str, Dataset[Any]]: Dictionary mapping split names
            ('train', 'val', 'test') to their corresponding dataset instances.
            All datasets are ready for immediate use with dataloaders.

    Processing Strategy:
        1. Extract configuration parameters and sample limits
        2. Attempt optimized dataset creation via create_split_datasets()
        3. On failure, fallback to manual splitting with get_all_samples()
        4. Create individual datasets for each split with appropriate
        transforms
        5. Apply sample limits and caching as configured

    Fallback Logic:
        - Primary: Uses create_split_datasets() for efficient splitting
        - Fallback: Manual sample discovery and index-based splitting
        - Handles FileNotFoundError and RuntimeError gracefully
        - Preserves all configuration options in fallback mode

    Examples:
        >>> # Standard usage (handles fallback automatically)
        >>> datasets = _create_or_load_split_datasets(
        ...     data_config=cfg.data,
        ...     transform_config=cfg.transforms,
        ...     dataloader_config=cfg.dataloader,
        ...     dataset_class=CrackSegmentationDataset
        ... )
        >>> train_dataset = datasets["train"]
        >>> val_dataset = datasets["val"]

    Configuration Requirements:
        data_config must contain:
        - data_root: Path to dataset directory
        - train_split, val_split, test_split: Split ratios (must sum to 1.0)
        - seed: Random seed for reproducible splitting
        - in_memory_cache: Boolean for caching strategy

        transform_config must contain:
        - train, val, test: Transform configurations for each split

        dataloader_config may contain:
        - max_train_samples, max_val_samples, max_test_samples: Sample limits

    Error Handling:
        - FileNotFoundError: Triggers fallback splitting mechanism
        - RuntimeError: Handles data discovery and validation errors
        - ValueError: Missing transform configurations for splits
        - Maintains error context and provides informative messages

    Performance Considerations:
        - Optimized path uses efficient dataset splitting algorithms
        - Fallback ensures compatibility with legacy data structures
        - Sample limits applied consistently across both paths
        - Caching strategy preserved in all scenarios

    Memory Management:
        - Respects in_memory_cache setting for both paths
        - Sample limits help control memory usage
        - Efficient index-based splitting in fallback mode
    """
    data_root = data_config["data_root"]
    in_memory_cache = data_config.get("in_memory_cache", False)
    seed = data_config.get("seed", 42)
    ratios = {
        "train": data_config["train_split"],
        "val": data_config["val_split"],
        "test": data_config["test_split"],
    }
    max_train_samples = dataloader_config.get("max_train_samples", None)
    max_val_samples = dataloader_config.get("max_val_samples", None)
    max_test_samples = dataloader_config.get("max_test_samples", None)

    # Primary approach: Use optimized dataset creation
    try:
        dataset_creation_cfg = DatasetCreationConfig(
            data_root=data_root,
            transform_cfg=transform_config,
            dataset_cls=dataset_class,
            seed=seed,
            cache_flag=in_memory_cache,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )
        temp_split_datasets = create_split_datasets(
            config=dataset_creation_cfg
        )
        return cast(dict[str, Dataset[Any]], temp_split_datasets)
    except (FileNotFoundError, RuntimeError) as e:
        # Fallback approach: Manual sample discovery and splitting
        from .splitting import get_all_samples, split_indices

        all_samples = get_all_samples(data_root)
        if not all_samples:
            raise RuntimeError(f"No valid samples found in {data_root}") from e

        # Create index-based splits
        indices_map = split_indices(
            num_samples=len(all_samples),
            ratios=ratios,
            seed=seed,
            shuffle=True,
        )

        # Create datasets for each split
        split_datasets_fallback: dict[str, Dataset[Any]] = {}
        for split_name in ["train", "val", "test"]:
            split_indices_list = indices_map[split_name]
            split_samples_list = [all_samples[i] for i in split_indices_list]

            if split_name not in transform_config:
                raise ValueError(
                    f"Transform config missing for split: {split_name}"
                ) from e

            split_transform_cfg = transform_config[split_name]

            # Apply sample limits per split
            max_samples_for_split = None
            if split_name == "train":
                max_samples_for_split = max_train_samples
            elif split_name == "val":
                max_samples_for_split = max_val_samples
            elif split_name == "test":
                max_samples_for_split = max_test_samples

            # Create dataset with fallback configuration
            split_datasets_fallback[split_name] = create_crackseg_dataset(
                data_cfg=data_config,
                transform_cfg=split_transform_cfg,
                mode=split_name,
                samples_list=split_samples_list,
                in_memory_cache=in_memory_cache,
                max_samples=max_samples_for_split,
            )
        return split_datasets_fallback


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

    Configuration Process:
        1. Validate data configuration against schema requirements
        2. Create split datasets using optimized splitting with fallback
        3. Prepare comprehensive dataloader configuration with distributed
           support
        4. Create individual dataloaders for each split with appropriate
           settings
        5. Return complete pipeline ready for training/evaluation

    Distributed Training Handling:
        - Automatic detection of PyTorch distributed initialization
        - Rank and world size resolution from environment or configuration
        - Automatic sampler type switching for distributed mode
        - Warning system for configuration mismatches

    Examples:
        >>> # Standard configuration
        >>> config, rank, world_size, is_dist, batch_size = (
        ...     _prepare_dataloader_params(
        ...         dataloader_config=cfg.dataloader,
        ...         data_config=cfg.data
        ...     )
        ... )
        >>>
        >>> # Configuration will be automatically adapted for distributed
        >>> # training
        >>> if torch.distributed.is_initialized():
            ...     # rank and world_size will be properly set
            ...     # sampler will be configured for distributed training

    Memory Optimization Features:
        - Adaptive batch sizing based on available GPU memory
        - FP16 data loading for memory efficiency
        - Maximum memory limits to prevent OOM errors
        - Prefetch factor optimization for GPU utilization

    Sampling Configuration:
        - Random sampling for standard training
        - Distributed sampling for multi-GPU training
        - Custom sampler support with full parameter passing
        - Automatic switching based on training mode

    Advanced DataLoader Features:
        - Configurable worker processes with auto-detection
        - Pin memory optimization for GPU transfers
        - Drop last batch handling for consistent batch sizes
        - Prefetch factor tuning for performance

    Error Handling:
        - Invalid configuration values trigger warnings
        - Missing parameters filled with intelligent defaults
        - Distributed configuration mismatches automatically corrected
        - Type validation for all configuration parameters

    Performance Tuning:
        - Worker count optimization based on CPU cores and data complexity
        - Memory usage optimization with adaptive batch sizing
        - Prefetch tuning for optimal GPU utilization
        - Pin memory configuration for transfer speed

    Configuration Schema:
        dataloader:
            batch_size: 16
            num_workers: 4
            shuffle: true
            pin_memory: true
            prefetch_factor: 2
            drop_last: false
            distributed:
                enabled: true
                rank: 0
                world_size: 1
            sampler:
                enabled: true
                kind: "distributed"
                shuffle: true
            memory:
                fp16: true
                max_memory_mb: 8000
                adaptive_batch_size: true
    """
    # Extract distributed configuration
    dl_dist = dataloader_config.get("distributed", OmegaConf.create({}))
    is_distributed = dl_dist.get("enabled", False)
    rank_val = dl_dist.get("rank", 0)
    world_size_val = dl_dist.get("world_size", 1)

    # Auto-detect PyTorch distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_distributed = True
        rank_val = get_rank()
        world_size_val = get_world_size()

    # Configure sampling strategy
    sampler_config_to_pass: dict[str, Any]
    sampler_cfg_node = dataloader_config.get("sampler", OmegaConf.create({}))
    if sampler_cfg_node.get("enabled", False):
        converted_sampler_cfg = OmegaConf.to_container(
            sampler_cfg_node, resolve=True
        )
        if isinstance(converted_sampler_cfg, dict):
            sampler_config_to_pass = cast(
                dict[str, Any], converted_sampler_cfg
            )
            # Auto-correct sampler for distributed training
            if (
                is_distributed
                and sampler_config_to_pass.get("kind") != "distributed"
            ):
                warnings.warn(
                    "Distributed training detected but sampler kind is not "
                    "'distributed'. Switching to distributed sampler.",
                    stacklevel=2,
                )
                sampler_config_to_pass["kind"] = "distributed"
            # Clean up configuration flags
            if "enabled" in sampler_config_to_pass:
                del sampler_config_to_pass["enabled"]
        else:
            warnings.warn(
                "sampler_cfg_node.get('sampler') did not convert to a dict. "
                "Using {}.",
                stacklevel=2,
            )
            sampler_config_to_pass = {}
    else:
        sampler_config_to_pass = {}

    # Extract memory optimization settings
    memory_cfg = dataloader_config.get("memory", OmegaConf.create({}))

    # Extract core DataLoader parameters with intelligent defaults
    num_workers = dataloader_config.get(
        "num_workers", data_config.get("num_workers", -1)
    )
    shuffle = dataloader_config.get("shuffle", True)
    pin_memory = dataloader_config.get("pin_memory", True)
    prefetch_factor = dataloader_config.get("prefetch_factor", 2)
    fp16 = memory_cfg.get("fp16", False)
    max_memory_mb = memory_cfg.get("max_memory_mb", None)
    adaptive_batch_size = memory_cfg.get("adaptive_batch_size", False)
    drop_last = dataloader_config.get("drop_last", False)

    # Extract additional DataLoader kwargs (advanced PyTorch features)
    container = OmegaConf.to_container(dataloader_config, resolve=True)
    if not isinstance(container, dict):
        container = {}

    # Filter out known parameters to create clean kwargs dict
    str_items = cast(
        list[tuple[str, Any]],
        [
            (k, v)
            for k, v in container.items()
            if isinstance(k, str)
            and k
            not in [
                "distributed",
                "sampler",
                "memory",
                "batch_size",
                "num_workers",
                "shuffle",
                "pin_memory",
                "prefetch_factor",
                "drop_last",
                "max_train_samples",
                "max_val_samples",
                "max_test_samples",
            ]
        ],
    )
    dataloader_extra_kwargs: dict[str, Any] = dict(str_items)
    if drop_last:
        dataloader_extra_kwargs["drop_last"] = drop_last

    # Create comprehensive DataLoader configuration
    loader_config = DataLoaderConfig(
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        sampler_config=sampler_config_to_pass,
        rank=rank_val if is_distributed else None,
        world_size=world_size_val if is_distributed else None,
        fp16=fp16,
        max_memory_mb=max_memory_mb,
        adaptive_batch_size=adaptive_batch_size,
        dataloader_extra_kwargs=dataloader_extra_kwargs,
    )

    # Extract batch size with fallback hierarchy
    batch_size = dataloader_config.get(
        "batch_size", data_config.get("batch_size", 8)
    )
    return loader_config, rank_val, world_size_val, is_distributed, batch_size


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
        data_config: Complete data configuration including paths, splits, and
            settings. Must contain data_root, split ratios, and optional
            caching/processing settings.
        transform_config: Transform configurations for each data split. Each
            split (train/val/test) must have corresponding transform
            specifications.
        dataloader_config: DataLoader configuration including batch size,
            workers, distributed settings, memory optimization, and sampling
            strategies.
        dataset_class: Dataset class to instantiate. Defaults to
            CrackSegmentationDataset. Must be compatible with the factory
            function interface.

    Returns:
        dict[str, dict[str, Dataset[Any] | DataLoader[Any]]]: Nested
            dictionary structure:
                {
                    "train": {
                        "dataset": Dataset instance for training,
                        "dataloader": DataLoader instance for training
                    },
                    "val": {
                        "dataset": Dataset instance for validation,
                        "dataloader": DataLoader instance for validation
                    },
                    "test": {
                        "dataset": Dataset instance for testing,
                        "dataloader": DataLoader instance for testing
                    }
                }

    Configuration Process:
        1. Validate data configuration against schema requirements
        2. Create split datasets using optimized splitting with fallback
        3. Prepare comprehensive dataloader configuration with distributed
           support
        4. Create individual dataloaders for each split with appropriate
           settings
        5. Return complete pipeline ready for training/evaluation

    Configuration Validation:
        - Schema validation for all configuration components
        - Path existence verification for data directories
        - Split ratio validation (must sum to 1.0)
        - Transform configuration completeness check
        - Memory and performance setting validation

    Examples:
        >>> # Complete pipeline creation from Hydra config
        >>> @hydra.main(config_path="configs", config_name="main")
        >>> def main(cfg: DictConfig) -> None:
        ...     data_pipeline = create_dataloaders_from_config(
        ...         data_config=cfg.data,
        ...         transform_config=cfg.transforms,
        ...         dataloader_config=cfg.dataloader
        ...     )
        ...
        ...     # Access components
        ...     train_loader = data_pipeline["train"]["dataloader"]
        ...     val_dataset = data_pipeline["val"]["dataset"]
        ...
        ...     # Ready for training
        ...     for epoch in range(num_epochs):
        ...         for batch in train_loader:
        ...             # Training loop
        >>>
        >>> # Manual configuration
        >>> data_cfg = OmegaConf.create({
        ...     "data_root": "/path/to/data",
        ...     "train_split": 0.7,
        ...     "val_split": 0.2,
        ...     "test_split": 0.1
        ... })
        >>> transform_cfg = OmegaConf.create({
        ...     "train": {"resize": {"height": 512, "width": 512}},
        ...     "val": {"resize": {"height": 512, "width": 512}},
        ...     "test": {"resize": {"height": 512, "width": 512}}
        ... })
        >>> dataloader_cfg = OmegaConf.create({
        ...     "batch_size": 16,
        ...     "num_workers": 4
        ... })
        >>> pipeline = create_dataloaders_from_config(
        ...     data_cfg, transform_cfg, dataloader_cfg
        ... )

    Distributed Training Support:
        - Automatic detection of PyTorch distributed environment
        - Proper sampler configuration for each split
        - Rank-aware shuffling (disabled for validation/test)
        - Memory optimization across multiple processes

    Memory Management:
        - Adaptive batch sizing to prevent OOM errors
        - Memory-efficient data loading with configurable workers
        - Optional FP16 data loading for GPU memory optimization
        - Smart caching strategies for small datasets

    Performance Optimizations:
        - Efficient data splitting algorithms with fallback mechanisms
        - Optimized worker process configuration
        - Pin memory for fast GPU transfers
        - Prefetch factor tuning for maximum throughput

    DataLoader Behavior Per Split:
        - Train: Shuffling enabled, distributed sampling if applicable
        - Val: Shuffling disabled, deterministic iteration order
        - Test: Shuffling disabled, consistent evaluation results

    Error Handling:
        - Comprehensive configuration validation before processing
        - Graceful fallback for dataset creation failures
        - Informative error messages with configuration context
        - Automatic recovery from common configuration issues

    Integration Features:
        - Seamless Hydra configuration integration
        - Compatible with experiment tracking systems
        - Supports custom dataset classes and transform pipelines
        - Ready for integration with training loops and evaluation scripts

    Configuration Requirements:
        data_config:
            - data_root: str (path to dataset)
            - train_split, val_split, test_split: float (ratios summing to 1.0)
            - Optional: seed, in_memory_cache, num_workers

        transform_config:
            - train: dict (training transforms)
            - val: dict (validation transforms)
            - test: dict (test transforms)

        dataloader_config:
            - batch_size: int
            - Optional: num_workers, distributed settings, memory options

    Performance Considerations:
        - Dataset creation time scales with data size and splitting complexity
        - Memory usage dependent on caching strategy and batch size
        - Worker processes should match CPU cores for optimal throughput
        - Distributed training adds communication overhead but enables scaling

    References:
        - Dataset Creation: src.data.dataset.create_crackseg_dataset
        - DataLoader Creation: src.data.dataloader.create_dataloader
        - Configuration Validation: src.data.validation.validate_data_config
        - Distributed Utilities: src.data.distributed
    """
    # Validate configuration before processing
    validate_data_config(data_config)

    # Create datasets for all splits with intelligent fallback
    split_datasets = _create_or_load_split_datasets(
        data_config, transform_config, dataloader_config, dataset_class
    )

    # Prepare comprehensive dataloader configuration
    (
        loader_config,
        rank_val,
        world_size_val,
        is_distributed,
        batch_size,
    ) = _prepare_dataloader_params(dataloader_config, data_config)

    # Create dataloaders for each split with appropriate configuration
    result: dict[str, dict[str, Dataset[Any] | DataLoader[Any]]] = {}
    for split_name, dataset_instance in split_datasets.items():
        # Create split-specific configuration
        current_loader_config = OmegaConf.structured(loader_config)

        # Disable shuffling for validation and test sets
        if split_name != "train":
            current_loader_config.shuffle = False

        # Configure distributed settings per split
        if not is_distributed:
            current_loader_config.rank = None
            current_loader_config.world_size = None
        else:
            current_loader_config.rank = rank_val
            current_loader_config.world_size = world_size_val

        # Create dataloader with split-specific configuration
        dataloader = create_dataloader(
            dataset=dataset_instance,
            batch_size=batch_size,
            config=current_loader_config,
        )

        # Store both dataset and dataloader for complete access
        result[split_name] = {
            "dataset": dataset_instance,
            "dataloader": dataloader,
        }
    return result
