"""DataLoader configuration for crack segmentation.

This module provides the DataLoaderConfig class and related configuration
utilities for creating optimized PyTorch DataLoaders.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataLoaderConfig:
    """
    Comprehensive configuration for creating optimized PyTorch DataLoaders.

    This configuration class provides fine-grained control over DataLoader
    behavior with intelligent defaults for different use cases. It supports
    both simple configurations for development and advanced settings for
    production training.

    Attributes:
        num_workers: Number of worker processes for data loading.
            Use -1 for automatic detection (recommended).

        shuffle: Whether to shuffle data at every epoch.
            Automatically disabled when using custom samplers.

        pin_memory: Whether to pin memory for faster GPU transfer.
            Recommended True for CUDA training, False for CPU-only.

        prefetch_factor: Number of samples loaded in advance by each worker.
            Higher values use more memory but may improve throughput.

        sampler_config: Configuration for custom samplers.
            Dict with 'kind' key specifying sampler type and parameters.

        rank: Process rank for distributed training (0-based).
            Required when using distributed samplers.

        world_size: Total number of processes in distributed training.
            Required when using distributed samplers.

        fp16: Whether to enable FP16 optimizations.
            Enables memory optimizations for mixed precision training.

        max_memory_mb: Maximum GPU memory to use for adaptive batch sizing.
            Prevents OOM by limiting memory usage. None for no limit.

        adaptive_batch_size: Whether to enable automatic batch size adaptation.
            Adjusts batch size based on available GPU memory.

        dataloader_extra_kwargs: Additional keyword arguments for DataLoader.
            For advanced use cases requiring custom DataLoader parameters.

    Examples:
        Development configuration:
        ```python
        config = DataLoaderConfig(
            num_workers=2,
            shuffle=True,
            pin_memory=False,  # CPU training
            adaptive_batch_size=False
        )
        ```

        Production configuration:
        ```python
        config = DataLoaderConfig(
            num_workers=-1,  # Auto-detect
            pin_memory=True,
            adaptive_batch_size=True,
            max_memory_mb=12000,  # Limit for 16GB GPU
            fp16=True
        )
        ```

        Distributed training configuration:
        ```python
        config = DataLoaderConfig(
            sampler_config={
                "kind": "distributed",
                "shuffle": True,
                "drop_last": True
            },
            rank=local_rank,
            world_size=world_size,
            num_workers=8
        )
        ```

    Note:
        When using distributed samplers, shuffle is automatically set to False
        to avoid conflicts. The sampler handles shuffling internally.
    """

    num_workers: int = -1
    shuffle: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    sampler_config: dict[str, Any] | None = None
    rank: int | None = None
    world_size: int | None = None
    fp16: bool = False
    max_memory_mb: float | None = None
    adaptive_batch_size: bool = False
    dataloader_extra_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.num_workers < -1:
            raise ValueError(
                f"num_workers must be >= -1, got {self.num_workers}"
            )

        if self.prefetch_factor < 1:
            raise ValueError(
                f"prefetch_factor must be >= 1, got {self.prefetch_factor}"
            )

        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError(
                f"max_memory_mb must be positive, got {self.max_memory_mb}"
            )

        if self.rank is not None and self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")

        if self.world_size is not None and self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")

        # Validate distributed training parameters
        if self.rank is not None and self.world_size is None:
            raise ValueError(
                "world_size must be provided when rank is specified"
            )

        if self.world_size is not None and self.rank is None:
            raise ValueError(
                "rank must be provided when world_size is specified"
            )

        # Validate sampler configuration
        if self.sampler_config is not None:
            if not isinstance(self.sampler_config, dict):
                raise ValueError("sampler_config must be a dictionary")

            if "kind" not in self.sampler_config:
                raise ValueError("sampler_config must contain 'kind' key")

        # Warn about potential issues
        if self.fp16 and not torch.cuda.is_available():
            warnings.warn(
                "FP16 requested but CUDA not available. "
                "Mixed precision optimizations will be disabled.",
                stacklevel=2,
            )

        if self.pin_memory and not torch.cuda.is_available():
            warnings.warn(
                "pin_memory=True requested but CUDA not available. "
                "Setting pin_memory=False.",
                stacklevel=2,
            )
