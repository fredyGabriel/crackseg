"""DataLoader implementations for crack segmentation.

This module provides comprehensive DataLoader creation capabilities with
memory optimization, distributed training support, and intelligent defaults.

Main Components:
- DataLoaderConfig: Configuration class for DataLoader settings
- create_dataloader: Main factory function for DataLoader creation
- Memory optimization utilities
- Worker configuration utilities
- Validation functions

Examples:
    Basic usage:
    ```python
    from crackseg.data.loaders import create_dataloader, DataLoaderConfig

    # Simple configuration
    dataloader = create_dataloader(dataset, batch_size=32)

    # Advanced configuration
    config = DataLoaderConfig(
        num_workers=8,
        pin_memory=True,
        adaptive_batch_size=True
    )
    dataloader = create_dataloader(dataset, batch_size=64, config=config)
    ```

    Distributed training:
    ```python
    config = DataLoaderConfig(
        sampler_config={"kind": "distributed", "shuffle": True},
        rank=local_rank,
        world_size=world_size
    )
    dataloader = create_dataloader(dataset, batch_size=32, config=config)
    ```
"""

from .config import DataLoaderConfig
from .factory import create_dataloader, create_dataloader_with_validation
from .memory import (
    calculate_adaptive_batch_size,
    configure_memory_settings,
    get_memory_info,
)
from .validation import (
    validate_dataloader_params,
    validate_memory_config,
    validate_sampler_config,
)
from .workers import (
    configure_num_workers,
    get_worker_info,
    validate_worker_config,
)

__all__ = [
    # Main factory functions
    "create_dataloader",
    "create_dataloader_with_validation",
    # Configuration
    "DataLoaderConfig",
    # Memory optimization
    "calculate_adaptive_batch_size",
    "configure_memory_settings",
    "get_memory_info",
    # Validation utilities
    "validate_dataloader_params",
    "validate_memory_config",
    "validate_sampler_config",
    # Worker configuration
    "configure_num_workers",
    "get_worker_info",
    "validate_worker_config",
]
