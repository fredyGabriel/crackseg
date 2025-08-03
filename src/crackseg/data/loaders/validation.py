"""Validation utilities for DataLoader parameters.

This module provides validation functions for DataLoader configuration
and parameter checking.
"""

from typing import Any

import torch


def validate_dataloader_params(
    batch_size: int, prefetch_factor: int, num_workers: int
) -> None:
    """
    Validate DataLoader parameters for correctness and compatibility.

    Args:
        batch_size: Number of samples per batch to load.
        prefetch_factor: Number of samples loaded in advance by each worker.
        num_workers: Number of worker processes for data loading.

    Raises:
        ValueError: If any parameter is invalid or incompatible.

    Examples:
        ```python
        # Valid parameters
        validate_dataloader_params(batch_size=32, prefetch_factor=2, num_workers=4)

        # Invalid parameters
        validate_dataloader_params(batch_size=0, prefetch_factor=2, num_workers=4)
        # Raises ValueError: batch_size must be positive
        ```

    Note:
        This function performs comprehensive validation including:
        - Parameter range checks
        - Compatibility checks between parameters
        - System capability validation
    """
    # Validate batch_size
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Validate prefetch_factor
    if prefetch_factor < 1:
        raise ValueError(
            f"prefetch_factor must be >= 1, got {prefetch_factor}"
        )

    # Validate num_workers
    if num_workers < -1:
        raise ValueError(f"num_workers must be >= -1, got {num_workers}")

    # Validate compatibility between num_workers and prefetch_factor
    if num_workers == 0 and prefetch_factor > 1:
        raise ValueError(
            f"prefetch_factor={prefetch_factor} is incompatible with "
            f"num_workers={num_workers}. Set prefetch_factor=1 for single-threaded loading."
        )

    # Validate system capabilities
    if num_workers > 0:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        if num_workers > cpu_count:
            raise ValueError(
                f"num_workers={num_workers} exceeds available CPU cores={cpu_count}"
            )

    # Validate CUDA availability for GPU-specific features
    if not torch.cuda.is_available():
        # Warn about potential performance issues
        if num_workers > 0:
            import warnings

            warnings.warn(
                "CUDA not available. Consider setting num_workers=0 for "
                "better performance on CPU-only systems.",
                stacklevel=2,
            )


def validate_sampler_config(sampler_config: dict[str, Any] | None) -> None:
    """
    Validate sampler configuration dictionary.

    Args:
        sampler_config: Configuration dictionary for custom samplers.

    Raises:
        ValueError: If sampler configuration is invalid.

    Examples:
        ```python
        # Valid configuration
        config = {"kind": "random", "shuffle": True}
        validate_sampler_config(config)

        # Invalid configuration
        config = {"shuffle": True}  # Missing 'kind'
        validate_sampler_config(config)
        # Raises ValueError: sampler_config must contain 'kind' key
        ```
    """
    if sampler_config is None:
        return

    if not isinstance(sampler_config, dict):
        raise ValueError("sampler_config must be a dictionary")

    if "kind" not in sampler_config:
        raise ValueError("sampler_config must contain 'kind' key")

    kind = sampler_config["kind"]
    if not isinstance(kind, str):
        raise ValueError("sampler_config['kind'] must be a string")

    # Validate specific sampler types
    valid_kinds = {"random", "sequential", "distributed", "weighted"}
    if kind not in valid_kinds:
        raise ValueError(
            f"sampler_config['kind'] must be one of {valid_kinds}, got {kind}"
        )

    # Validate distributed sampler requirements
    if kind == "distributed":
        required_keys = {"rank", "world_size"}
        missing_keys = required_keys - sampler_config.keys()
        if missing_keys:
            raise ValueError(
                f"Distributed sampler requires keys: {missing_keys}"
            )

        rank = sampler_config.get("rank")
        world_size = sampler_config.get("world_size")

        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f"rank must be non-negative integer, got {rank}")

        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError(
                f"world_size must be positive integer, got {world_size}"
            )

        if rank >= world_size:
            raise ValueError(
                f"rank={rank} must be less than world_size={world_size}"
            )

    # Validate weighted sampler requirements
    if kind == "weighted":
        if "weights" not in sampler_config:
            raise ValueError("Weighted sampler requires 'weights' key")

        weights = sampler_config["weights"]
        if not isinstance(weights, list | tuple) or len(weights) == 0:
            raise ValueError("weights must be non-empty list or tuple")


def validate_memory_config(
    max_memory_mb: float | None, adaptive_batch_size: bool
) -> None:
    """
    Validate memory-related configuration parameters.

    Args:
        max_memory_mb: Maximum GPU memory to use for adaptive batch sizing.
        adaptive_batch_size: Whether to enable automatic batch size adaptation.

    Raises:
        ValueError: If memory configuration is invalid.

    Examples:
        ```python
        # Valid configuration
        validate_memory_config(max_memory_mb=8000, adaptive_batch_size=True)

        # Invalid configuration
        validate_memory_config(max_memory_mb=-1000, adaptive_batch_size=True)
        # Raises ValueError: max_memory_mb must be positive
        ```
    """
    if max_memory_mb is not None:
        if max_memory_mb <= 0:
            raise ValueError(
                f"max_memory_mb must be positive, got {max_memory_mb}"
            )

        if max_memory_mb > 100000:  # 100GB limit
            raise ValueError(
                f"max_memory_mb={max_memory_mb} seems unreasonably large. "
                "Please verify the value."
            )

    if adaptive_batch_size and max_memory_mb is None:
        import warnings

        warnings.warn(
            "adaptive_batch_size=True but max_memory_mb=None. "
            "Adaptive batch sizing will be disabled.",
            stacklevel=2,
        )
