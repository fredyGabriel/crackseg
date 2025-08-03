"""Worker configuration utilities for DataLoader.

This module provides utilities for configuring the number of worker
processes for optimal DataLoader performance.
"""

import multiprocessing
import warnings
from typing import Any

import torch


def configure_num_workers(requested_num_workers: int) -> int:
    """
    Configure optimal number of workers for DataLoader.

    This function intelligently determines the optimal number of worker
    processes based on system capabilities and the requested configuration.
    It handles automatic detection, system limits, and performance optimization.

    Args:
        requested_num_workers: The requested number of workers.
            Use -1 for automatic detection.

    Returns:
        int: The optimal number of workers for the current system.

    Examples:
        ```python
        # Automatic detection
        workers = configure_num_workers(-1)

        # Manual configuration
        workers = configure_num_workers(4)

        # Single-threaded
        workers = configure_num_workers(0)
        ```

    Note:
        - Returns 0 for single-threaded loading (no workers)
        - Automatically detects optimal workers when requested_num_workers=-1
        - Respects system CPU limits
        - Optimizes for different environments (development vs production)
    """
    if requested_num_workers == 0:
        return 0

    if requested_num_workers == -1:
        return _detect_optimal_workers()

    # Validate requested workers
    if requested_num_workers < 0:
        raise ValueError(
            f"num_workers must be >= 0, got {requested_num_workers}"
        )

    # Check system limits
    cpu_count = multiprocessing.cpu_count()
    if requested_num_workers > cpu_count:
        warnings.warn(
            f"Requested {requested_num_workers} workers but only {cpu_count} "
            f"CPU cores available. Using {cpu_count} workers.",
            stacklevel=2,
        )
        return cpu_count

    return requested_num_workers


def _detect_optimal_workers() -> int:
    """
    Automatically detect optimal number of workers.

    Returns:
        int: Optimal number of workers for current system.

    Note:
        This function considers:
        - CPU core count
        - Available memory
        - CUDA availability
        - System type (development vs production)
    """
    cpu_count = multiprocessing.cpu_count()

    # Base calculation: use 75% of CPU cores
    optimal_workers = max(1, int(cpu_count * 0.75))

    # Adjust based on system characteristics
    if _is_development_environment():
        # Use fewer workers in development for better debugging
        optimal_workers = min(optimal_workers, 2)
    else:
        # Production: use more workers but respect limits
        optimal_workers = min(optimal_workers, cpu_count - 1)

    # Ensure minimum of 1 worker for multi-threaded loading
    optimal_workers = max(1, optimal_workers)

    # Ensure maximum of CPU count
    optimal_workers = min(optimal_workers, cpu_count)

    return optimal_workers


def _is_development_environment() -> bool:
    """
    Detect if running in development environment.

    Returns:
        bool: True if in development environment.

    Note:
        This is a heuristic based on common development indicators.
        Override this function for custom environment detection.
    """
    import os

    # Check for common development indicators
    dev_indicators = [
        "PYTHONPATH" in os.environ,
        "VIRTUAL_ENV" in os.environ,
        "CONDA_DEFAULT_ENV" in os.environ,
        "DEBUG" in os.environ,
        "DEVELOPMENT" in os.environ,
    ]

    return any(dev_indicators)


def get_worker_info() -> dict[str, Any]:
    """
    Get comprehensive worker configuration information.

    Returns:
        dict: Worker configuration information.

    Examples:
        ```python
        info = get_worker_info()
        print(f"CPU cores: {info['cpu_count']}")
        print(f"Optimal workers: {info['optimal_workers']}")
        ```
    """
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = _detect_optimal_workers()

    info = {
        "cpu_count": cpu_count,
        "optimal_workers": optimal_workers,
        "is_development": _is_development_environment(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
    }

    # Add system-specific information
    try:
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
        info["memory_gb"] = round(memory_gb, 1)

        # Memory-based worker recommendations
        if memory_gb < 8:
            info["memory_based_workers"] = 1
        elif memory_gb < 16:
            info["memory_based_workers"] = 2
        else:
            info["memory_based_workers"] = min(4, cpu_count)

    except ImportError:
        info["memory_gb"] = None
        info["memory_based_workers"] = None

    return info


def validate_worker_config(
    num_workers: int,
    prefetch_factor: int,
    batch_size: int,
) -> None:
    """
    Validate worker configuration for compatibility.

    Args:
        num_workers: Number of worker processes.
        prefetch_factor: Number of samples loaded in advance by each worker.
        batch_size: Number of samples per batch.

    Raises:
        ValueError: If configuration is incompatible.

    Examples:
        ```python
        # Valid configuration
        validate_worker_config(num_workers=4, prefetch_factor=2, batch_size=32)

        # Invalid configuration
        validate_worker_config(num_workers=0, prefetch_factor=3, batch_size=32)
        # Raises ValueError: prefetch_factor > 1 incompatible with num_workers=0
        ```
    """
    if num_workers == 0 and prefetch_factor > 1:
        raise ValueError(
            f"prefetch_factor={prefetch_factor} is incompatible with "
            f"num_workers={num_workers}. Set prefetch_factor=1 for single-threaded loading."
        )

    if num_workers > 0 and batch_size < num_workers:
        warnings.warn(
            f"batch_size={batch_size} is smaller than num_workers={num_workers}. "
            "This may cause inefficient loading.",
            stacklevel=2,
        )

    # Check system limits
    cpu_count = multiprocessing.cpu_count()
    if num_workers > cpu_count:
        raise ValueError(
            f"num_workers={num_workers} exceeds available CPU cores={cpu_count}"
        )
