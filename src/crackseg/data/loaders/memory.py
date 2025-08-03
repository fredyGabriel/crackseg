"""Memory optimization utilities for DataLoader.

This module provides memory management and adaptive batch sizing
capabilities for optimizing DataLoader performance.
"""

import warnings
from typing import Any

import torch

from ..memory import get_available_gpu_memory


def calculate_adaptive_batch_size(
    current_batch_size: int,
    adaptive_enabled: bool,
    max_memory_limit_mb: float | None,
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    This function implements intelligent batch size adaptation to prevent
    out-of-memory (OOM) errors while maximizing training throughput.
    It analyzes available GPU memory and adjusts batch size accordingly.

    Args:
        current_batch_size: The requested batch size.
        adaptive_enabled: Whether to enable adaptive batch sizing.
        max_memory_limit_mb: Maximum GPU memory to use (in MB).

    Returns:
        int: The adjusted batch size, optimized for available memory.

    Examples:
        ```python
        # Basic adaptive sizing
        batch_size = calculate_adaptive_batch_size(
            current_batch_size=64,
            adaptive_enabled=True,
            max_memory_limit_mb=8000
        )

        # Disabled adaptive sizing
        batch_size = calculate_adaptive_batch_size(
            current_batch_size=32,
            adaptive_enabled=False,
            max_memory_limit_mb=None
        )
        ```

    Note:
        - If adaptive_enabled=False, returns current_batch_size unchanged
        - If CUDA not available, returns current_batch_size with warning
        - Memory estimation is conservative to prevent OOM errors
        - Batch size is never increased beyond current_batch_size
    """
    if not adaptive_enabled:
        return current_batch_size

    if not torch.cuda.is_available():
        warnings.warn(
            "Adaptive batch sizing requested but CUDA not available. "
            "Returning original batch size.",
            stacklevel=2,
        )
        return current_batch_size

    if max_memory_limit_mb is None:
        warnings.warn(
            "Adaptive batch sizing enabled but max_memory_limit_mb=None. "
            "Returning original batch size.",
            stacklevel=2,
        )
        return current_batch_size

    try:
        # Get available GPU memory
        available_memory_mb = get_available_gpu_memory()

        # Calculate safe memory limit (use 80% of available or max limit)
        safe_memory_mb = min(available_memory_mb * 0.8, max_memory_limit_mb)

        # Estimate memory per sample (conservative estimate)
        # This is a rough estimate - actual usage depends on model and data
        estimated_memory_per_sample_mb = _estimate_memory_per_sample()

        # Calculate maximum safe batch size
        max_safe_batch_size = int(
            safe_memory_mb / estimated_memory_per_sample_mb
        )

        # Ensure batch size doesn't exceed original request
        optimal_batch_size = min(max_safe_batch_size, current_batch_size)

        # Ensure minimum batch size for training stability
        optimal_batch_size = max(optimal_batch_size, 1)

        if optimal_batch_size < current_batch_size:
            warnings.warn(
                f"Reduced batch size from {current_batch_size} to {optimal_batch_size} "
                f"due to memory constraints (available: {available_memory_mb:.1f}MB, "
                f"limit: {max_memory_limit_mb}MB)",
                stacklevel=2,
            )

        return optimal_batch_size

    except Exception as e:
        warnings.warn(
            f"Failed to calculate adaptive batch size: {e}. "
            "Using original batch size.",
            stacklevel=2,
        )
        return current_batch_size


def _estimate_memory_per_sample() -> float:
    """
    Estimate memory usage per sample in MB.

    This is a conservative estimate based on typical crack segmentation
    image sizes and model memory requirements. The actual memory usage
    depends on the specific model architecture and data preprocessing.

    Returns:
        float: Estimated memory per sample in MB.

    Note:
        This is a rough estimate. For more accurate memory profiling,
        consider using torch.cuda.memory_profiler or similar tools.
    """
    # Conservative estimate for crack segmentation:
    # - Image size: 512x512 (typical for crack detection)
    # - 3 channels (RGB)
    # - Float32 precision
    # - Additional overhead for model gradients and activations

    # Base memory for image data (512x512x3x4 bytes)
    image_memory_mb = (512 * 512 * 3 * 4) / (1024 * 1024)  # ~3MB

    # Additional overhead for model activations, gradients, etc.
    # Conservative estimate: 5x base memory
    overhead_multiplier = 5.0

    return image_memory_mb * overhead_multiplier


def configure_memory_settings(
    config: Any,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Configure memory-related settings for DataLoader.

    Args:
        config: DataLoader configuration object.
        device: Target device for memory optimization.

    Returns:
        dict: Memory-optimized configuration settings.

    Examples:
        ```python
        from crackseg.data.loaders.config import DataLoaderConfig

        config = DataLoaderConfig(
            adaptive_batch_size=True,
            max_memory_mb=8000
        )

        memory_settings = configure_memory_settings(config, torch.device('cuda'))
        ```
    """
    settings = {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure pin_memory based on device
    if device is not None and device.type == "cuda":
        settings["pin_memory"] = getattr(config, "pin_memory", True)
    else:
        settings["pin_memory"] = False

    # Configure prefetch_factor based on memory availability
    if device is not None and device.type == "cuda":
        try:
            available_memory_mb = get_available_gpu_memory()
            # Use higher prefetch for systems with more memory
            if available_memory_mb > 8000:  # 8GB
                settings["prefetch_factor"] = 3
            elif available_memory_mb > 4000:  # 4GB
                settings["prefetch_factor"] = 2
            else:
                settings["prefetch_factor"] = 1
        except Exception:
            settings["prefetch_factor"] = 2  # Default fallback
    else:
        settings["prefetch_factor"] = 1  # Lower for CPU

    return settings


def get_memory_info() -> dict[str, Any]:
    """
    Get comprehensive memory information for debugging.

    Returns:
        dict: Memory information including available, total, and usage.

    Examples:
        ```python
        memory_info = get_memory_info()
        print(f"Available: {memory_info['available_mb']:.1f}MB")
        print(f"Total: {memory_info['total_mb']:.1f}MB")
        ```
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
    }

    if torch.cuda.is_available():
        try:
            available_mb = get_available_gpu_memory()
            total_mb = torch.cuda.get_device_properties(0).total_memory / (
                1024 * 1024
            )

            info.update(
                {
                    "available_mb": available_mb,
                    "total_mb": total_mb,
                    "used_mb": total_mb - available_mb,
                    "usage_percent": ((total_mb - available_mb) / total_mb)
                    * 100,
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                }
            )
        except Exception as e:
            info["error"] = str(e)
    else:
        info.update(
            {
                "available_mb": 0,
                "total_mb": 0,
                "used_mb": 0,
                "usage_percent": 0,
                "current_device": None,
                "device_name": "CPU",
            }
        )

    return info
