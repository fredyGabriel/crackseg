"""Memory management utilities for crack segmentation data.

This module provides utilities for GPU memory optimization, batch size
estimation, and memory usage monitoring within the crack segmentation
pipeline.
"""

from .memory import (
    BatchSizeEstimationArgs,
    calculate_gradient_accumulation_steps,
    enable_mixed_precision,
    estimate_batch_size,
    format_memory_stats,
    get_available_gpu_memory,
    get_gpu_memory_usage,
    memory_summary,
    optimize_batch_size,
)

__all__ = [
    "optimize_batch_size",
    "get_available_gpu_memory",
    "get_gpu_memory_usage",
    "BatchSizeEstimationArgs",
    "estimate_batch_size",
    "calculate_gradient_accumulation_steps",
    "enable_mixed_precision",
    "format_memory_stats",
    "memory_summary",
]
