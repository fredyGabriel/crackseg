import gc
import math
import warnings
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler


def get_available_gpu_memory(
    device: torch.device | str | int | None = None, in_mb: bool = True
) -> float:
    """
    Get available memory on specified CUDA device.

    Args:
        device: CUDA device (None for current device)
        in_mb: Return value in MB if True, bytes if False

    Returns:
        Available memory in MB or bytes
    """
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA not available, returning 0 for available memory",
            stacklevel=2,
        )
        return 0.0

    device = device or torch.cuda.current_device()

    torch.cuda.empty_cache()
    gc.collect()

    memory_reserved = torch.cuda.memory_reserved(device)
    memory_allocated = torch.cuda.memory_allocated(device)
    max_memory = cast(
        int,
        torch.cuda.get_device_properties(device).total_memory,
    )

    available = (
        max_memory - memory_reserved + (memory_reserved - memory_allocated)
    )

    if in_mb:
        return available / (1024 * 1024)
    return float(available)


def get_gpu_memory_usage(
    device: torch.device | str | int | None = None, in_mb: bool = True
) -> dict[str, float]:
    """
    Get detailed GPU memory usage.

    Args:
        device: CUDA device (None for current device)
        in_mb: Return values in MB if True, bytes if False

    Returns:
        Dict with 'allocated', 'reserved', 'total', and 'available' memory
    """
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA not available, returning zeros for memory usage",
            stacklevel=2,
        )
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "total": 0.0,
            "available": 0.0,
        }

    device = device or torch.cuda.current_device()

    memory_allocated = torch.cuda.memory_allocated(device)
    memory_reserved = torch.cuda.memory_reserved(device)
    max_memory = cast(
        int,
        torch.cuda.get_device_properties(device).total_memory,
    )
    available = (
        max_memory - memory_reserved + (memory_reserved - memory_allocated)
    )

    if in_mb:
        div = 1024 * 1024
        return {
            "allocated": memory_allocated / div,
            "reserved": memory_reserved / div,
            "total": max_memory / div,
            "available": available / div,
        }

    return {
        "allocated": float(memory_allocated),
        "reserved": float(memory_reserved),
        "total": float(max_memory),
        "available": float(available),
    }


@dataclass
class BatchSizeEstimationArgs:
    """Arguments for estimating maximum batch size."""

    model: nn.Module
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...] | None = None
    max_memory_mb: float | None = None
    start_batch_size: int = 32
    min_batch_size: int = 1
    safety_factor: float = 0.8
    fp16: bool = False
    device: torch.device | str | int | None = None


@dataclass
class _AttemptBatchSizeConfig:
    """Configuration for a single batch size attempt and memory check."""

    model: torch.nn.Module
    current_bs: int
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...] | None
    fp16: bool
    device: torch.device
    max_allowable_mem_mb: float


def _attempt_batch_size_memory_check(config: _AttemptBatchSizeConfig) -> bool:
    """
    Attempts a forward pass with the given batch size and checks memory.
    Returns True if successful (fits in memory), False otherwise (OOM or too
    high).
    """
    try:
        torch.cuda.empty_cache()
        gc.collect()

        dummy_input = torch.rand(
            (config.current_bs,) + config.input_shape, device=config.device
        )
        if config.fp16:
            dummy_input = dummy_input.half()
        # Model is assumed to be already .half() if fp16, done outside this
        # helper

        if config.target_shape is not None:
            dummy_target = torch.rand(
                (config.current_bs,) + config.target_shape,
                device=config.device,
            )
            if config.fp16:
                dummy_target = dummy_target.half()

        with torch.no_grad():
            _ = config.model(dummy_input)  # Perform forward pass

        current_mem_mb = get_gpu_memory_usage(config.device)["allocated"]
        return current_mem_mb < config.max_allowable_mem_mb
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False  # OOM is a failure for this batch_size
        raise  # Different runtime error, propagate it


def estimate_batch_size(args: BatchSizeEstimationArgs) -> int:
    """
    Estimate maximum batch size that fits in memory.

    Args:
        args: Configuration object for batch size estimation.

    Returns:
        Estimated maximum batch size
    """
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA not available, returning default batch size",
            stacklevel=2,
        )
        return min(args.start_batch_size, 8)  # Conservative default

    # Determine target device and move model
    torch_device_name = args.device or torch.cuda.current_device()
    torch_device = torch.device(f"cuda:{torch_device_name}")
    model_to_device = args.model.to(torch_device)

    # Convert model to FP16 once if specified
    if args.fp16:
        model_to_device = model_to_device.half()

    # Calculate maximum memory to use with safety factor
    current_max_memory_mb = args.max_memory_mb
    if current_max_memory_mb is None:
        current_max_memory_mb = (
            get_available_gpu_memory(torch_device)
            * args.safety_factor  # Use torch_device here
        )
    else:
        current_max_memory_mb *= args.safety_factor

    batch_size = args.start_batch_size

    while batch_size >= args.min_batch_size:
        attempt_config = _AttemptBatchSizeConfig(
            model=model_to_device,
            current_bs=batch_size,
            input_shape=args.input_shape,
            target_shape=args.target_shape,
            fp16=args.fp16,
            device=torch_device,  # Pass the torch.device object
            max_allowable_mem_mb=current_max_memory_mb,
        )
        if _attempt_batch_size_memory_check(attempt_config):
            return batch_size  # Success!

        # Attempt failed (OOM or memory usage too high)
        if batch_size == args.min_batch_size:
            break  # Current batch_size is min_batch_size, and it failed

        # Calculate next smaller batch size to try
        halved_bs = batch_size // 2
        batch_size = max(halved_bs, args.min_batch_size)

    # Fallback to minimum if loop completes (e.g., min_batch_size also failed)
    return args.min_batch_size


def calculate_gradient_accumulation_steps(
    target_batch_size: int, actual_batch_size: int
) -> int:
    """
    Calculate gradient accumulation steps to reach target effective batch size.

    Args:
        target_batch_size: Desired effective batch size
        actual_batch_size: Actual batch size that fits in memory

    Returns:
        Number of gradient accumulation steps
    """
    if actual_batch_size >= target_batch_size:
        return 1

    return math.ceil(target_batch_size / actual_batch_size)


def enable_mixed_precision() -> GradScaler | None:
    """
    Enable mixed precision training.

    Returns:
        GradScaler for mixed precision training
    """
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA not available, mixed precision not enabled",
            stacklevel=2,
        )
        return None

    return GradScaler("cuda")


def format_memory_stats(stats: dict[str, float]) -> str:
    """
    Format memory statistics into a readable string.

    Args:
        stats: Memory statistics from get_gpu_memory_usage()

    Returns:
        Formatted string with memory usage
    """
    return (
        f"GPU Memory: {stats['allocated']:.1f}/{stats['total']:.1f} MB "
        f"({stats['allocated'] / stats['total'] * 100:.1f}%) | "
        f"Available: {stats['available']:.1f} MB"
    )


def memory_summary(model: nn.Module | None = None) -> str:
    """
    Create a summary of memory usage, optionally including model details.

    Args:
        model: Model to include in summary (optional)

    Returns:
        String with memory usage summary
    """
    if not torch.cuda.is_available():
        return "CUDA not available, no memory statistics."

    stats = get_gpu_memory_usage()
    summary = [format_memory_stats(stats)]

    if model is not None:
        # Estimate model size
        params_size = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        model_size_mb = params_size / (1024 * 1024)
        summary.append(f"Model size: {model_size_mb:.2f} MB")

        # Parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        summary.append(
            f"Parameters: {total_params:,} total, {trainable:,} trainable"
        )

    return " | ".join(summary)
