import torch
import gc
import warnings
import math
from typing import Tuple, Optional, Dict


def get_available_gpu_memory(device=None, in_mb=True) -> float:
    """
    Get available memory on specified CUDA device.

    Args:
        device: CUDA device (None for current device)
        in_mb: Return value in MB if True, bytes if False

    Returns:
        Available memory in MB or bytes
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, returning 0 for available memory")
        return 0.0

    device = device or torch.cuda.current_device()

    torch.cuda.empty_cache()
    gc.collect()

    memory_reserved = torch.cuda.memory_reserved(device)
    memory_allocated = torch.cuda.memory_allocated(device)
    max_memory = torch.cuda.get_device_properties(device).total_memory

    available = max_memory - memory_reserved + \
        (memory_reserved - memory_allocated)

    if in_mb:
        return available / (1024 * 1024)
    return float(available)


def get_gpu_memory_usage(device=None, in_mb=True) -> Dict[str, float]:
    """
    Get detailed GPU memory usage.

    Args:
        device: CUDA device (None for current device)
        in_mb: Return values in MB if True, bytes if False

    Returns:
        Dict with 'allocated', 'reserved', 'total', and 'available' memory
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, returning zeros for memory usage")
        return {
            'allocated': 0.0,
            'reserved': 0.0,
            'total': 0.0,
            'available': 0.0
        }

    device = device or torch.cuda.current_device()

    memory_allocated = torch.cuda.memory_allocated(device)
    memory_reserved = torch.cuda.memory_reserved(device)
    max_memory = torch.cuda.get_device_properties(device).total_memory
    available = max_memory - memory_reserved + \
        (memory_reserved - memory_allocated)

    if in_mb:
        div = (1024 * 1024)
        return {
            'allocated': memory_allocated / div,
            'reserved': memory_reserved / div,
            'total': max_memory / div,
            'available': available / div
        }

    return {
        'allocated': float(memory_allocated),
        'reserved': float(memory_reserved),
        'total': float(max_memory),
        'available': float(available)
    }


def estimate_batch_size(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    target_shape: Optional[Tuple[int, ...]] = None,
    max_memory_mb: Optional[float] = None,
    start_batch_size: int = 32,
    min_batch_size: int = 1,
    safety_factor: float = 0.8,
    fp16: bool = False,
    device=None
) -> int:
    """
    Estimate maximum batch size that fits in memory.

    Args:
        model: PyTorch model to use for estimation
        input_shape: Shape of a single input (excluding batch dimension)
        target_shape: Shape of a single target (excluding batch dimension)
        max_memory_mb: Maximum memory to use (MB). If None, uses available.
        start_batch_size: Starting batch size for estimation
        min_batch_size: Minimum acceptable batch size
        safety_factor: Memory safety factor (0.0-1.0)
        fp16: Whether to use half precision (FP16)
        device: Device to use for estimation

    Returns:
        Estimated maximum batch size
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, returning default batch size")
        return min(start_batch_size, 8)  # Conservative default

    device = device or torch.cuda.current_device()
    torch_device = torch.device(f'cuda:{device}')
    model = model.to(torch_device)

    # If max_memory not specified, use available memory
    if max_memory_mb is None:
        max_memory_mb = get_available_gpu_memory(device) * safety_factor
    else:
        max_memory_mb *= safety_factor

    batch_size = start_batch_size
    while batch_size >= min_batch_size:
        try:
            # Clear memory first
            torch.cuda.empty_cache()
            gc.collect()

            # Create dummy input and target tensors
            dummy_input = torch.rand(
                (batch_size,) + input_shape, device=torch_device
            )
            if fp16:
                dummy_input = dummy_input.half()
                model = model.half()  # Convert model to half precision

            if target_shape is not None:
                dummy_target = torch.rand(
                    (batch_size,) + target_shape, device=torch_device
                )
                if fp16:
                    dummy_target = dummy_target.half()

            # Forward pass to trigger memory allocation
            with torch.no_grad():
                _ = model(dummy_input)  # Discard output, just check memory

            # Check memory usage
            current_mem_mb = get_gpu_memory_usage(device)['allocated']
            if current_mem_mb < max_memory_mb:
                # Found suitable batch size
                return batch_size

            # Try smaller batch size
            batch_size //= 2

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Memory error, reduce batch size
                batch_size //= 2
            else:
                # Unexpected error
                raise e

    # Fallback to minimum
    return min_batch_size


def calculate_gradient_accumulation_steps(
    target_batch_size: int,
    actual_batch_size: int
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


def enable_mixed_precision() -> torch.amp.GradScaler:
    """
    Enable mixed precision training.

    Returns:
        GradScaler for mixed precision training
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, mixed precision not enabled")
        return None

    return torch.amp.GradScaler('cuda')


def format_memory_stats(stats: Dict[str, float]) -> str:
    """
    Format memory statistics into a readable string.

    Args:
        stats: Memory statistics from get_gpu_memory_usage()

    Returns:
        Formatted string with memory usage
    """
    return (
        f"GPU Memory: {stats['allocated']:.1f}/{stats['total']:.1f} MB "
        f"({stats['allocated']/stats['total']*100:.1f}%) | "
        f"Available: {stats['available']:.1f} MB"
    )


def memory_summary(model: Optional[torch.nn.Module] = None) -> str:
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
