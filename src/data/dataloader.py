import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .memory import get_available_gpu_memory
from .sampler import SamplerFactoryArgs, sampler_factory


@dataclass
class DataLoaderConfig:
    """Configuration for creating a DataLoader."""

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


def _validate_dataloader_params(
    batch_size: int, prefetch_factor: int, num_workers: int
) -> None:
    """Validates basic DataLoader parameters."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if prefetch_factor <= 0:
        raise ValueError(
            f"prefetch_factor must be positive, got {prefetch_factor}"
        )
    if num_workers < -1:
        raise ValueError(f"num_workers must be >= -1, got {num_workers}")


def _calculate_adaptive_batch_size(
    current_batch_size: int,
    adaptive_enabled: bool,
    max_memory_limit_mb: float | None,
) -> int:
    """Calculates an adaptive batch size based on available GPU memory."""
    if not (adaptive_enabled and torch.cuda.is_available()):
        return current_batch_size

    # Calculate available memory (retain 10% for safety)
    current_available_gpu_mem = get_available_gpu_memory() * 0.9
    if max_memory_limit_mb is None:
        available_mb = current_available_gpu_mem
    else:
        available_mb = min(max_memory_limit_mb, current_available_gpu_mem)

    # Very rough heuristic for sample size in MB.
    # TODO: This should be made more sophisticated or configurable.
    approx_sample_size_mb = 4 * 0.001  # Example: 4MB, needs actual estimation
    if approx_sample_size_mb <= 0:
        warnings.warn(
            "Approximate sample size for adaptive batch is not positive. "
            "Adaptive batch sizing might not work as expected.",
            stacklevel=2,
        )
        return current_batch_size

    max_possible_batch_size = int(available_mb // approx_sample_size_mb)

    # Adjust batch size: take the minimum of current, and max possible based
    # on memory.
    new_batch_size = min(current_batch_size, max_possible_batch_size)
    new_batch_size = max(1, new_batch_size)  # Ensure at least 1

    if new_batch_size != current_batch_size:
        warnings.warn(
            f"Adaptive batch size adjusted to: {new_batch_size} (from "
            f"{current_batch_size}, limited by memory)",
            stacklevel=2,
        )
    return new_batch_size


def _configure_num_workers(requested_num_workers: int) -> int:
    """Determines the actual number of workers to use."""
    if requested_num_workers == -1:  # Auto-detect
        try:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                # Using half of the CPU cores is a common heuristic
                return max(1, cpu_count // 2)
            else:
                warnings.warn(
                    "Could not determine CPU count via os.cpu_count(), "
                    "defaulting num_workers to 1.",
                    stacklevel=2,
                )
                return 1
        except NotImplementedError:
            warnings.warn(
                "os.cpu_count() not implemented on this system, "
                "defaulting num_workers to 1.",
                stacklevel=2,
            )
            return 1
    return requested_num_workers


def _create_sampler_from_config(
    sampler_config_param: dict[str, Any] | None,
    dataset_param: Dataset[Any],
    world_size_param: int | None,
    rank_param: int | None,
    shuffle_param: bool,
) -> tuple[Sampler[Any] | None, bool]:
    """Creates a sampler from configuration and updates shuffle flag."""
    sampler_instance: Sampler[Any] | None = None
    current_shuffle_status = shuffle_param

    if sampler_config_param:
        sampler_kind_str: str | None = sampler_config_param.get("kind")
        if sampler_kind_str:
            # Prepare args for SamplerFactoryArgs
            sfa_kwargs = sampler_config_param.copy()
            sfa_kwargs.pop(
                "kind", None
            )  # Remove kind as it's not in SamplerFactoryArgs

            if sampler_kind_str == "distributed":
                if world_size_param is not None:
                    sfa_kwargs["num_replicas"] = world_size_param
                if rank_param is not None:
                    sfa_kwargs["rank"] = rank_param
                # shuffle for distributed sampler is handled by
                # SamplerFactoryArgs default or sfa_kwargs

            # Ensure only valid keys for SamplerFactoryArgs are passed
            valid_sfa_keys = SamplerFactoryArgs.__annotations__.keys()
            filtered_sfa_kwargs = {
                k: v for k, v in sfa_kwargs.items() if k in valid_sfa_keys
            }

            sampler_args_obj = SamplerFactoryArgs(**filtered_sfa_kwargs)

            sampler_instance = sampler_factory(
                kind=sampler_kind_str,
                data_source=dataset_param,
                args=sampler_args_obj,
            )
            if current_shuffle_status:
                warnings.warn(
                    "Both sampler and shuffle are set. "
                    "Setting shuffle=False (PyTorch does not allow both).",
                    stacklevel=2,
                )
                current_shuffle_status = False
        else:
            warnings.warn(
                "Sampler configuration provided but 'kind' is missing or "
                f"None: {sampler_config_param}",
                stacklevel=2,
            )
    return sampler_instance, current_shuffle_status


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int = 32,
    config: DataLoaderConfig | None = None,
) -> DataLoader[Any]:
    """
    Creates and configures a PyTorch DataLoader with sensible defaults.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int): How many samples per batch to load. Default: 32.
        config (DataLoaderConfig): Configuration object for the dataloader.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.

    Raises:
        ValueError: If batch_size or prefetch_factor are not positive,
            or if num_workers is less than -1.
        ValueError: If both shuffle and sampler are set (PyTorch limitation).

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
    _validate_dataloader_params(
        batch_size, config.prefetch_factor, config.num_workers
    )

    # --- Memory Optimization (if requested) ---
    actual_batch_size = _calculate_adaptive_batch_size(
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
    actual_num_workers = _configure_num_workers(config.num_workers)

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
        **dataloader_kwargs,
    )

    return dataloader_instance
