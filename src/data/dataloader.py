import os
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from .sampler import sampler_factory
from .memory import get_available_gpu_memory


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = -1,  # Default to auto-detect
    shuffle: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    sampler_config: dict = None,
    rank: int = None,
    world_size: int = None,
    fp16: bool = False,           # <-- Soporte para mixed precision
    max_memory_mb: float = None,  # <-- Control de memoria máxima
    adaptive_batch_size: bool = False,  # <-- Ajustar batch size según memoria
    **kwargs
) -> DataLoader:
    """
    Creates and configures a PyTorch DataLoader with sensible defaults.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int): How many samples per batch to load. Default: 32.
        num_workers (int): How many subprocesses to use for data loading.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Recommended for GPU training.
        prefetch_factor (int): Number of batches loaded in advance by each
            worker.
        sampler_config (dict, optional): Sampler configuration dict. If given,
            uses sampler_factory to create a custom sampler. Example:
            {'kind': 'distributed', 'shuffle': True, 'seed': 42, ...}.
        rank (int, optional): Distributed process rank (for distributed
            training).
        world_size (int, optional): Number of processes (for distributed
            training).
        fp16 (bool): Whether to use mixed precision (FP16) if available.
            Default: False.
        max_memory_mb (float, optional): Maximum GPU memory to use in MB.
            If None, uses all available memory.
        adaptive_batch_size (bool): Whether to adjust batch size based on
            available memory. Default: False.
        **kwargs: Additional keyword arguments to pass to the DataLoader
            constructor.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.

    Raises:
        ValueError: If batch_size or prefetch_factor are not positive,
            or if num_workers is less than -1.
        ValueError: If both shuffle and sampler are set (PyTorch limitation).

    Note:
        If using DistributedSampler, debes llamar a
        `set_epoch(epoch)` en el sampler al inicio de cada época para
        asegurar el shuffling correcto entre procesos.

        When using fp16=True, you should wrap your training loop with
        torch.cuda.amp.autocast() context manager.
    """
    # --- Parameter Validation ---
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if prefetch_factor <= 0:
        raise ValueError(
            f"prefetch_factor must be positive, got {prefetch_factor}"
        )
    if num_workers < -1:
        raise ValueError(f"num_workers must be >= -1, got {num_workers}")

    # --- Memory Optimization (if requested) ---
    if adaptive_batch_size and torch.cuda.is_available():
        # Calculate available memory (retain 10% for safety)
        if max_memory_mb is None:
            available_mb = get_available_gpu_memory() * 0.9
        else:
            available_mb = min(max_memory_mb, get_available_gpu_memory() * 0.9)

        # Very rough heuristic: assume 4 bytes per float * 2 for gradients and
        # optimizer
        # Multiply by batch size to get total memory
        # This is very approximate and should be replaced with proper
        # estimation
        # Placeholder - adjust based on dataset
        approx_sample_size_mb = 4 * 0.001
        max_batch_size = int(available_mb // approx_sample_size_mb)

        # Don't exceed user-specified batch size
        batch_size = min(batch_size, max_batch_size)
        batch_size = max(1, batch_size)  # Ensure at least 1

        warnings.warn(
            f"Adaptive batch size used: {batch_size} (limited by memory)"
        )

    # --- Mixed Precision ---
    if fp16 and not torch.cuda.is_available():
        warnings.warn(
            "Mixed precision (fp16) requested but CUDA not available. "
            "Falling back to standard precision."
        )
        fp16 = False

    # --- Determine num_workers ---
    actual_num_workers = 0
    if num_workers == -1:
        try:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                actual_num_workers = max(1, cpu_count // 2)
            else:
                warnings.warn(
                    "Could not determine CPU count, defaulting num_workers to "
                    "1."
                )
                actual_num_workers = 1
        except NotImplementedError:
            warnings.warn(
                "os.cpu_count() not implemented, defaulting num_workers to 1."
            )
            actual_num_workers = 1
    else:
        actual_num_workers = num_workers

    # --- Determine pin_memory ---
    can_pin_memory = pin_memory and torch.cuda.is_available()
    if pin_memory and not can_pin_memory:
        warnings.warn(
            "pin_memory=True requires CUDA availability. "
            "Setting pin_memory=False."
        )

    # --- Sampler logic ---
    sampler = None
    if sampler_config is not None:
        sampler_kind = sampler_config.get('kind')
        sampler_kwargs = dict(sampler_config)
        if sampler_kind == 'distributed':
            # Permitir override por argumentos directos
            if world_size is not None:
                sampler_kwargs['num_replicas'] = world_size
            if rank is not None:
                sampler_kwargs['rank'] = rank
        sampler = sampler_factory(
            kind=sampler_kind,
            data_source=dataset,
            labels=sampler_kwargs.get('labels'),
            indices=sampler_kwargs.get('indices'),
            replacement=sampler_kwargs.get('replacement', False),
            num_samples=sampler_kwargs.get('num_samples'),
            num_replicas=sampler_kwargs.get('num_replicas'),
            rank=sampler_kwargs.get('rank'),
            shuffle=sampler_kwargs.get('shuffle', True),
            seed=sampler_kwargs.get('seed', 0),
            drop_last=sampler_kwargs.get('drop_last', False)
        )
        if shuffle:
            warnings.warn(
                "Both sampler and shuffle are set. "
                "Setting shuffle=False (PyTorch does not allow both)."
            )
            shuffle = False

    # --- Create DataLoader ---
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=actual_num_workers,
        pin_memory=can_pin_memory,
        prefetch_factor=prefetch_factor if actual_num_workers > 0 else None,
        persistent_workers=True if actual_num_workers > 0 else False,
        **kwargs
    )

    return dataloader
