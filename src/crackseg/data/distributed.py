import torch
from torch.utils.data import Dataset, DistributedSampler


def create_distributed_sampler(
    dataset: Dataset,
    num_replicas: int | None = None,
    rank: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
) -> DistributedSampler:
    """Create a distributed sampler for multi-GPU training.

    This function provides a simple interface for creating DistributedSampler
    instances with automatic rank and world size detection when possible.

    Args:
        dataset: Dataset to sample from
        num_replicas: Number of processes (world size). If None, auto-detected.
        rank: Rank of current process. If None, auto-detected.
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling
        drop_last: Whether to drop the last incomplete batch

    Returns:
        Configured DistributedSampler instance

    Example:
        >>> dataset = MyDataset()
        >>> sampler = create_distributed_sampler(
        ...     dataset=dataset,
        ...     shuffle=True,
        ...     drop_last=True
        ... )
        >>> dataloader = DataLoader(
        ...     dataset=dataset,
        ...     batch_size=32,
        ...     sampler=sampler
        ... )
    """
    # Auto-detect distributed training parameters if not provided
    if num_replicas is None:
        num_replicas = get_world_size()

    if rank is None:
        rank = get_rank()

    return DistributedSampler(
        dataset=dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def is_distributed_available_and_initialized():
    """Check if torch.distributed is available and initialized."""
    return (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def get_rank():
    """Get the current process rank in distributed mode, or 0 if not."""
    if is_distributed_available_and_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size():
    """Get the world size (number of processes) in distributed mode, or 1 if
    not.
    """
    if is_distributed_available_and_initialized():
        return torch.distributed.get_world_size()
    return 1


def sync_distributed():
    """Synchronize all processes (barrier) if in distributed mode."""
    if is_distributed_available_and_initialized():
        torch.distributed.barrier()
