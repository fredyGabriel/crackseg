import torch


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
        torch.distributed.barrier()  # type: ignore
