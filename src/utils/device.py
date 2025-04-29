import torch


def get_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
