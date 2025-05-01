"""Utility functions for device management."""

import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate torch device.

    Args:
        device_str: Device specification ('cuda', 'cpu', or 'auto').
                   If 'auto', will use CUDA if available.

    Returns:
        torch.device: The selected device.
    """
    if device_str == "auto":
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_str)
