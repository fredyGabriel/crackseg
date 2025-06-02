"""Device utilities for the crack segmentation project."""

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_device(device_str: str | None = None) -> torch.device:
    """Get the appropriate device for model training.

    Args:
        device_str: Optional device string ('cpu', 'cuda', 'cuda:0', 'cuda:1',
                    'auto', etc.) or None. If None, 'auto', or 'cuda', uses
                    the first available GPU or falls back to CPU.

    Returns:
        torch.device: The selected device.
    """
    if device_str == "cpu":
        logger.info("Explicitly requested CPU.")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        return torch.device("cpu")

    # If None, 'auto', or 'cuda', default to cuda:0
    if device_str is None or device_str in ["auto", "cuda"]:
        device_id = 0
        device = torch.device(f"cuda:{device_id}")
        logger.info("CUDA available, using default GPU 0.")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        return device

    # If specific cuda device requested (e.g., 'cuda:1')
    if device_str.startswith("cuda:"):
        try:
            device_id = int(device_str.split(":")[1])
            if device_id >= torch.cuda.device_count():
                logger.warning(
                    f"GPU {device_id} requested but not available "
                    + f"(count: {torch.cuda.device_count()}), "
                    + "falling back to GPU 0."
                )
                device_id = 0
            device = torch.device(f"cuda:{device_id}")
            logger.info(
                f"Using specified GPU: {torch.cuda.get_device_name(device_id)}"
            )
            return device
        except (IndexError, ValueError):
            logger.warning(
                f"Invalid CUDA device string '{device_str}'. "
                + "Falling back to GPU 0."
            )
            device_id = 0
            device = torch.device(f"cuda:{device_id}")
            logger.info(
                f"Using fallback GPU: {torch.cuda.get_device_name(device_id)}"
            )
            return device

    # Fallback for unexpected device_str format
    logger.warning(f"Unexpected device string: '{device_str}'. Using CPU.")
    return torch.device("cpu")
