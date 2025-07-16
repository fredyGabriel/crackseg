"""Random seed utilities for reproducibility."""

# pyright: reportUnknownMemberType=false
# Suppression for PyTorch manual_seed methods with incomplete type stubs

import random

import numpy as np
import torch

from crackseg.utils.logging import get_logger

logger = get_logger(__name__)


def set_random_seeds(
    seed: int | None = None, deterministic: bool = True
) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to use. If None, a random seed will be generated.
        deterministic: If True, sets CUDA to use deterministic algorithms.
            This may impact performance but ensures reproducibility.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using randomly generated seed: {seed}")
    else:
        logger.info(f"Setting random seed to: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("CUDA set to deterministic mode")
        else:
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA using benchmark mode for better performance")
