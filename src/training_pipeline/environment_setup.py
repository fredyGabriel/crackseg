"""Environment setup for training pipeline.

This module provides environment configuration functions for the training
pipeline including device selection, random seed initialization, and
CUDA availability validation.
"""

import logging

import torch
from omegaconf import DictConfig

from crackseg.utils import get_device, set_random_seeds

# Configure standard logger
log = logging.getLogger(__name__)


def setup_environment(cfg: DictConfig) -> torch.device:
    """
    Set up the training environment with proper device selection and random
    seeds.

    This function configures the global training environment including:
    - Random seed initialization for reproducibility
    - CUDA availability validation
    - Device selection and configuration

    Args:
        cfg: Hydra configuration containing environment settings.
            Expected keys:
            - random_seed (int, optional): Random seed for reproducibility.
              Defaults to 42.
            - require_cuda (bool, optional): Whether CUDA is required.
              Defaults to True.

    Returns:
        torch.device: The selected device for training (e.g., 'cuda:0' or
        'cpu').

    Raises:
        ResourceError: If CUDA is required but not available on the system.

    Examples:
        ```python
        cfg = OmegaConf.create({"random_seed": 42, "require_cuda": True})
        device = setup_environment(cfg)
        print(f"Training on device: {device}")
        ```

    Note:
        The function automatically detects CUDA availability and falls back
        to CPU if CUDA is not available and not required.
    """
    log.info("Setting up training environment...")

    # Set random seeds for reproducibility
    random_seed = cfg.get("random_seed", 42)
    set_random_seeds(random_seed)
    log.info("Random seed set to: %s", random_seed)

    # Get device for training
    device = get_device()
    log.info("Training device: %s", device)

    return device
