"""Environment setup for training (consolidated)."""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig

from crackseg.utils import get_device, set_random_seeds

log = logging.getLogger(__name__)


def setup_environment(cfg: DictConfig) -> torch.device:
    log.info("Setting up training environment...")
    random_seed = cfg.get("random_seed", 42)
    set_random_seeds(random_seed)
    log.info("Random seed set to: %s", random_seed)
    device = get_device()
    log.info("Training device: %s", device)
    return device
