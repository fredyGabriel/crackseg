"""Model creation for training (consolidated)."""

from __future__ import annotations

import logging
from typing import Any, cast

import hydra
import torch
from hydra import errors as hydra_errors
from omegaconf import DictConfig
from torch.nn import Module

from crackseg.utils import ModelError

log = logging.getLogger(__name__)


def create_model(cfg: DictConfig, device: Any) -> torch.nn.Module:
    log.info("Creating model...")
    try:
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model = cast(Module, model)
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        log.info(
            "Created %s model with %s parameters",
            type(model).__name__,
            num_params,
        )
        assert isinstance(model, Module)
        return model
    except (
        ModelError,
        hydra_errors.InstantiationException,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as e:
        log.error("Error creating model: %s", str(e))
        raise ModelError(f"Error creating model: {str(e)}") from e
