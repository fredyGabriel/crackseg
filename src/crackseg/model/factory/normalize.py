"""Normalization utilities for model factory configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from .factory_utils import hydra_to_dict


def normalize_config(
    config: Mapping[str, Any] | DictConfig,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a configuration dictionary for consistency."""
    if isinstance(config, DictConfig):
        config_dict: dict[str, Any] = hydra_to_dict(config)
    else:
        config_dict = dict(config)
    if defaults:
        for key, value in defaults.items():
            if key not in config_dict:
                config_dict[key] = value
    return config_dict


__all__ = ["normalize_config"]
