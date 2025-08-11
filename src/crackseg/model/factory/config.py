"""Shim module re-exporting split factory functionality.

This preserves the public API of `crackseg.model.factory.config` while the
implementation lives in focused modules.
"""

from __future__ import annotations

from .instantiate import (
    InstantiationError,
    create_model_from_config,
    instantiate_bottleneck,
    instantiate_decoder,
    instantiate_encoder,
    instantiate_hybrid_model,
    parse_architecture_config,
)
from .normalize import normalize_config
from .validation import validate_architecture_config, validate_component_config

__all__ = [
    # Exceptions
    "InstantiationError",
    # Validation
    "validate_component_config",
    "validate_architecture_config",
    # Normalize
    "normalize_config",
    # Parse
    "parse_architecture_config",
    # Instantiate
    "instantiate_encoder",
    "instantiate_bottleneck",
    "instantiate_decoder",
    "instantiate_hybrid_model",
    # High-level
    "create_model_from_config",
]
