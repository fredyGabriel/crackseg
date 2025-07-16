"""
Configuration Validation System.

This package provides a validation system for model configurations.
"""

from crackseg.model.config.factory import (
    create_hybrid_model_from_config,
    create_model_from_config,
    get_model_config_schema,
    parse_architecture_config,
)

from .core import ConfigParam, ConfigSchema, ParamType
from .instantiation import (
    InstantiationError,
    instantiate_additional_component,
    instantiate_bottleneck,
    instantiate_decoder,
    instantiate_encoder,
    instantiate_hybrid_model,
)
from .schemas import (
    COMPONENT_VALIDATORS,
    create_architecture_schema,
    create_bottleneck_schema,
    create_decoder_schema,
    create_encoder_schema,
    create_hybrid_schema,
)
from .validation import (
    normalize_config,
    validate_architecture_config,
    validate_component_config,
)

__all__ = [
    # Core classes
    "ConfigParam",
    "ConfigSchema",
    "ParamType",
    # Validation functions
    "validate_component_config",
    "validate_architecture_config",
    "normalize_config",
    # Schema functions
    "create_encoder_schema",
    "create_bottleneck_schema",
    "create_decoder_schema",
    "create_architecture_schema",
    "create_hybrid_schema",
    # Factory functions
    "parse_architecture_config",
    "create_model_from_config",
    "create_hybrid_model_from_config",
    "get_model_config_schema",
    # Instantiation functions
    "instantiate_encoder",
    "instantiate_bottleneck",
    "instantiate_decoder",
    "instantiate_hybrid_model",
    "instantiate_additional_component",
    "InstantiationError",
    # Registry
    "COMPONENT_VALIDATORS",
]
