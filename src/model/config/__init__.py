"""
Configuration Validation System.

This package provides a validation system for model configurations.
"""

from .validation import (
    validate_component_config,
    validate_architecture_config,
    normalize_config
)

from .core import (
    ConfigParam,
    ConfigSchema,
    ParamType
)

from .schemas import (
    create_encoder_schema,
    create_bottleneck_schema,
    create_decoder_schema,
    create_architecture_schema,
    create_hybrid_schema,
    COMPONENT_VALIDATORS
)

from src.model.config.factory import (
    parse_architecture_config,
    get_model_config_schema
)

from .instantiation import (
    instantiate_encoder,
    instantiate_bottleneck,
    instantiate_decoder,
    instantiate_hybrid_model,
    instantiate_additional_component,
    InstantiationError,
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
    "COMPONENT_VALIDATORS"
]
