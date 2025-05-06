"""
Model module initialization.

Imports and exposes the main components and utilities from the model module:
- Factory functions for creating model components
- Configuration validation utilities
- Abstract base models
"""

# Abstract base model and factory functions
from .factory import (
    create_unet,
    validate_config,
    ConfigurationError
)
from .base import UNetBase as ModelBase

# Configuration validation
from .config import (
    validate_component_config,
    validate_architecture_config,
    normalize_config,
    parse_architecture_config,
    create_model_from_config,
    # Component instantiation
    instantiate_encoder,
    instantiate_bottleneck,
    instantiate_decoder,
    instantiate_hybrid_model,
    InstantiationError
)

# Registries for component classes
from .registry import Registry # Import only Registry

# Make key classes and functions available at the module level
__all__ = [
    # Model base
    "ModelBase",

    # Factory functions
    "create_unet",
    "validate_config",
    "ConfigurationError",
    "create_model_from_config",
    "parse_architecture_config",

    # Configuration validation
    "validate_component_config",
    "validate_architecture_config",
    "normalize_config",

    # Component instantiation
    "instantiate_encoder",
    "instantiate_bottleneck",
    "instantiate_decoder",
    "instantiate_hybrid_model",
    "InstantiationError",

    # Registry
    "Registry",
]
