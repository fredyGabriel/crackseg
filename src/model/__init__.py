"""
Model module initialization.

Imports and exposes the main components and utilities from the model module:
- Factory functions for creating model components
- Configuration validation utilities
- Abstract base models
- Configuration schema dataclasses
"""

# Abstract base model and factory functions
from .factory.factory import (
    create_unet,
    validate_config,
    ConfigurationError
)
from .base.abstract import UNetBase as ModelBase

# Configuration validation
from .factory.config import (
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

# Import configuration dataclasses
from .factory.config_schema import (
    EncoderConfig,
    BottleneckConfig,
    DecoderConfig,
    UNetConfig,
    load_unet_config_from_yaml,
    validate_unet_config
)

# Registries for component classes
from .factory.registry import Registry  # Import only Registry

# Main concrete implementation
from .core.unet import BaseUNet

# Abstract base classes
from .base import EncoderBase, DecoderBase, BottleneckBase, UNetBase

# Main concrete implementations
from .architectures import (
    CNNEncoder,
    ConvLSTMBottleneck,
    CNNDecoder,
    CNNConvLSTMUNet
)

# Optional: advanced/variant implementations
from .architectures.swinv2_cnn_aspp_unet import SwinV2CnnAsppUNet
from .encoder.swin_v2_adapter import SwinV2EncoderAdapter
from .components.aspp import ASPPModule
from .bottleneck.cnn_bottleneck import BottleneckBlock

# Make key classes and functions available at the module level
__all__ = [
    # Model base
    "ModelBase",

    # Concrete implementation
    "BaseUNet",

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

    # Configuration schema
    "EncoderConfig",
    "BottleneckConfig",
    "DecoderConfig",
    "UNetConfig",
    "load_unet_config_from_yaml",
    "validate_unet_config",

    # Registry
    "Registry",

    # Abstract base classes
    "EncoderBase",
    "DecoderBase",
    "BottleneckBase",
    "UNetBase",

    # Main concrete implementations
    "CNNEncoder",
    "ConvLSTMBottleneck",
    "CNNDecoder",
    "CNNConvLSTMUNet",

    # Optional: advanced/variant implementations
    "SwinV2CnnAsppUNet",
    "SwinV2EncoderAdapter",
    "ASPPModule",
    "BottleneckBlock",
]
