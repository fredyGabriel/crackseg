"""
Factory and registry functionality for model components.

This module provides functions to create and register model components
(encoders, bottlenecks, decoders) and complete UNet models.
"""

from .factory import ConfigurationError, create_unet, validate_config
from .registry import Registry
from .registry_setup import component_registries

__all__ = [
    "create_unet",
    "validate_config",
    "ConfigurationError",
    "Registry",
    "component_registries",
]
