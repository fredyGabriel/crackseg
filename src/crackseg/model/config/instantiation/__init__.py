"""
Model Instantiation Package.

This package provides the infrastructure for instantiating PyTorch models
from configuration dictionaries. It handles component instantiation,
activation layers, and model assembly.
"""

from crackseg.model.factory.registry_setup import (
    bottleneck_registry,
    component_registries,
    decoder_registry,
    encoder_registry,
)

from .activation_handler import apply_final_activation
from .components import (
    InstantiationError,
    _instantiate_component,
    _prepare_component_config,
    cache_component,
    generate_cache_key,
    get_cached_component,
    instantiate_additional_component,
    instantiate_component_from_registry,
)
from .hybrid import (
    instantiate_bottleneck,
    instantiate_decoder,
    instantiate_encoder,
    instantiate_hybrid_model,
    instantiate_model,
    instantiate_model_from_config,
)


def _try_instantiate_encoder(config: dict | None):  # type: ignore[override]
    """Helper used in tests; instantiate if config provided, else None."""
    if config is None:
        return None
    return instantiate_encoder(config)


def _try_instantiate_bottleneck(config: dict | None, encoder):  # type: ignore[override]
    """Helper used in tests; instantiate if config provided, else None."""
    if config is None:
        return None
    return instantiate_bottleneck(config)


def _try_instantiate_decoder(config: dict | None, encoder, bottleneck):  # type: ignore[override]
    """Helper used in tests; instantiate if config provided, else None."""
    if config is None:
        return None
    return instantiate_decoder(config)


__all__ = [
    # Exceptions
    "InstantiationError",
    # Low-level component helpers
    "instantiate_component_from_registry",
    "_instantiate_component",
    "_prepare_component_config",
    "instantiate_additional_component",
    # Expose cache helpers for tests
    "cache_component",
    "get_cached_component",
    "generate_cache_key",
    # Activation
    "apply_final_activation",
    # High-level API
    "instantiate_encoder",
    "instantiate_bottleneck",
    "instantiate_decoder",
    "instantiate_hybrid_model",
    "instantiate_model_from_config",
    "instantiate_model",
    # Registries for patching in tests
    "encoder_registry",
    "bottleneck_registry",
    "decoder_registry",
    "component_registries",
    # Test helpers
    "_try_instantiate_encoder",
    "_try_instantiate_bottleneck",
    "_try_instantiate_decoder",
]
