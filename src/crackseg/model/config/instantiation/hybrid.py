"""
High-level model instantiation helpers (hybrid models).

This module provides public functions to instantiate encoder, bottleneck,
decoder, and complete hybrid models from configuration dictionaries, using
the shared component instantiation utilities.
"""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

from crackseg.model.factory.registry_setup import (
    bottleneck_registry,
    decoder_registry,
    encoder_registry,
)

from . import (
    InstantiationError,
    apply_final_activation,
    instantiate_component_from_registry,
)

log = logging.getLogger(__name__)


def instantiate_encoder(
    config: dict[str, Any], use_cache: bool = True
) -> nn.Module:
    """Instantiate an encoder component.

    Args:
        config: Encoder configuration
        use_cache: Whether to use component caching

    Returns:
        Instantiated encoder
    """
    return instantiate_component_from_registry(
        config, encoder_registry, "encoder", use_cache=use_cache
    )


def instantiate_bottleneck(
    config: dict[str, Any],
    runtime_params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> nn.Module:
    """Instantiate a bottleneck component.

    Args:
        config: Bottleneck configuration
        runtime_params: Runtime parameters (e.g., input channels)
        use_cache: Whether to use component caching

    Returns:
        Instantiated bottleneck
    """
    return instantiate_component_from_registry(
        config, bottleneck_registry, "bottleneck", runtime_params, use_cache
    )


def instantiate_decoder(
    config: dict[str, Any],
    runtime_params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> nn.Module:
    """Instantiate a decoder component.

    Args:
        config: Decoder configuration
        runtime_params: Runtime parameters (e.g., input channels)
        use_cache: Whether to use component caching

    Returns:
        Instantiated decoder
    """
    return instantiate_component_from_registry(
        config, decoder_registry, "decoder", runtime_params, use_cache
    )


def instantiate_hybrid_model(
    config: dict[str, Any],
    input_channels: int = 3,
    use_cache: bool = True,
) -> nn.Module:
    """Instantiate a complete hybrid model from configuration.

    This function coordinates the instantiation of encoder, bottleneck,
    and decoder components, then assembles them into a complete model.

    Args:
        config: Complete model configuration
        input_channels: Number of input channels (default: 3 for RGB)
        use_cache: Whether to use component caching

    Returns:
        Complete instantiated model

    Raises:
        InstantiationError: If any component instantiation fails
    """
    try:
        log.info("Starting hybrid model instantiation")

        # Validate required components
        if "encoder" not in config:
            raise InstantiationError(
                "Model config must include 'encoder' section"
            )
        if "decoder" not in config:
            raise InstantiationError(
                "Model config must include 'decoder' section"
            )

        # Instantiate encoder
        encoder = instantiate_encoder(config["encoder"], use_cache=use_cache)
        log.debug("Encoder instantiated successfully")

        # Instantiate bottleneck (optional)
        bottleneck = None
        if "bottleneck" in config:
            bottleneck_config = config["bottleneck"]
            bottleneck_params = {"in_channels": input_channels}
            bottleneck = instantiate_bottleneck(
                bottleneck_config, bottleneck_params, use_cache=use_cache
            )
            log.debug("Bottleneck instantiated successfully")

        # Instantiate decoder
        decoder_config = config["decoder"]
        decoder_params = {"in_channels": input_channels}
        decoder = instantiate_decoder(
            decoder_config, decoder_params, use_cache=use_cache
        )
        log.debug("Decoder instantiated successfully")

        # Assemble model
        if bottleneck:
            model = nn.Sequential(encoder, bottleneck, decoder)
            log.info("Hybrid model assembled with bottleneck")
        else:
            model = nn.Sequential(encoder, decoder)
            log.info("Hybrid model assembled without bottleneck")

        # Apply final activation if configured
        model = apply_final_activation(model, config)

        log.info("Hybrid model instantiation completed successfully")
        return model

    except InstantiationError:
        # Re-raise InstantiationError directly
        raise
    except Exception as e:  # noqa: BLE001
        # Wrap unexpected errors
        error_msg = f"Unexpected error during hybrid model instantiation: {e}"
        log.error(error_msg, exc_info=True)
        raise InstantiationError(error_msg) from e


def instantiate_model_from_config(
    config: dict[str, Any],
    model_type: str = "hybrid",
    **kwargs: Any,
) -> nn.Module:
    """Main entry point for model instantiation.

    Args:
        config: Model configuration dictionary
        model_type: Type of model to instantiate (default: "hybrid")
        **kwargs: Additional arguments passed to specific instantiation functions

    Returns:
        Instantiated model

    Raises:
        InstantiationError: If instantiation fails
        ValueError: If model_type is not supported
    """
    if model_type == "hybrid":
        return instantiate_hybrid_model(config, **kwargs)
    msg = f"Unsupported model type: {model_type}"
    raise ValueError(msg)


def instantiate_model(config: dict[str, Any], **kwargs: Any) -> nn.Module:
    """Legacy function name for backward compatibility.

    Use instantiate_model_from_config instead.
    """
    log.warning(
        "instantiate_model is deprecated. Use instantiate_model_from_config instead."
    )
    return instantiate_model_from_config(config, **kwargs)
