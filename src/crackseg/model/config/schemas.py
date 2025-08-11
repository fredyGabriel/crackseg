"""
Component Configuration Schema Definitions.

This module provides schema definitions for validating configuration of various
model components:
- Encoders (SwinV2, CNN, etc.)
- Bottlenecks (ASPP, ConvLSTM, etc.)
- Decoders
- Attention mechanisms (CBAM, etc.)
- Complete architecture configurations
"""

from typing import Any

from .core import ConfigSchema
from .schemas_utils import (
    build_architecture_schema,
    build_aspp_schema,
    build_bottleneck_schema,
    build_cbam_schema,
    build_convlstm_schema,
    build_decoder_schema,
    build_encoder_schema,
    build_hybrid_schema,
    build_swinv2_schema,
)


# Component-specific schema definitions
def create_encoder_schema() -> ConfigSchema:
    """Create schema for encoder components.

    Returns:
        ConfigSchema: The encoder schema definition.
    """
    return build_encoder_schema()


def create_bottleneck_schema() -> ConfigSchema:
    """Create schema for bottleneck components.

    Returns:
        ConfigSchema: The bottleneck schema definition.
    """
    return build_bottleneck_schema()


def create_decoder_schema() -> ConfigSchema:
    """Create schema for decoder components.

    Returns:
        ConfigSchema: The decoder schema definition.
    """
    return build_decoder_schema()


def create_architecture_schema() -> ConfigSchema:
    """Create schema for full architecture configuration.

    Returns:
        ConfigSchema: The architecture schema definition.
    """
    return build_architecture_schema()


def create_hybrid_schema() -> ConfigSchema:
    """Create schema for hybrid architecture configuration.

    Returns:
        ConfigSchema: The hybrid architecture schema definition.
    """
    return build_hybrid_schema()


# Component-specific validators
def validate_swinv2_config(
    config: dict[str, Any],
) -> tuple[bool, dict[str, str] | None]:
    """Validate SwinV2 encoder configuration.

    Args:
        config (dict[str, Any]): Configuration dictionary.

    Returns:
        tuple[bool, dict[str, str] | None]: (is_valid, errors)
    """
    schema = build_swinv2_schema()
    return schema.validate(config)


def validate_aspp_config(
    config: dict[str, Any],
) -> tuple[bool, dict[str, str] | None]:
    """Validate ASPP bottleneck configuration.

    Args:
        config (dict[str, Any]): Configuration dictionary.

    Returns:
        tuple[bool, dict[str, str] | None]: (is_valid, errors)
    """
    schema = build_aspp_schema()
    return schema.validate(config)


def validate_convlstm_config(
    config: dict[str, Any],
) -> tuple[bool, dict[str, str] | None]:
    """Validate ConvLSTM bottleneck configuration.

    Args:
        config (dict[str, Any]): Configuration dictionary.

    Returns:
        tuple[bool, dict[str, str] | None]: (is_valid, errors)
    """
    schema = build_convlstm_schema()
    return schema.validate(config)


def validate_cbam_config(
    config: dict[str, Any],
) -> tuple[bool, dict[str, str] | None]:
    """Validate CBAM attention configuration.

    Args:
        config (dict[str, Any]): Configuration dictionary.

    Returns:
        tuple[bool, dict[str, str] | None]: (is_valid, errors)
    """
    schema = build_cbam_schema()
    return schema.validate(config)


# Create validator registry
COMPONENT_VALIDATORS = {
    # Encoders
    "SwinV2": validate_swinv2_config,
    # Bottlenecks
    "ASPPModule": validate_aspp_config,
    "ConvLSTMBottleneck": validate_convlstm_config,
    # Attention modules
    "CBAM": validate_cbam_config,
}
