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

from .core import ConfigParam, ConfigSchema, ParamType


# Component-specific schema definitions
def create_encoder_schema() -> ConfigSchema:
    """Create schema for encoder components.

    Returns:
        ConfigSchema: The encoder schema definition.
    """
    return ConfigSchema(
        name="encoder",
        params=[
            ConfigParam(
                name="type",
                param_type=ParamType.STRING,
                required=True,
                description="Type of encoder component",
            ),
            ConfigParam(
                name="in_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of input channels",
            ),
            ConfigParam(
                name="hidden_dims",
                param_type=ParamType.LIST,
                required=False,
                default=[64, 128, 256, 512],
                description="List of hidden dimensions for encoder blocks",
            ),
            ConfigParam(
                name="dropout",
                param_type=ParamType.FLOAT,
                required=False,
                default=0.0,
                description="Dropout rate",
            ),
        ],
    )


def create_bottleneck_schema() -> ConfigSchema:
    """Create schema for bottleneck components.

    Returns:
        ConfigSchema: The bottleneck schema definition.
    """
    return ConfigSchema(
        name="bottleneck",
        params=[
            ConfigParam(
                name="type",
                param_type=ParamType.STRING,
                required=True,
                description="Type of bottleneck component",
            ),
            ConfigParam(
                name="in_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of input channels",
            ),
            ConfigParam(
                name="out_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of output channels",
            ),
            # ASPP specific parameters
            ConfigParam(
                name="atrous_rates",
                param_type=ParamType.LIST,
                required=False,
                default=[6, 12, 18],
                description="Atrous rates for ASPP module",
            ),
            # ConvLSTM specific parameters
            ConfigParam(
                name="hidden_channels",
                param_type=ParamType.INTEGER,
                required=False,
                description="Number of hidden channels for ConvLSTM",
            ),
            ConfigParam(
                name="kernel_size",
                param_type=ParamType.INTEGER,
                required=False,
                default=3,
                description="Kernel size for ConvLSTM",
            ),
        ],
    )


def create_decoder_schema() -> ConfigSchema:
    """Create schema for decoder components.

    Returns:
        ConfigSchema: The decoder schema definition.
    """
    return ConfigSchema(
        name="decoder",
        params=[
            ConfigParam(
                name="type",
                param_type=ParamType.STRING,
                required=True,
                description="Type of decoder component",
            ),
            ConfigParam(
                name="in_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of input channels",
            ),
            ConfigParam(
                name="out_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of output channels",
            ),
            ConfigParam(
                name="hidden_dims",
                param_type=ParamType.LIST,
                required=False,
                description="List of hidden dimensions for decoder blocks",
            ),
            ConfigParam(
                name="use_attention",
                param_type=ParamType.BOOLEAN,
                required=False,
                default=False,
                description="Whether to use attention mechanism",
            ),
            ConfigParam(
                name="attention_type",
                param_type=ParamType.STRING,
                required=False,
                description="Type of attention mechanism to use",
            ),
        ],
    )


def create_architecture_schema() -> ConfigSchema:
    """Create schema for full architecture configuration.

    Returns:
        ConfigSchema: The architecture schema definition.
    """
    encoder_schema = create_encoder_schema()
    bottleneck_schema = create_bottleneck_schema()
    decoder_schema = create_decoder_schema()

    return ConfigSchema(
        name="architecture",
        params=[
            ConfigParam(
                name="type",
                param_type=ParamType.STRING,
                required=True,
                description="Type of architecture",
            ),
            ConfigParam(
                name="encoder",
                param_type=ParamType.NESTED,
                required=True,
                nested_schema=encoder_schema,
                description="Encoder configuration",
            ),
            ConfigParam(
                name="bottleneck",
                param_type=ParamType.NESTED,
                required=True,
                nested_schema=bottleneck_schema,
                description="Bottleneck configuration",
            ),
            ConfigParam(
                name="decoder",
                param_type=ParamType.NESTED,
                required=True,
                nested_schema=decoder_schema,
                description="Decoder configuration",
            ),
            ConfigParam(
                name="in_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of input channels",
            ),
            ConfigParam(
                name="out_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of output channels",
            ),
        ],
    )


def create_hybrid_schema() -> ConfigSchema:
    """Create schema for hybrid architecture configuration.

    Returns:
        ConfigSchema: The hybrid architecture schema definition.
    """
    basic_schema = create_architecture_schema()

    # Add hybrid-specific parameters
    hybrid_params = basic_schema.params + [
        ConfigParam(
            name="components",
            param_type=ParamType.DICT,
            required=False,
            description="Additional components for hybrid architectures",
        ),
        ConfigParam(
            name="connections",
            param_type=ParamType.LIST,
            required=False,
            description="Connection definitions between components",
        ),
    ]

    return ConfigSchema(
        name="hybrid_architecture",
        params=hybrid_params,
        allow_unknown=True,  # Allow flexibility for diff hybrid archs
    )


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
    schema = ConfigSchema(
        name="swinv2_encoder",
        params=[
            ConfigParam(
                name="type",
                param_type=ParamType.STRING,
                required=True,
                choices=["SwinV2"],
                description="Must be 'SwinV2'",
            ),
            ConfigParam(
                name="in_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of input channels",
            ),
            ConfigParam(
                name="embed_dim",
                param_type=ParamType.INTEGER,
                required=False,
                default=96,
                description="Embedding dimension",
            ),
            ConfigParam(
                name="depths",
                param_type=ParamType.LIST,
                required=False,
                default=[2, 2, 6, 2],
                description="Depth of Swin layers",
            ),
            ConfigParam(
                name="num_heads",
                param_type=ParamType.LIST,
                required=False,
                default=[3, 6, 12, 24],
                description="Attention heads per layer",
            ),
            ConfigParam(
                name="window_size",
                param_type=ParamType.INTEGER,
                required=False,
                default=7,
                description="Window size for attention",
            ),
            ConfigParam(
                name="pretrained",
                param_type=ParamType.BOOLEAN,
                required=False,
                default=False,
                description="Whether to use pretrained weights",
            ),
            ConfigParam(
                name="pretrained_path",
                param_type=ParamType.STRING,
                required=False,
                description="Path to pretrained weights file",
            ),
        ],
    )

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
    schema = ConfigSchema(
        name="aspp_bottleneck",
        params=[
            ConfigParam(
                name="type",
                param_type=ParamType.STRING,
                required=True,
                choices=["ASPPModule"],
                description="Must be 'ASPPModule'",
            ),
            ConfigParam(
                name="in_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of input channels",
            ),
            ConfigParam(
                name="out_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of output channels",
            ),
            ConfigParam(
                name="atrous_rates",
                param_type=ParamType.LIST,
                required=False,
                default=[6, 12, 18],
                description="Atrous rates for ASPP module",
            ),
            ConfigParam(
                name="dropout_rate",
                param_type=ParamType.FLOAT,
                required=False,
                default=0.1,
                description="Dropout rate for ASPP module",
            ),
        ],
    )

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
    schema = ConfigSchema(
        name="convlstm_bottleneck",
        params=[
            ConfigParam(
                name="type",
                param_type=ParamType.STRING,
                required=True,
                choices=["ConvLSTMBottleneck"],
                description="Must be 'ConvLSTMBottleneck'",
            ),
            ConfigParam(
                name="in_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of input channels",
            ),
            ConfigParam(
                name="hidden_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of hidden channels",
            ),
            ConfigParam(
                name="out_channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of output channels",
            ),
            ConfigParam(
                name="kernel_size",
                param_type=ParamType.INTEGER,
                required=False,
                default=3,
                description="Kernel size for ConvLSTM",
            ),
            ConfigParam(
                name="num_layers",
                param_type=ParamType.INTEGER,
                required=False,
                default=1,
                description="Number of ConvLSTM layers",
            ),
            ConfigParam(
                name="batch_first",
                param_type=ParamType.BOOLEAN,
                required=False,
                default=True,
                description="Whether input has batch dimension first",
            ),
        ],
    )

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
    schema = ConfigSchema(
        name="cbam_attention",
        params=[
            ConfigParam(
                name="type",
                param_type=ParamType.STRING,
                required=True,
                choices=["CBAM"],
                description="Must be 'CBAM'",
            ),
            ConfigParam(
                name="channels",
                param_type=ParamType.INTEGER,
                required=True,
                description="Number of channels to apply attention to",
            ),
            ConfigParam(
                name="reduction_ratio",
                param_type=ParamType.INTEGER,
                required=False,
                default=16,
                description="Channel reduction ratio",
            ),
            ConfigParam(
                name="spatial_kernel_size",
                param_type=ParamType.INTEGER,
                required=False,
                default=7,
                description="Kernel size for spatial attention",
            ),
        ],
    )

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
