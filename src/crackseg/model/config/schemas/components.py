from __future__ import annotations

from ..core import ConfigParam, ConfigSchema, ParamType


def build_encoder_schema() -> ConfigSchema:
    return ConfigSchema(
        name="encoder",
        params=[
            ConfigParam(
                "type",
                ParamType.STRING,
                True,
                description="Type of encoder component",
            ),
            ConfigParam(
                "in_channels",
                ParamType.INTEGER,
                True,
                description="Number of input channels",
            ),
            ConfigParam(
                "hidden_dims",
                ParamType.LIST,
                False,
                default=[64, 128, 256, 512],
                description="Hidden dims",
            ),
            ConfigParam(
                "dropout",
                ParamType.FLOAT,
                False,
                default=0.0,
                description="Dropout rate",
            ),
        ],
    )


def build_bottleneck_schema() -> ConfigSchema:
    return ConfigSchema(
        name="bottleneck",
        params=[
            ConfigParam(
                "type",
                ParamType.STRING,
                True,
                description="Type of bottleneck component",
            ),
            ConfigParam(
                "in_channels",
                ParamType.INTEGER,
                True,
                description="Number of input channels",
            ),
            ConfigParam(
                "out_channels",
                ParamType.INTEGER,
                True,
                description="Number of output channels",
            ),
            ConfigParam(
                "atrous_rates",
                ParamType.LIST,
                False,
                default=[6, 12, 18],
                description="Atrous rates",
            ),
            ConfigParam(
                "hidden_channels",
                ParamType.INTEGER,
                False,
                description="ConvLSTM hidden channels",
            ),
            ConfigParam(
                "kernel_size",
                ParamType.INTEGER,
                False,
                default=3,
                description="ConvLSTM kernel size",
            ),
        ],
    )


def build_decoder_schema() -> ConfigSchema:
    return ConfigSchema(
        name="decoder",
        params=[
            ConfigParam(
                "type",
                ParamType.STRING,
                True,
                description="Type of decoder component",
            ),
            ConfigParam(
                "in_channels",
                ParamType.INTEGER,
                True,
                description="Number of input channels",
            ),
            ConfigParam(
                "out_channels",
                ParamType.INTEGER,
                True,
                description="Number of output channels",
            ),
            ConfigParam(
                "hidden_dims",
                ParamType.LIST,
                False,
                description="Hidden dimensions for blocks",
            ),
            ConfigParam(
                "use_attention",
                ParamType.BOOLEAN,
                False,
                default=False,
                description="Use attention",
            ),
            ConfigParam(
                "attention_type",
                ParamType.STRING,
                False,
                description="Attention type",
            ),
        ],
    )


def build_architecture_schema() -> ConfigSchema:
    encoder_schema = build_encoder_schema()
    bottleneck_schema = build_bottleneck_schema()
    decoder_schema = build_decoder_schema()

    return ConfigSchema(
        name="architecture",
        params=[
            ConfigParam(
                "type",
                ParamType.STRING,
                True,
                description="Type of architecture",
            ),
            ConfigParam(
                "encoder",
                ParamType.NESTED,
                True,
                nested_schema=encoder_schema,
                description="Encoder config",
            ),
            ConfigParam(
                "bottleneck",
                ParamType.NESTED,
                True,
                nested_schema=bottleneck_schema,
                description="Bottleneck config",
            ),
            ConfigParam(
                "decoder",
                ParamType.NESTED,
                True,
                nested_schema=decoder_schema,
                description="Decoder config",
            ),
            ConfigParam(
                "in_channels",
                ParamType.INTEGER,
                True,
                description="Number of input channels",
            ),
            ConfigParam(
                "out_channels",
                ParamType.INTEGER,
                True,
                description="Number of output channels",
            ),
        ],
    )
