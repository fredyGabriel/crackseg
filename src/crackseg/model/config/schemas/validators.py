from __future__ import annotations

from ..core import ConfigParam, ConfigSchema, ParamType


def build_swinv2_schema() -> ConfigSchema:
    return ConfigSchema(
        name="swinv2_encoder",
        params=[
            ConfigParam(
                "type",
                ParamType.STRING,
                True,
                choices=["SwinV2"],
                description="Must be 'SwinV2'",
            ),
            ConfigParam(
                "in_channels",
                ParamType.INTEGER,
                True,
                description="Number of input channels",
            ),
            ConfigParam(
                "embed_dim",
                ParamType.INTEGER,
                False,
                default=96,
                description="Embedding dimension",
            ),
            ConfigParam(
                "depths",
                ParamType.LIST,
                False,
                default=[2, 2, 6, 2],
                description="Swin depths",
            ),
            ConfigParam(
                "num_heads",
                ParamType.LIST,
                False,
                default=[3, 6, 12, 24],
                description="Attention heads",
            ),
            ConfigParam(
                "window_size",
                ParamType.INTEGER,
                False,
                default=7,
                description="Window size",
            ),
            ConfigParam(
                "pretrained",
                ParamType.BOOLEAN,
                False,
                default=False,
                description="Use pretrained",
            ),
            ConfigParam(
                "pretrained_path",
                ParamType.STRING,
                False,
                description="Weights path",
            ),
        ],
    )


def build_aspp_schema() -> ConfigSchema:
    return ConfigSchema(
        name="aspp_bottleneck",
        params=[
            ConfigParam(
                "type",
                ParamType.STRING,
                True,
                choices=["ASPPModule"],
                description="Must be 'ASPPModule'",
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
                "dropout_rate",
                ParamType.FLOAT,
                False,
                default=0.1,
                description="Dropout rate",
            ),
        ],
    )


def build_convlstm_schema() -> ConfigSchema:
    return ConfigSchema(
        name="convlstm_bottleneck",
        params=[
            ConfigParam(
                "type",
                ParamType.STRING,
                True,
                choices=["ConvLSTMBottleneck"],
                description="Must be 'ConvLSTMBottleneck'",
            ),
            ConfigParam(
                "in_channels",
                ParamType.INTEGER,
                True,
                description="Number of input channels",
            ),
            ConfigParam(
                "hidden_channels",
                ParamType.INTEGER,
                True,
                description="Number of hidden channels",
            ),
            ConfigParam(
                "out_channels",
                ParamType.INTEGER,
                True,
                description="Number of output channels",
            ),
            ConfigParam(
                "kernel_size",
                ParamType.INTEGER,
                False,
                default=3,
                description="Kernel size",
            ),
            ConfigParam(
                "num_layers",
                ParamType.INTEGER,
                False,
                default=1,
                description="Number of layers",
            ),
            ConfigParam(
                "batch_first",
                ParamType.BOOLEAN,
                False,
                default=True,
                description="Batch dimension first",
            ),
        ],
    )


def build_cbam_schema() -> ConfigSchema:
    return ConfigSchema(
        name="cbam_attention",
        params=[
            ConfigParam(
                "type",
                ParamType.STRING,
                True,
                choices=["CBAM"],
                description="Must be 'CBAM'",
            ),
            ConfigParam(
                "channels", ParamType.INTEGER, True, description="Channels"
            ),
            ConfigParam(
                "reduction_ratio",
                ParamType.INTEGER,
                False,
                default=16,
                description="Reduction ratio",
            ),
            ConfigParam(
                "spatial_kernel_size",
                ParamType.INTEGER,
                False,
                default=7,
                description="Spatial kernel",
            ),
        ],
    )
