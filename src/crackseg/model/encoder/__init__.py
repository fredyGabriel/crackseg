"""Encoder components for the CrackSeg project."""

from crackseg.model.encoder.cnn_encoder import CNNEncoder, EncoderBlock
from crackseg.model.encoder.feature_info_utils import (
    build_feature_info_from_channels,
    create_feature_info_entry,
    validate_feature_info,
)
from crackseg.model.encoder.swin_transformer_encoder import (
    SwinTransformerEncoder,
)
from crackseg.model.encoder.swin_v2_adapter import SwinV2EncoderAdapter

__all__ = [
    "CNNEncoder",
    "EncoderBlock",
    "SwinTransformerEncoder",
    "SwinV2EncoderAdapter",
    "create_feature_info_entry",
    "build_feature_info_from_channels",
    "validate_feature_info",
]
