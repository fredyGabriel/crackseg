"""Model package initialization."""

# Import base classes and components directly if needed elsewhere,
# but remove factory imports as they are commented out.
from src.model.base import UNetBase, EncoderBase, BottleneckBase, DecoderBase
# from src.model.components import *  # Example if components are directly used
from src.model.registry import Registry

# Remove factory imports
# from src.model.factory import (
#     create_encoder,
#     create_bottleneck,
#     create_decoder,
#     create_unet,
#     encoder_registry,
#     bottleneck_registry,
#     decoder_registry,
# )

# Expose registries if they are intended to be public API
# __all__ = [
#     "UNetBase",
#     "EncoderBase",
#     "BottleneckBase",
#     "DecoderBase",
#     "Registry",
#     "encoder_registry",
#     "bottleneck_registry",
#     "decoder_registry",
#     "create_encoder",
#     "create_bottleneck",
#     "create_decoder",
#     "create_unet"
# ]

# Simplified __all__ assuming direct class usage and no factory exposure
__all__ = [
    "UNetBase",
    "EncoderBase",
    "BottleneckBase",
    "DecoderBase",
    "Registry",
]
