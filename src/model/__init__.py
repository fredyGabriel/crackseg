"""Model components for the CrackSeg project."""

# Import registry for component registration
from src.model.registry import Registry

# Import base classes
from src.model.base import EncoderBase, BottleneckBase, DecoderBase, UNetBase

# Import factories
from src.model.factory import (
    create_encoder,
    create_bottleneck,
    create_decoder,
    create_unet,
    encoder_registry,
    bottleneck_registry,
    decoder_registry
)

# Import encoders, bottlenecks, and decoders
from src.model.encoder.cnn_encoder import CNNEncoder, EncoderBlock
from src.model.bottleneck.cnn_bottleneck import BottleneckBlock
from src.model.decoder.cnn_decoder import CNNDecoder, DecoderBlock

# Import UNet model
from src.model.unet import BaseUNet

__all__ = [
    'Registry',
    'EncoderBase', 'BottleneckBase', 'DecoderBase', 'UNetBase',
    'create_encoder', 'create_bottleneck', 'create_decoder', 'create_unet',
    'encoder_registry', 'bottleneck_registry', 'decoder_registry',
    'CNNEncoder', 'EncoderBlock',
    'BottleneckBlock',
    'CNNDecoder', 'DecoderBlock',
    'BaseUNet'
] 