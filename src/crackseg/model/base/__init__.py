"""
Base abstract model classes.

This module contains all abstract base classes that define interfaces for
the model components (encoder, bottleneck, decoder) and the U-Net model itself.
"""

from .bottleneck_base import BottleneckBase
from .decoder_base import DecoderBase
from .encoder_base import EncoderBase
from .unet_base import UNetBase

__all__ = ["EncoderBase", "BottleneckBase", "DecoderBase", "UNetBase"]
