"""
Base abstract model classes.

This module contains all abstract base classes that define interfaces for
the model components (encoder, bottleneck, decoder) and the U-Net model itself.
"""

from .abstract import BottleneckBase, DecoderBase, EncoderBase, UNetBase

__all__ = ["EncoderBase", "BottleneckBase", "DecoderBase", "UNetBase"]
