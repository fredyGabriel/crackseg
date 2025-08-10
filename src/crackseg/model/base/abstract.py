"""Compatibility shim for legacy imports.

Allows importing base classes from `crackseg.model.base.abstract` while the
actual implementations live in dedicated modules.
"""

from __future__ import annotations

from .bottleneck_base import BottleneckBase
from .decoder_base import DecoderBase
from .encoder_base import EncoderBase
from .unet_base import UNetBase

__all__ = [
    "EncoderBase",
    "BottleneckBase",
    "DecoderBase",
    "UNetBase",
]
