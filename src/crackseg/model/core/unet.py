"""Shim for backward compatibility: `BaseUNet` moved to `unet_core`.

Exports:
- BaseUNet: crackseg.model.core.unet_core.BaseUNet
"""

from __future__ import annotations

from .unet_core import BaseUNet

__all__ = ["BaseUNet"]
