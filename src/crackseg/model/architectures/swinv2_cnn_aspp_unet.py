"""Shim for backward compatibility: re-export hybrid architecture core.

Exports:
- SwinV2CnnAsppUNet: crackseg.model.architectures.swinv2_cnn_aspp_unet_core
"""

from __future__ import annotations

from .swinv2_cnn_aspp_unet_core import SwinV2CnnAsppUNet

__all__ = ["SwinV2CnnAsppUNet"]
