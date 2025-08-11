"""Thin compatibility layer to keep previous imports stable.

These builders are re-exported from the new modular package
``src/crackseg/model/config/schemas/``.
"""

from __future__ import annotations

from .schemas import (
    build_architecture_schema,
    build_aspp_schema,
    build_bottleneck_schema,
    build_cbam_schema,
    build_convlstm_schema,
    build_decoder_schema,
    build_encoder_schema,
    build_swinv2_schema,
)

__all__ = [
    "build_encoder_schema",
    "build_bottleneck_schema",
    "build_decoder_schema",
    "build_architecture_schema",
    "build_swinv2_schema",
    "build_aspp_schema",
    "build_convlstm_schema",
    "build_cbam_schema",
]
