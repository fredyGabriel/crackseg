"""Legacy shim for decoder components.

The actual implementations live in:
- `crackseg.model.decoder.blocks` (DecoderBlock and related)
- `crackseg.model.decoder.decoder_head` (CNNDecoder and helpers)

This module keeps backward-compatible import paths during the refactor.
"""

from __future__ import annotations

from .blocks import DecoderBlock, DecoderBlockAlias, DecoderBlockConfig
from .decoder_head import (
    CNNDecoder,
    CNNDecoderConfig,
    migrate_decoder_state_dict,
)

__all__ = [
    "DecoderBlockConfig",
    "DecoderBlock",
    "DecoderBlockAlias",
    "CNNDecoderConfig",
    "CNNDecoder",
    "migrate_decoder_state_dict",
]
