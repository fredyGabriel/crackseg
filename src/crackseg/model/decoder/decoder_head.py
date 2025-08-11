"""
Decoder head implementation (CNNDecoder) split from legacy module.

Public API:
- CNNDecoderConfig
- CNNDecoder
- migrate_decoder_state_dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from crackseg.model.base.abstract import DecoderBase
from crackseg.model.decoder.common.channel_utils import (
    calculate_decoder_channels,
    validate_skip_channels_order,
)
from crackseg.model.factory.registry_setup import decoder_registry

from .blocks import DecoderBlock, DecoderBlockConfig

logger = logging.getLogger(__name__)


@dataclass
class CNNDecoderConfig:
    """Global configuration for `CNNDecoder` behavior and blocks."""

    upsample_scale_factor: int = 2
    upsample_mode: str = "bilinear"
    kernel_size: int = 3
    padding: int = 1
    use_cbam: bool = False
    cbam_reduction: int = 16


@decoder_registry.register("CNNDecoder", force=True)
class CNNDecoder(DecoderBase):
    """Standard CNN Decoder for U-Net built from `DecoderBlock`s."""

    decoder_blocks: nn.ModuleList
    final_conv: nn.Conv2d
    _out_channels: int
    skip_channels_list: list[int]
    decoder_channels: list[int]
    expected_channels: list[int]
    target_size: tuple[int, int] | None

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        skip_channels_list: list[int],
        out_channels: int = 1,
        depth: int | None = None,
        target_size: tuple[int, int] | None = None,
        config: CNNDecoderConfig | None = None,
    ) -> None:
        if config is None:
            config = CNNDecoderConfig()

        if not skip_channels_list or not all(
            c > 0 for c in skip_channels_list
        ):
            raise ValueError(
                "skip_channels_list must be a non-empty list of positive integers."
            )
        validate_skip_channels_order(skip_channels_list)

        super().__init__(in_channels, skip_channels=skip_channels_list)

        actual_depth = len(skip_channels_list) if depth is None else depth
        if depth is not None and depth != len(skip_channels_list):
            raise ValueError(
                "Length of skip_channels_list must match depth. "
                f"Got skip_channels_list={len(skip_channels_list)}, depth={depth}."
            )
        self.target_size = target_size
        self._out_channels = out_channels
        self.skip_channels_list = skip_channels_list

        decoder_block_out_channels = calculate_decoder_channels(
            in_channels, skip_channels_list
        )
        if len(decoder_block_out_channels) != actual_depth:
            raise ValueError(
                "Calculated decoder channels do not match depth: "
                f"{len(decoder_block_out_channels)} vs {actual_depth}"
            )
        self.decoder_channels = decoder_block_out_channels
        self.expected_channels = [in_channels] + decoder_block_out_channels

        for i, (skip_ch, dec_ch) in enumerate(
            zip(skip_channels_list, decoder_block_out_channels, strict=False)
        ):
            if skip_ch <= 0:
                raise ValueError(
                    f"Skip channel at index {i} must be positive, got {skip_ch}"
                )
            if dec_ch <= 0:
                raise ValueError(
                    f"Decoder channel at index {i} must be positive, got {dec_ch}"
                )

        decoder_block_cfg = DecoderBlockConfig(
            kernel_size=config.kernel_size,
            padding=config.padding,
            upsample_scale_factor=config.upsample_scale_factor,
            upsample_mode=config.upsample_mode,
            use_cbam=config.use_cbam,
            cbam_reduction=config.cbam_reduction,
        )

        self.decoder_blocks = nn.ModuleList()
        for i in range(actual_depth):
            block = DecoderBlock(
                in_channels=self.expected_channels[i],
                skip_channels=skip_channels_list[i],
                out_channels=decoder_block_out_channels[i],
                config=decoder_block_cfg,
            )
            self.decoder_blocks.append(block)

        self.final_conv = nn.Conv2d(
            self.expected_channels[-1], out_channels, kernel_size=1
        )

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        if len(skips) != len(self.skip_channels_list):
            raise ValueError(
                f"Expected {len(self.skip_channels_list)} skip connections, got {len(skips)}."
            )
        for i, (skip, expected_ch) in enumerate(
            zip(skips, self.skip_channels_list, strict=False)
        ):
            if skip.shape[1] != expected_ch:
                raise ValueError(
                    f"Skip connection {i} has {skip.shape[1]} channels, expected {expected_ch}."
                )

        out: torch.Tensor = x
        for block, skip in zip(self.decoder_blocks, skips, strict=False):
            out = cast(torch.Tensor, block(out, [skip]))

        if hasattr(self, "final_conv"):
            out = cast(torch.Tensor, self.final_conv(out))

        if self.target_size is not None:
            current_size = (out.shape[2], out.shape[3])
            if current_size != self.target_size:
                out = cast(
                    torch.Tensor,
                    F.interpolate(
                        out,
                        size=self.target_size,
                        mode="bilinear",
                        align_corners=False,
                    ),
                )
                logger.debug(
                    f"Final upsampling from {current_size} to {self.target_size}"
                )
        return out

    @property
    def out_channels(self) -> int:
        return self._out_channels


def migrate_decoder_state_dict(
    old_state_dict: dict[str, Any],
    decoder: nn.Module,
    verbose: bool = True,
) -> dict[str, Any]:
    """Map parameters from an old decoder state_dict into the new structure."""
    new_state_dict = decoder.state_dict()
    mapped = 0
    skipped = 0
    for k in new_state_dict.keys():
        if k in old_state_dict:
            new_state_dict[k] = old_state_dict[k]
            mapped += 1
            continue
        base_name = k.split(".")[-1]
        candidates = [ok for ok in old_state_dict if ok.endswith(base_name)]
        if candidates:
            new_state_dict[k] = old_state_dict[candidates[0]]
            mapped += 1
            if verbose:
                print(
                    f"[migrate_decoder_state_dict] Mapped {candidates[0]} -> {k}"
                )
        else:
            skipped += 1
            if verbose:
                print(
                    f"[migrate_decoder_state_dict] Could not map parameter: {k}"
                )
    if verbose:
        print(
            f"[migrate_decoder_state_dict] Migration complete: {mapped} mapped, {skipped} skipped."
        )
    return new_state_dict


__all__ = ["CNNDecoderConfig", "CNNDecoder", "migrate_decoder_state_dict"]
