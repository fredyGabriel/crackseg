"""
Decoder block implementations extracted from the legacy cnn_decoder module.

Public API:
- DecoderBlockConfig
- DecoderBlock
- DecoderBlockAlias
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from crackseg.model.base.abstract import DecoderBase
from crackseg.model.factory.registry_setup import component_registries

logger = logging.getLogger(__name__)

# Local constant for channel sanity warnings
MAX_RECOMMENDED_CHANNELS = 2048


@dataclass
class DecoderBlockConfig:
    """Configuration parameters for individual DecoderBlock components.

    See original documentation for details on each field and performance
    implications.
    """

    kernel_size: int = 3
    padding: int = 1
    upsample_scale_factor: int = 2
    upsample_mode: str = "bilinear"
    use_cbam: bool = False
    cbam_reduction: int = 16


class DecoderBlock(DecoderBase):
    """CNN Decoder block for U-Net architecture with static channel alignment."""

    in_channels: int
    _out_channels: int
    middle_channels: int
    kernel_size: int
    padding: int
    upsample_scale_factor: int
    upsample_mode: str
    use_cbam: bool
    cbam_reduction: int
    upsample: nn.Upsample
    up_conv: nn.Conv2d
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    relu1: nn.ReLU
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d
    relu2: nn.ReLU

    def _validate_input_channels(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        middle_channels: int,
    ) -> None:
        for name, value in [
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("middle_channels", middle_channels),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if skip_channels < 0:
            raise ValueError("skip_channels must be >= 0, got {skip_channels}")
        concat_channels = out_channels + skip_channels
        if concat_channels <= 0:
            pass
        if middle_channels < out_channels:
            raise ValueError(
                f"middle_channels ({middle_channels}) should be >= "
                f"out_channels ({out_channels})"
            )

    def _log_channel_warnings(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        middle_channels: int,
    ) -> None:
        if skip_channels == 0:
            logger.info(
                "DecoderBlock initialized with skip_channels=0, concatenation "
                "will be bypassed."
            )
        if in_channels < out_channels:
            logger.warning(
                "Upsampling via up_conv potentially increases channels from "
                f"{in_channels} to {out_channels}, this logic assumes up_conv "
                "maintains/reduces channels primarily."
            )
        if any(
            val > MAX_RECOMMENDED_CHANNELS
            for val in [
                in_channels,
                skip_channels,
                out_channels,
                middle_channels,
            ]
        ):
            logger.warning(
                "Very large channel dimension detected. This may cause memory "
                "issues."
            )

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int | None = None,
        middle_channels: int | None = None,
        config: DecoderBlockConfig | None = None,
    ):
        effective_out_channels = (
            in_channels // 2 if out_channels is None else out_channels
        )
        effective_middle_channels = (
            effective_out_channels * 2
            if middle_channels is None
            else middle_channels
        )

        if config is None:
            config = DecoderBlockConfig()

        self._validate_input_channels(
            in_channels,
            skip_channels,
            effective_out_channels,
            effective_middle_channels,
        )
        self._log_channel_warnings(
            in_channels,
            skip_channels,
            effective_out_channels,
            effective_middle_channels,
        )

        super().__init__(in_channels, skip_channels=[skip_channels])

        self.in_channels = in_channels
        self._skip_channels: list[int] = [skip_channels]
        self._out_channels = effective_out_channels
        self.middle_channels = effective_middle_channels

        self.kernel_size = config.kernel_size
        self.padding = config.padding
        self.upsample_scale_factor = config.upsample_scale_factor
        self.upsample_mode = config.upsample_mode
        self.use_cbam = config.use_cbam
        self.cbam_reduction = config.cbam_reduction

        self.upsample = nn.Upsample(
            scale_factor=self.upsample_scale_factor,
            mode=self.upsample_mode,
            align_corners=True if self.upsample_mode == "bilinear" else None,
        )
        self.up_conv = nn.Conv2d(
            self.in_channels, self._out_channels, kernel_size=1
        )

        concat_channels_for_cbam_and_conv1 = (
            self._out_channels + self._skip_channels[0]
        )
        if (
            concat_channels_for_cbam_and_conv1 <= 0
            and self._skip_channels[0] > 0
        ):
            raise ValueError(
                "concat_channels_for_cbam_and_conv1 "
                f"({concat_channels_for_cbam_and_conv1}) must be positive when"
                " skip_channels > 0"
            )

        self.cbam: nn.Module
        if self.use_cbam:
            if (
                concat_channels_for_cbam_and_conv1 <= self.cbam_reduction
                and self._skip_channels[0] > 0
            ):
                raise ValueError(
                    f"CBAM reduction ({self.cbam_reduction}) must be less "
                    "than concatenated channels "
                    f"({concat_channels_for_cbam_and_conv1}) when "
                    "skip_channels > 0"
                )
            attention_registry = component_registries.get("attention")
            if attention_registry is None:
                raise RuntimeError("Attention registry not found for CBAM.")
            self.cbam = attention_registry.instantiate(
                "CBAM",
                in_channels=(
                    concat_channels_for_cbam_and_conv1
                    if self._skip_channels[0] > 0
                    else self._out_channels
                ),
                reduction=self.cbam_reduction,
            )
        else:
            self.cbam = nn.Identity()

        conv1_in_channels = (
            concat_channels_for_cbam_and_conv1
            if self._skip_channels[0] > 0
            else self._out_channels
        )
        self.conv1 = nn.Conv2d(
            conv1_in_channels,
            self.middle_channels,
            self.kernel_size,
            padding=self.padding,
        )
        self.bn1 = nn.BatchNorm2d(self.middle_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            self.middle_channels,
            self._out_channels,
            self.kernel_size,
            padding=self.padding,
        )
        self.bn2 = nn.BatchNorm2d(self._out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def _validate_channel_compatibility(
        self,
    ) -> None:  # kept for compatibility
        pass

    def validate_forward_inputs(
        self, x: torch.Tensor, skip: torch.Tensor | None
    ) -> None:
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Input tensor has {x.size(1)} channels, expected {self.in_channels}"
            )
        if skip is not None and skip.size(1) != self.skip_channels[0]:
            raise ValueError(
                f"Skip connection has {skip.size(1)} channels, expected {self.skip_channels[0]}"
            )
        if skip is not None and x.shape[2:] != skip.shape[2:]:
            raise ValueError(
                f"Spatial dimensions mismatch: x {x.shape[2:]}, skip {skip.shape[2:]}"
            )

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        if not skips or len(skips) != 1:
            raise ValueError(
                "DecoderBlock expects exactly one skip connection tensor."
            )
        skip: torch.Tensor = skips[0]
        if x.shape[0] != skip.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x batch {x.shape[0]}, skip batch {skip.shape[0]}"
            )
        logger.debug(
            f"DecoderBlock input: {x.shape}, skip: {skip.shape}, expected output: {self.out_channels} channels"
        )
        x = cast(torch.Tensor, self.upsample(x))
        x = cast(torch.Tensor, self.up_conv(x))
        if self._skip_channels[0] == 0:
            x = cast(torch.Tensor, self.cbam(x))
            expected_channels: int = self.conv1.in_channels
            actual_channels: int = x.size(1)
            if actual_channels != expected_channels:
                raise ValueError(
                    "Critical channel mismatch in DecoderBlock: expected "
                    f"{expected_channels}, got {actual_channels}. This indicates a bug."
                )
            x = cast(torch.Tensor, self.conv1(x))
            x = cast(torch.Tensor, self.bn1(x))
            x = cast(torch.Tensor, self.relu1(x))
            x = cast(torch.Tensor, self.conv2(x))
            x = cast(torch.Tensor, self.bn2(x))
            x = cast(torch.Tensor, self.relu2(x))
            return x
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            h_factor: float = skip.shape[2] / x.shape[2]
            w_factor: float = skip.shape[3] / x.shape[3]
            if not (h_factor.is_integer() and w_factor.is_integer()):
                raise ValueError(
                    "Spatial upsampling factor must be integer. "
                    f"Got x: {x.shape[2:]} -> skip: {skip.shape[2:]} "
                    f"(h_factor={h_factor}, w_factor={w_factor})"
                )
            target_size: tuple[int, int] = (skip.shape[2], skip.shape[3])
            x = cast(
                torch.Tensor,
                F.interpolate(
                    x, size=target_size, mode="bilinear", align_corners=False
                ),
            )
        if x.shape[2:] != skip.shape[2:]:
            raise ValueError(
                f"Spatial dimension mismatch after upsampling: x {x.shape[2:]}, skip {skip.shape[2:]}"
            )
        try:
            x = torch.cat([x, skip], dim=1)
        except RuntimeError as e:
            logger.error(
                f"torch.cat failed! x shape: {x.shape}, skip shape: {skip.shape}. Error: {e}"
            )
            raise e
        x = cast(torch.Tensor, self.cbam(x))
        expected_channels = self.conv1.in_channels
        actual_channels = x.size(1)
        if actual_channels != expected_channels:
            raise ValueError(
                "Critical channel mismatch in DecoderBlock: expected "
                f"{expected_channels}, got {actual_channels}. This indicates a bug."
            )
        x = cast(torch.Tensor, self.conv1(x))
        x = cast(torch.Tensor, self.bn1(x))
        x = cast(torch.Tensor, self.relu1(x))
        x = cast(torch.Tensor, self.conv2(x))
        x = cast(torch.Tensor, self.bn2(x))
        x = cast(torch.Tensor, self.relu2(x))
        return x

    @property
    def skip_channels(self) -> list[int]:
        return self._skip_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels


class DecoderBlockAlias(DecoderBlock):
    """Alias; kept for potential registry clarity."""

    pass


__all__ = ["DecoderBlockConfig", "DecoderBlock", "DecoderBlockAlias"]
