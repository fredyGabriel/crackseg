"""Hybrid SwinV2 + ASPP + CNN U-Net architecture (core implementation).

This module contains the production implementation of `SwinV2CnnAsppUNet`.
Heavy narrative documentation and verbose debug logging have been removed to
keep the module compact and under line-limit guardrails. See user docs for
extended explanations.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch import nn

from crackseg.model.base.abstract import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)
from crackseg.model.components.aspp import ASPPModule
from crackseg.model.decoder.decoder_head import CNNDecoder, CNNDecoderConfig
from crackseg.model.encoder.swin_v2_adapter import SwinV2EncoderAdapter
from crackseg.model.factory.registry_setup import architecture_registry

logger = logging.getLogger(__name__)


@architecture_registry.register(
    name="SwinV2CnnAsppUNet",
    tags=["hybrid", "transformer", "aspp", "cnn", "swinv2"],
    force=True,
)
class SwinV2CnnAsppUNet(UNetBase):
    """Hybrid U-Net with SwinV2 encoder, ASPP bottleneck, and CNN decoder."""

    def __init__(
        self,
        encoder_cfg: dict[str, Any] | EncoderBase,
        bottleneck_cfg: dict[str, Any] | BottleneckBase,
        decoder_cfg: dict[str, Any] | DecoderBase,
        num_classes: int = 1,
        final_activation: str | None = "sigmoid",
    ) -> None:
        """Create hybrid architecture from configs or component instances."""
        # Encoder (config or instance)
        if isinstance(encoder_cfg, dict | DictConfig):
            if "in_channels" not in encoder_cfg:
                encoder_cfg["in_channels"] = 3
            target_img_size = encoder_cfg.get(
                "target_img_size"
            ) or encoder_cfg.get("img_size", 256)
            encoder = SwinV2EncoderAdapter(**encoder_cfg)
        else:
            encoder = encoder_cfg
            target_img_size = getattr(
                encoder, "target_img_size", None
            ) or getattr(encoder, "img_size", 256)

        # Bottleneck (config or instance)
        if isinstance(bottleneck_cfg, dict | DictConfig):
            bottleneck_in_channels = encoder.out_channels
            bottleneck = cast(
                BottleneckBase,
                ASPPModule(
                    in_channels=bottleneck_in_channels, **bottleneck_cfg
                ),
            )
        else:
            bottleneck = bottleneck_cfg

        # Decoder (config or instance)
        if isinstance(decoder_cfg, dict | DictConfig):
            decoder_in_channels = bottleneck.out_channels
            decoder_skip_channels = list(reversed(encoder.skip_channels))
            decoder_depth = len(decoder_skip_channels)

            params: dict[str, Any] = {}
            for key in (
                "use_cbam",
                "cbam_reduction",
                "upsample_mode",
                "kernel_size",
                "padding",
                "upsample_scale_factor",
            ):
                if key in decoder_cfg:
                    params[key] = decoder_cfg[key]

            decoder_config = CNNDecoderConfig(**params) if params else None

            decoder = cast(
                DecoderBase,
                CNNDecoder(
                    in_channels=decoder_in_channels,
                    skip_channels_list=decoder_skip_channels,
                    out_channels=num_classes,
                    target_size=(target_img_size, target_img_size),
                    depth=decoder_depth,
                    config=decoder_config,
                ),
            )
        else:
            decoder = decoder_cfg

        # Initialize UNetBase (validates compatibility)
        super().__init__(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder
        )

        # Final activation
        if final_activation:
            if final_activation.lower() == "sigmoid":
                self.final_activation_layer: nn.Module = nn.Sigmoid()
            elif final_activation.lower() == "softmax":
                self.final_activation_layer = nn.Softmax(dim=1)
            else:
                raise ValueError(
                    f"Unsupported final_activation: {final_activation}"
                )
        else:
            self.final_activation_layer = nn.Identity()

        logger.info("SwinV2CnnAsppUNet initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder → bottleneck → decoder → activation."""
        assert self.encoder is not None
        assert self.bottleneck is not None
        assert self.decoder is not None

        features, skip_connections = self.encoder(x)
        bottleneck_output = self.bottleneck(features)
        reversed_skips = list(reversed(skip_connections))
        decoder_output = self.decoder(bottleneck_output, reversed_skips)
        return self.final_activation_layer(decoder_output)


__all__ = ["SwinV2CnnAsppUNet"]
