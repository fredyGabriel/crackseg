"""Base U-Net implementation (core).

This module contains the production U-Net `BaseUNet` with:
- Component compatibility validation
- Optional final activation (direct module or Hydra config)
- Forward pass with skip connections
- Lightweight analysis helpers (summary/print/visualize)

Kept intentionally compact to comply with line-limit guardrails. Rich, long
documentation was removed from here and should live in user-facing docs.
"""

from __future__ import annotations

from io import StringIO
from typing import Any, cast

import hydra.utils
import torch
from torch import nn

from crackseg.model.base.abstract import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)
from crackseg.model.common import (
    count_parameters,
    estimate_memory_usage,
    estimate_receptive_field,
    get_layer_hierarchy,
    render_unet_architecture_diagram,
)


class BaseUNet(UNetBase):
    """U-Net with encoder, bottleneck, decoder and optional final activation."""

    def __init__(
        self,
        encoder: EncoderBase,
        bottleneck: BottleneckBase,
        decoder: DecoderBase,
        final_activation: nn.Module | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(encoder, bottleneck, decoder)

        # Final activation (module instance or Hydra config)
        self.final_activation: nn.Module | None = None
        if final_activation is not None:
            if isinstance(final_activation, nn.Module):
                self.final_activation = final_activation
            else:
                try:
                    instantiated = hydra.utils.instantiate(final_activation)
                    if not isinstance(instantiated, nn.Module):
                        raise TypeError(
                            "final_activation config did not yield nn.Module"
                        )
                    self.final_activation = instantiated
                except Exception:
                    # Fallback to no activation; errors are non-fatal for core
                    self.final_activation = None

        # Component presence
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.bottleneck is not None

        # Skip connection compatibility: encoder.skip_channels vs reversed decoder.skip_channels
        if hasattr(self.encoder, "skip_channels") and hasattr(
            self.decoder, "skip_channels"
        ):
            encoder_skip = list(self.encoder.skip_channels)
            decoder_skip = list(reversed(self.decoder.skip_channels))
            if encoder_skip != decoder_skip:
                raise ValueError(
                    "Encoder skip channels and decoder skip channels are incompatible. "
                    "Decoder skips must be the reverse of encoder skips."
                )

        # Channel compatibility: encoder.out_channels == bottleneck.in_channels
        if hasattr(self.encoder, "out_channels") and hasattr(
            self.bottleneck, "in_channels"
        ):
            if self.encoder.out_channels != self.bottleneck.in_channels:
                raise ValueError(
                    "Encoder out_channels must match bottleneck in_channels."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.encoder is not None
        assert self.bottleneck is not None
        assert self.decoder is not None

        features, skip_connections = self.encoder(x)
        bottleneck_output = self.bottleneck(features)
        reversed_skips = list(reversed(skip_connections))
        output = self.decoder(bottleneck_output, reversed_skips)

        if self.final_activation is not None:
            output = self.final_activation(output)

        return cast(torch.Tensor, output)

    def get_output_channels(self) -> int:
        assert self.decoder is not None
        return self.decoder.out_channels

    def get_input_channels(self) -> int:
        assert self.encoder is not None
        return self.encoder.in_channels

    # ------- Lightweight analysis helpers (kept concise) -------------------
    def summary(
        self, input_shape: tuple[int, ...] | None = None
    ) -> dict[str, Any]:
        trainable, non_trainable = count_parameters(self)
        total = trainable + non_trainable
        info = {
            "model_type": self.__class__.__name__,
            "input_channels": self.get_input_channels(),
            "output_channels": self.get_output_channels(),
            "encoder_type": self.encoder.__class__.__name__,
            "bottleneck_type": self.bottleneck.__class__.__name__,
            "decoder_type": self.decoder.__class__.__name__,
            "has_final_activation": self.final_activation is not None,
            "final_activation_type": (
                self.final_activation.__class__.__name__
                if self.final_activation is not None
                else None
            ),
        }
        summary_dict = {
            **info,
            "parameters": {
                "total": total,
                "trainable": trainable,
                "non_trainable": non_trainable,
                "trainable_percent": (
                    (trainable / total * 100) if total > 0 else 0
                ),
            },
            "receptive_field": estimate_receptive_field(self.encoder),
            "memory_usage": estimate_memory_usage(
                self, self.encoder, self.get_output_channels, input_shape
            ),
            "layer_hierarchy": get_layer_hierarchy(
                self.encoder,
                self.bottleneck,
                self.decoder,
                self.final_activation,
            ),
        }
        return summary_dict

    def print_summary(
        self,
        input_shape: tuple[int, ...] | None = None,
        file: Any | None = None,
        return_string: bool = False,
    ) -> str | None:
        stream = StringIO()
        target = (
            stream if return_string else (file or __import__("sys").stdout)
        )

        s = self.summary(input_shape)
        # Header
        print("\n" + "=" * 80, file=target)
        print("U-Net Model Summary", file=target)
        print("=" * 80, file=target)
        print(f"\nModel Type: {s['model_type']}", file=target)
        print(f"Input Channels: {s['input_channels']}", file=target)
        print(f"Output Channels: {s['output_channels']}", file=target)
        # Parameters
        p = s["parameters"]
        print(f"\nTotal Parameters: {p['total']:,}", file=target)
        print(
            f"Trainable Parameters: {p['trainable']:,} ({p['trainable_percent']:.2f}%)",
            file=target,
        )
        print(f"Non-trainable Parameters: {p['non_trainable']:,}", file=target)
        # Architecture
        print("\nArchitecture:", file=target)
        print(f"  Encoder: {s['encoder_type']}", file=target)
        print(f"  Bottleneck: {s['bottleneck_type']}", file=target)
        print(f"  Decoder: {s['decoder_type']}", file=target)
        if s["has_final_activation"]:
            print(
                f"  Final Activation: {s['final_activation_type']}",
                file=target,
            )
        # Hierarchy
        print("\nLayer Hierarchy:", file=target)
        print("-" * 80, file=target)
        print(
            f"{'Layer':<25} {'Type':<20} {'Parameters':>12} {'Input Ch':>10} {'Output Ch':>10}",
            file=target,
        )
        print("-" * 80, file=target)
        for layer in s["layer_hierarchy"]:
            layer_name = layer["name"]
            layer_type = layer.get("type", "")
            params = layer.get("params", 0)
            in_ch = layer.get("in_channels", "")
            out_ch = layer.get("out_channels", "")
            print(
                f"{layer_name:<25} {layer_type:<20} {params:>12,} {in_ch:>10} {out_ch:>10}",
                file=target,
            )
        print("-" * 80, file=target)
        print("=" * 80 + "\n", file=target)

        if return_string:
            return stream.getvalue()
        return None

    def visualize_architecture(
        self, filename: str | None = None, view: bool = False
    ) -> None:
        hierarchy = get_layer_hierarchy(
            self.encoder, self.bottleneck, self.decoder, self.final_activation
        )
        render_unet_architecture_diagram(hierarchy, filename, view)
