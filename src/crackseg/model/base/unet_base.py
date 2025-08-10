from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .bottleneck_base import BottleneckBase
from .decoder_base import DecoderBase
from .encoder_base import EncoderBase


class UNetBase(torch.nn.Module, ABC):
    """Base class for UNet-style architectures."""

    encoder: EncoderBase | None
    bottleneck: BottleneckBase | None
    decoder: DecoderBase | None

    def __init__(
        self,
        encoder: EncoderBase | None,
        bottleneck: BottleneckBase | None,
        decoder: DecoderBase | None,
        *_,
        **__,
    ) -> None:
        super().__init__()
        self._validate_components(encoder, bottleneck, decoder)
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

    def _validate_encoder_bottleneck_channels(
        self, encoder: EncoderBase, bottleneck: BottleneckBase
    ) -> None:
        if encoder.out_channels != bottleneck.in_channels:
            raise ValueError(
                "Encoder output channels must match bottleneck input channels"
            )

    def _validate_bottleneck_decoder_channels(
        self, bottleneck: BottleneckBase, decoder: DecoderBase
    ) -> None:
        if bottleneck.out_channels != decoder.in_channels:
            raise ValueError(
                "Bottleneck output channels must match decoder input channels"
            )

    def _validate_encoder_decoder_skips(
        self, encoder: EncoderBase, decoder: DecoderBase
    ) -> None:
        encoder_skips = encoder.skip_channels
        expected_decoder_skips_for_comparison = list(
            reversed(decoder.skip_channels)
        )
        if len(encoder_skips) != len(expected_decoder_skips_for_comparison):
            raise ValueError(
                "Encoder and decoder skip channel counts must match"
            )
        for enc_ch, dec_ch in zip(
            encoder_skips, expected_decoder_skips_for_comparison, strict=False
        ):
            if enc_ch != dec_ch:
                raise ValueError(
                    "Encoder and decoder skip channels must align"
                )

    def _validate_components(
        self,
        encoder: EncoderBase | None,
        bottleneck: BottleneckBase | None,
        decoder: DecoderBase | None,
    ) -> None:
        is_dummy_registration = (
            type(self).__name__ == "DummyArchitectureForRegistration"
        )
        if not isinstance(encoder, EncoderBase) and not (
            is_dummy_registration and encoder is None
        ):
            raise TypeError(
                "encoder must be EncoderBase or None for dummy registration"
            )
        if not isinstance(bottleneck, BottleneckBase) and not (
            is_dummy_registration and bottleneck is None
        ):
            raise TypeError(
                "bottleneck must be BottleneckBase or None for dummy registration"
            )
        if encoder is not None and bottleneck is not None:
            self._validate_encoder_bottleneck_channels(encoder, bottleneck)
        if (
            bottleneck is not None
            and decoder is not None
            and hasattr(decoder, "in_channels")
        ):
            self._validate_bottleneck_decoder_channels(bottleneck, decoder)
        if (
            encoder is not None
            and decoder is not None
            and hasattr(decoder, "skip_channels")
            and hasattr(encoder, "skip_channels")
            and not is_dummy_registration
        ):
            self._validate_encoder_decoder_skips(encoder, decoder)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def in_channels(self) -> int:
        return self.encoder.in_channels if self.encoder is not None else 0

    @property
    def out_channels(self) -> int:
        return self.decoder.out_channels if self.decoder is not None else 0
