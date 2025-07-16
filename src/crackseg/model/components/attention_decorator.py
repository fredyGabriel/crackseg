"""
Attention Decorator Pattern.

This module provides a decorator pattern for adding attention mechanisms
to decoders while preserving their interface and behavior.
"""

from typing import override

import torch
from torch import nn

from crackseg.model.base import DecoderBase


class AttentionDecorator(DecoderBase):
    """
    Decorator that applies an attention module after a decoder's processing.

    This class wraps a decoder with an attention module while preserving
    the decoder's interface, including support for skip_connections
    and all relevant attributes.

    Args:
        decoder (DecoderBase): The decoder to wrap
        attention_module (nn.Module): The attention module to apply after
        decoding

    Example:
        >>> decoder = CNNDecoder(...)
        >>> cbam = CBAM(in_channels=decoder.out_channels, ...)
        >>> attention_decoder = AttentionDecorator(decoder, cbam)
        >>> output = attention_decoder(x, skip_connections)
    """

    def __init__(
        self, decoder: DecoderBase, attention_module: nn.Module
    ) -> None:
        # Inicializar con los mismos parámetros que el decoder original
        super().__init__(
            in_channels=decoder.in_channels,
            skip_channels=decoder.skip_channels,
        )
        self.decoder = decoder
        self.attention = attention_module

    @override
    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply decoder and then the attention module.

        Args:
            x: Input tensor
            skips: Skip connections from encoder

        Returns:
            Output after applying decoder and attention
        """
        # First apply decoder with skip connections
        decoded = self.decoder(x, skips)
        # Then apply attention module
        return self.attention(decoded)

    @property
    @override
    def out_channels(self) -> int:
        """
        Return the number of output channels from the decorated decoder.
        Required implementation of abstract property from DecoderBase.
        """
        return self.decoder.out_channels

    @property
    @override
    def skip_channels(self) -> list[int]:
        """
        Return the skip channels list from the decorated decoder.
        Overrides the base implementation to get the current value from
        the decorated decoder.
        """
        return self.decoder.skip_channels
