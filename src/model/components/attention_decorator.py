"""
Attention Decorator Pattern.

This module provides a decorator pattern for adding attention mechanisms
to decoders while preserving their interface and behavior.
"""

from src.model.base import DecoderBase


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

    def __init__(self, decoder, attention_module):
        # Inicializar con los mismos parámetros que el decoder original
        super().__init__(
            in_channels=getattr(decoder, "in_channels", 0),
            skip_channels=getattr(decoder, "skip_channels", []),
        )
        self.decoder = decoder
        self.attention = attention_module

    def forward(self, x, skip_connections=None):
        """
        Apply decoder and then the attention module.

        Args:
            x: Input tensor
            skip_connections: Skip connections from encoder

        Returns:
            Output after applying decoder and attention
        """
        # First apply decoder with skip connections
        decoded = (
            self.decoder(x, skip_connections)
            if self.decoder is not None
            else None if self.decoder is not None else (None, None)
        )
        # Then apply attention module
        return self.attention(decoded)

    @property
    def out_channels(self):
        """
        Return the number of output channels from the decorated decoder.
        Required implementation of abstract property from DecoderBase.
        """
        return self.decoder.out_channels if self.decoder is not None else 0

    @property
    def skip_channels(self):
        """
        Return the skip channels list from the decorated decoder.
        Overrides the base implementation to get the current value from
        the decorated decoder.
        """
        return self.decoder.skip_channels if self.decoder is not None else 0
