"""
Base UNet model implementation that integrates abstract components.

This module provides a concrete implementation of the UNetBase abstract
class that integrates EncoderBase, BottleneckBase, and DecoderBase
components into a complete UNet model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import hydra.utils  # Import hydra utils

from src.model.base import UNetBase, EncoderBase, BottleneckBase, DecoderBase


class BaseUNet(UNetBase):
    """
    Base UNet model implementation that integrates abstract components.

    This class provides a concrete implementation of the UNetBase abstract
    class, combining encoder, bottleneck and decoder components into a
    complete U-Net architecture for image segmentation tasks.
    """

    def __init__(
        self,
        encoder: EncoderBase,
        bottleneck: BottleneckBase,
        decoder: DecoderBase,
        final_activation: Optional[Dict[str, Any]] = None  # Expect config dict
    ):
        """
        Initialize the BaseUNet model.

        Args:
            encoder (EncoderBase): Encoder component for feature extraction.
            bottleneck (BottleneckBase): Bottleneck component for feature
                processing at the lowest resolution.
            decoder (DecoderBase): Decoder component for upsampling and
                generating the segmentation map.
            final_activation (Optional[Dict[str, Any]]): Optional activation
                function configuration (Hydra format with _target_).
                Default: None.
        """
        super().__init__(encoder, bottleneck, decoder)

        # Instantiate final activation from config if provided
        self.final_activation: Optional[nn.Module] = None
        if final_activation is not None:
            try:
                self.final_activation = hydra.utils.instantiate(
                    final_activation)
                if not isinstance(self.final_activation, nn.Module):
                    raise TypeError("final_activation config did not \
instantiate an nn.Module")
            except Exception as e:
                # Consider logging a warning or raising a more specific error
                print(f"Warning: Could not instantiate final_activation: {e}")
                self.final_activation = None  # Fallback to no activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                B: batch size, C: input channels, H: height, W: width.

        Returns:
            torch.Tensor: Output segmentation map of shape [B, O, H, W],
                where O is the number of output channels.
        """
        # Pass input through encoder to get features and skip connections
        features, skip_connections = self.encoder(x)

        # Pass features through bottleneck
        bottleneck_output = self.bottleneck(features)

        # Pass bottleneck output and skip connections through decoder
        output = self.decoder(bottleneck_output, skip_connections)

        # Apply final activation if specified
        if self.final_activation is not None:
            output = self.final_activation(output)

        return output

    def get_output_channels(self) -> int:
        """
        Get the number of output channels from the model.

        Returns:
            int: Number of output channels (from decoder).
        """
        return self.decoder.out_channels

    def get_input_channels(self) -> int:
        """
        Get the number of input channels the model expects.

        Returns:
            int: Number of input channels (from encoder).
        """
        return self.encoder.in_channels

    def summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the model architecture.

        Returns:
            Dict[str, Any]: Dictionary containing model summary information.
        """
        return {
            "model_type": "BaseUNet",
            "input_channels": self.get_input_channels(),
            "output_channels": self.get_output_channels(),
            "encoder_type": self.encoder.__class__.__name__,
            "bottleneck_type": self.bottleneck.__class__.__name__,
            "decoder_type": self.decoder.__class__.__name__,
            "has_final_activation": self.final_activation is not None,
            "final_activation_type": (
                self.final_activation.__class__.__name__
                if self.final_activation is not None else None
            )
        }
