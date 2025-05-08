import torch
import torch.nn as nn
from typing import List, Tuple

# Import base classes
from src.model.base.abstract import (
    EncoderBase, BottleneckBase, DecoderBase, UNetBase
)
# Import reusable components
from src.model.encoder.cnn_encoder import EncoderBlock
from src.model.components.convlstm import ConvLSTM
from src.model.decoder.cnn_decoder import DecoderBlock

# Import registries
# from src.model.factory import (
#     encoder_registry, bottleneck_registry, decoder_registry
# )

# No registry needed here, registration happens in specific component files


# Register with specific name
# @encoder_registry.register(name="CNNEncoderForConvLSTM")
class CNNEncoder(EncoderBase):
    """
    CNN Encoder for the CNN-ConvLSTM U-Net architecture.

    Uses a series of EncoderBlocks to downsample the input and extract
    features.
    Matches the structure of the existing CNNEncoder but placed within this
    architecture file for clarity as it's part of this specific U-Net variant.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        base_filters (int): Number of filters in the first convolutional layer.
            Subsequent layers double the filters. Default: 64.
        depth (int): Number of downsampling blocks (encoder depth). Default: 5.
        kernel_size (int): Convolution kernel size for encoder blocks.
            Default: 3.
        pool_size (int): Pooling factor for downsampling. Default: 2.
        # BatchNorm is handled within EncoderBlock
    """
    def __init__(
        self,
        in_channels: int,
        base_filters: int = 64,
        depth: int = 5,
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__(in_channels)

        if depth < 1:
            raise ValueError("Encoder depth must be at least 1.")

        self.encoder_blocks = nn.ModuleList()
        self._skip_channels: List[int] = []  # Type hint added for clarity
        current_channels = in_channels

        for i in range(depth):
            out_channels = base_filters * (2**i)
            # Pooling is used in all blocks except potentially the last one
            # depending on architecture, but standard U-Net pools at each
            # stage.
            use_pool = True  # Standard U-Net pools at each depth level

            block = EncoderBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Calculate padding for 'same' conv
                pool_size=pool_size,
                use_pool=use_pool,
            )
            self.encoder_blocks.append(block)
            # Skip channels are the output channels *before* pooling
            self._skip_channels.append(out_channels)
            current_channels = out_channels

        # The final output channels of the encoder are the output channels
        # of the last block.
        # Note: EncoderBase expects out_channels to be the final output before
        # bottleneck, which is the output of the last block *after* pooling.
        # However, our current EncoderBlock/CNNEncoder returns the channels
        # *before* the final implied pool (consistent with its skip channel).
        # This might need adjustment depending on the Bottleneck
        # implementation.
        # For now, stick to the pattern: output channels = last block's
        # out_channels.
        self._out_channels = current_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                List[torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Final feature map (output of the last block after pooling).
                - List of skip connection tensors (pre-pooled, high-to-low res
                ).
        """
        skip_connections: List[torch.Tensor] = []
        output = x  # Start with the input
        for block in self.encoder_blocks:
            # EncoderBlock returns (output_after_pool, [skip_before_pool])
            # Pass the output of the previous block
            output, skip_list = block(output)
            if skip_list:
                skip_connections.append(skip_list[0])

        # The final 'output' is the result after the last block's pooling
        return output, skip_connections

    @property
    def out_channels(self) -> int:
        """Number of channels in the final output tensor (after last pool)."""
        return self._out_channels

    @property
    def skip_channels(self) -> List[int]:
        """List of channels for each skip connection (high-res to low-res)."""
        return self._skip_channels


# Register bottleneck
# @bottleneck_registry.register(name="ConvLSTMBottleneck")
class ConvLSTMBottleneck(BottleneckBase):
    """
    Bottleneck for the CNN-ConvLSTM U-Net using ConvLSTM layers.

    Processes the feature map from the encoder using ConvLSTM to capture
    potential temporal dependencies or enhance feature representation before
    passing to the decoder.

    Args:
        in_channels (int): Number of channels from the encoder output.
        hidden_dim (int | list[int]): Hidden dimensions for ConvLSTM layers.
        kernel_size (tuple | list): Kernel sizes for ConvLSTM layers.
        num_layers (int): Number of ConvLSTM layers in the bottleneck.
        bias (bool): Bias flag for ConvLSTM cells.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int | list[int],
        kernel_size: tuple[int, int] | list[tuple[int, int]],
        num_layers: int = 1,
        bias: bool = True
    ):
        super().__init__(in_channels)

        # If hidden_dim is a list, the last element determines the output
        # channels
        if isinstance(hidden_dim, list):
            if not hidden_dim:
                raise ValueError("hidden_dim list cannot be empty")
            self._out_channels = hidden_dim[-1]
        else:
            self._out_channels = hidden_dim

        self.bottleneck_convlstm = ConvLSTM(
            input_dim=in_channels, hidden_dim=hidden_dim,
            kernel_size=kernel_size, num_layers=num_layers,
            batch_first=True, bias=bias, return_all_layers=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvLSTM bottleneck.

        Args:
            x (torch.Tensor): Input tensor from encoder (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor (B, Cout, H, W).
        """
        # ConvLSTM expects input as (B, T, C, H, W).
        # Treat the spatial map as a sequence of length T=1.
        b, c, h, w = x.shape
        x_seq = x.unsqueeze(1)  # Add time dimension: (B, 1, C, H, W)

        # Pass through ConvLSTM
        # layer_output_list contains output of last layer:
        # [(B, T, Cout, H, W)]
        # last_state_list contains final (h, c) states for each layer
        layer_output_list, _ = self.bottleneck_convlstm(x_seq)

        # Get the output of the last layer and remove the time dimension (T=1)
        # Output shape: (B, 1, Cout, H, W) -> (B, Cout, H, W)
        output = layer_output_list[0].squeeze(1)

        return output

    @property
    def out_channels(self) -> int:
        """Number of output channels from the bottleneck."""
        return self._out_channels


# Register with specific name
# @decoder_registry.register(name="CNNDecoderForConvLSTM")
class CNNDecoder(DecoderBase):
    """
    CNN Decoder for the CNN-ConvLSTM U-Net architecture.

    Uses a series of DecoderBlocks to upsample features and merge them
    with skip connections from the encoder.
    Mirrors the structure of the existing CNNDecoder.

    Args:
        in_channels (int): Channels from the bottleneck.
        skip_channels_list (List[int]): List of channels for each skip
            connection from the encoder, ordered high-res to low-res.
        out_channels (int): Number of final output channels (e.g., classes).
        depth (int): Number of decoder blocks (should match encoder depth).
        kernel_size (int): Convolution kernel size for decoder blocks.
        upsample_mode (str): Upsampling mode ('bilinear', 'nearest').
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels_list: List[int],
                 out_channels: int = 1,
                 depth: int = 5,
                 kernel_size: int = 3,
                 upsample_mode: str = 'bilinear'):

        # DecoderBase expects skip channels ordered low-res to high-res
        reversed_skip_channels = list(reversed(skip_channels_list))
        super().__init__(in_channels, skip_channels=reversed_skip_channels)

        if len(skip_channels_list) != depth:
            # Use a formatted string for better readability
            msg = (
                f"Length of skip_channels_list ({len(skip_channels_list)}) "
                f"must match decoder depth ({depth})"
            )
            raise ValueError(msg)

        self.decoder_blocks = nn.ModuleList()
        self._final_out_channels = out_channels

        current_channels = in_channels  # Use a separate var for iteration
        for i in range(depth):
            skip_ch = reversed_skip_channels[i]  # Low-res skip first
            # Output channels of the block often match the skip connection dims
            # This ensures consistency for the next block's input
            block_out_channels = skip_ch

            block = DecoderBlock(
                in_channels=current_channels,
                skip_channels=skip_ch,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                upsample_scale_factor=2,  # Assuming standard 2x upsampling
                upsample_mode=upsample_mode
            )
            self.decoder_blocks.append(block)
            current_channels = block_out_channels  # Update for next block

        # Final 1x1 convolution to map to the desired number of output channels
        self.final_conv = nn.Conv2d(
            current_channels, self._final_out_channels, kernel_size=1
        )

    def forward(
        self, x: torch.Tensor, skips: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor from the bottleneck.
            skips (List[torch.Tensor]): List of skip connection tensors from
                the encoder (ordered high-res to low-res).

        Returns:
            torch.Tensor: Final segmentation map.
        """
        if len(skips) != len(self.decoder_blocks):
            # Use a formatted string for better readability
            msg = (
                f"Number of skips ({len(skips)}) must match decoder depth "
                f"({len(self.decoder_blocks)})"
            )
            raise ValueError(msg)

        # Skips are high-res to low-res, need to reverse for decoder blocks
        reversed_skips = list(reversed(skips))

        output = x  # Start with bottleneck output
        for i, block in enumerate(self.decoder_blocks):
            # Pass feature map and corresponding skip connection
            # (low-res first)
            skip_connection = reversed_skips[i]
            output = block(output, [skip_connection])

        # Apply final conv to get desired output channels
        output = self.final_conv(output)
        return output

    @property
    def out_channels(self) -> int:
        """Number of output channels from the decoder."""
        # Return the actual final output channels
        return self._final_out_channels


# Removed duplicate import of UNetBase


# No need to register the final UNet class itself if using create_unet factory
class CNNConvLSTMUNet(UNetBase):
    """
    Complete U-Net architecture combining CNNEncoder, a Bottleneck, and
    CNNDecoder.

    While named CNNConvLSTMUNet, it can accept any BottleneckBase compliant
    module.

    Args:
        encoder (CNNEncoder): The CNN encoder component.
        bottleneck (BottleneckBase): The bottleneck component
        (e.g., ConvLSTM or ASPP).
        decoder (CNNDecoder): The CNN decoder component.
    """
    def __init__(self, encoder: CNNEncoder, bottleneck: BottleneckBase,
                 decoder: CNNDecoder):
        # Validate component types
        if not isinstance(encoder, EncoderBase):
            raise TypeError(f"Expected EncoderBase, got {type(encoder)}")
        if not isinstance(bottleneck, BottleneckBase):
            raise TypeError(f"Expected BottleneckBase, got {type(bottleneck)}")
        # Validación flexible para el decoder
        if not (
            isinstance(decoder, DecoderBase) or (
                hasattr(decoder, 'forward') and
                callable(
                    getattr(decoder, 'forward', None)
                ) and
                hasattr(decoder, 'out_channels')
            )
        ):
            raise TypeError(
                f"Expected DecoderBase, got {type(decoder)}. "
                "If using a wrapper, it must implement 'forward' and "
                "'out_channels'."
            )

        # Initialize the base class (performs further validation)
        super().__init__(encoder=encoder, bottleneck=bottleneck,
                         decoder=decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the complete U-Net model.

        Args:
            x (torch.Tensor): Input tensor (batch of images).

        Returns:
            torch.Tensor: Output segmentation map.
        """
        encoder_output, skips = self.encoder(x)
        bottleneck_output = self.bottleneck(encoder_output)
        decoder_output = self.decoder(bottleneck_output, skips)
        return decoder_output

# Final architecture assembled.
# Next steps involve configuration, registration (Task 16.7),
# and comprehensive testing (Task 16.8).
