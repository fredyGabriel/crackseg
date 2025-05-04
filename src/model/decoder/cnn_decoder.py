import torch
import torch.nn as nn
from typing import List
from src.model.base import DecoderBase


# No longer registering
# @decoder_registry.register("DecoderBlock")
class DecoderBlock(DecoderBase):
    """
    CNN Decoder block for U-Net architecture.

    Upsamples the input features and concatenates them with skip connection
    features. Followed by two Conv2d layers (with BatchNorm and ReLU).
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,  # Channels from the corresponding encoder skip
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        upsample_scale_factor: int = 2,
        upsample_mode: str = 'bilinear'
    ):
        """
        Initialize DecoderBlock.

        Args:
            in_channels (int): Number of input channels (from previous layer).
            skip_channels (int): Number of channels in the skip connection
                                 tensor.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size. Default: 3.
            padding (int): Padding for convolutions. Default: 1.
            upsample_scale_factor (int): Factor for upsampling. Default: 2.
            upsample_mode (str): Mode for nn.Upsample. Default: 'bilinear'.
        """
        # Pass the list of skip channels expected by the base class
        super().__init__(in_channels, skip_channels=[skip_channels])
        self._out_channels = out_channels

        # Upsampling followed by convolution
        self.upsample = nn.Upsample(
            scale_factor=upsample_scale_factor,
            mode=upsample_mode,
            align_corners=(True if upsample_mode == 'bilinear' else None)
        )
        # Convolution after upsampling to adjust channels
        self.up_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        # Convolution layers after concatenating upsampled and skip features
        # Input channels = (upsampled channels + skip channels)
        concat_channels = (in_channels // 2) + skip_channels
        self.conv1 = nn.Conv2d(
            concat_channels, out_channels, kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]
                ) -> torch.Tensor:
        """
        Forward pass for the decoder block.

        Args:
            x (torch.Tensor): Input tensor from the previous layer.
            skips (List[torch.Tensor]): List containing ONE skip connection
                                       tensor from the corresponding encoder
                                       stage.

        Returns:
            torch.Tensor: Output tensor after upsampling, concatenation, and
                          convs.
        """
        if not skips or len(skips) != 1:
            raise ValueError(
                "DecoderBlock expects exactly one skip connection tensor."
            )
        skip = skips[0]

        x = self.upsample(x)
        x = self.up_conv(x)

        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    @property
    def out_channels(self) -> int:
        """Number of output channels after convolutions."""
        return self._out_channels


# No longer registering
# @decoder_registry.register(name="CNNDecoderBlock")
class DecoderBlockAlias(DecoderBlock):
    # Optional: Alias for clarity in registry if needed
    pass


# No longer registering
# @decoder_registry.register("CNNDecoder")
class CNNDecoder(DecoderBase):
    """
    Standard CNN Decoder for U-Net.

    Composed of multiple DecoderBlocks that upsample features and merge them
    with skip connections from the encoder.
    """
    def __init__(self,
                 in_channels: int,             # From bottleneck
                 skip_channels_list: List[int],  # Enc: high-res -> low-res
                 out_channels: int = 1,        # Final output classes
                 depth: int = 4,
                 upsample_scale_factor: int = 2,
                 upsample_mode: str = 'bilinear',
                 kernel_size: int = 3,
                 padding: int = 1):
        """
        Initialize the CNNDecoder.

        Args:
            in_channels (int): Channels from the bottleneck.
            skip_channels_list (List[int]): List of channels for each skip
                connection from the encoder, ordered from high resolution
                (closest to input) to low resolution (closest to bottleneck).
                Example: [64, 128, 256, 512] for depth=4.
            out_channels (int): Number of output segmentation classes.
                                Default: 1.
            depth (int): Number of decoder blocks (must match encoder depth).
                         Default: 4.
            upsample_scale_factor (int): Upsampling factor. Default: 2.
            upsample_mode (str): Upsampling mode. Default: 'bilinear'.
            kernel_size (int): Convolution kernel size. Default: 3.
            padding (int): Convolution padding. Default: 1.
        """
        # Need skip channels in reverse order for DecoderBase init
        reversed_skip_channels = list(reversed(skip_channels_list))
        super().__init__(in_channels, skip_channels=reversed_skip_channels)

        if len(skip_channels_list) != depth:
            raise ValueError("Length of skip_channels_list must match depth")

        self.decoder_blocks = nn.ModuleList()
        self._out_channels = out_channels

        channels = in_channels  # Start with channels from bottleneck
        for i in range(depth):
            skip_ch = reversed_skip_channels[i]  # Match skip: low-res -> high
            # Calculate output channels for this block.
            # Assume it matches the skip channel size (common pattern).
            block_out_channels = skip_ch

            block = DecoderBlock(
                in_channels=channels,
                skip_channels=skip_ch,
                out_channels=block_out_channels,
                kernel_size=kernel_size,
                padding=padding,
                upsample_scale_factor=upsample_scale_factor,
                upsample_mode=upsample_mode
            )
            self.decoder_blocks.append(block)
            channels = block_out_channels  # Output becomes next input

        # Final 1x1 convolution to map to the desired output classes
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]
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
            raise ValueError("Number of skips must match decoder depth")

        # Skips are high-res to low-res, need to reverse for decoder
        reversed_skips = list(reversed(skips))

        for i, block in enumerate(self.decoder_blocks):
            # Pass feature map and corresponding skip connection
            x = block(x, [reversed_skips[i]])

        # Apply final convolution
        x = self.final_conv(x)
        return x

    @property
    def out_channels(self) -> int:
        """Number of output channels (segmentation classes)."""
        return self._out_channels
