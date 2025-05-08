from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.model.components.cbam import CBAM
from src.model.factory.registry_setup import decoder_registry
from src.model.base.abstract import DecoderBase

logger = logging.getLogger(__name__)

# @decoder_registry.register("DecoderBlock")


class DecoderBlock(DecoderBase):
    """
    CNN Decoder block for U-Net architecture.

    Upsamples the input features and concatenates them with skip connection
    features. Followed by two Conv2d layers (with BatchNorm and ReLU).
    Optionally applies CBAM attention after concatenation.
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding: int = 1,
        upsample_scale_factor: int = 2,
        upsample_mode: str = 'bilinear',
        use_cbam: bool = False
    ):
        """
        Initialize DecoderBlock.

        Args:
            in_channels (int): Number of input channels (from previous layer).
            skip_channels (int): Number of channels in the skip connection
                tensor.
            out_channels (Optional[int]): Number of output channels. If None,
                defaults to in_channels // 2.
            kernel_size (int): Convolution kernel size. Default: 3.
            padding (int): Padding for convolutions. Default: 1.
            upsample_scale_factor (int): Factor for upsampling. Default: 2.
            upsample_mode (str): Mode for nn.Upsample. Default: 'bilinear'.
            use_cbam (bool): Whether to use CBAM attention. Default: False.
        """
        super().__init__(in_channels, skip_channels=[skip_channels])
        self._out_channels = out_channels if out_channels is not None \
            else in_channels // 2
        self.upsample = nn.Upsample(
            scale_factor=upsample_scale_factor,
            mode=upsample_mode,
            align_corners=(True if upsample_mode == 'bilinear' else None)
        )
        self.up_conv = nn.Conv2d(
            in_channels, self._out_channels, kernel_size=1
        )
        concat_channels = self._out_channels + skip_channels
        if use_cbam:
            self.cbam = CBAM(in_channels=concat_channels)
        else:
            self.cbam = nn.Identity()
        self.conv1 = nn.Conv2d(
            concat_channels, self._out_channels, kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(self._out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            self._out_channels, self._out_channels, kernel_size,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(self._out_channels)
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
            torch.Tensor: Output tensor after upsampling, concatenation, CBAM
                          (optional), and convs.
        """
        if not skips or len(skips) != 1:
            raise ValueError(
                "DecoderBlock expects exactly one skip connection tensor."
            )
        skip = skips[0]
        x = self.upsample(x)
        x = self.up_conv(x)
        target_h, target_w = skip.shape[2:]
        current_h, current_w = x.shape[2:]
        if (current_h != target_h) or (current_w != target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear',
                              align_corners=False)
        try:
            x = torch.cat([x, skip], dim=1)
        except RuntimeError as e:
            logger.error(f"torch.cat failed! x shape: {x.shape}, "
                         f"skip shape: {skip.shape}. Error: {e}")
            raise e
        x = self.cbam(x)
        try:
            x = self.conv1(x)
        except RuntimeError as e:
            logger.error(f"self.conv1 failed! Input shape: {x.shape}, "
                         "expected channels: {self.conv1.in_channels}. "
                         "Error: {e}")
            raise e
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


@decoder_registry.register("CNNDecoder")
class CNNDecoder(DecoderBase):
    """
    Standard CNN Decoder for U-Net.
    Composed of multiple DecoderBlocks. Resizes final output to match the
    spatial dimensions of the highest-resolution skip connection.
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels_list: List[int],
                 out_channels: int = 1,
                 depth: int = 4,
                 target_size: Optional[Tuple[int, int]] = None,
                 upsample_scale_factor: int = 2,
                 upsample_mode: str = 'bilinear',
                 kernel_size: int = 3,
                 padding: int = 1,
                 use_cbam: bool = False
                 ):
        """
        Initialize the CNNDecoder.

        Args:
            in_channels (int): Channels from the bottleneck.
            skip_channels_list (List[int]): List of channels for each skip
                connection from the encoder, ordered from low resolution
                (closest to bottleneck) to high resolution (closest to input).
                Example: [512, 256, 128, 64] for depth=4.
                This follows the UNet contract where encoder.skip_channels is
                [64, 128, 256, 512] (high->low resolution).
            out_channels (int): Number of output segmentation classes.
                                Default: 1.
            depth (int): Number of decoder blocks (must match encoder depth).
                         Default: 4.
            target_size (Optional[Tuple[int, int]]): Target (H, W) for the
                final output. If None, the spatial size of the highest-res
                skip connection is used. Default: None.
            upsample_scale_factor (int): Upsampling factor. Default: 2.
            upsample_mode (str): Upsampling mode. Default: 'bilinear'.
            kernel_size (int): Convolution kernel size. Default: 3.
            padding (int): Convolution padding. Default: 1.
            use_cbam (bool): Whether to use CBAM attention in all decoder
                             blocks. Default: False.
        """
        super().__init__(in_channels, skip_channels=skip_channels_list)

        if depth != len(skip_channels_list):
            raise ValueError(
                f"Length of skip_channels_list must match depth. "
                f"Got skip_channels_list={len(skip_channels_list)}, "
                f"depth={depth}."
            )

        self.target_size = target_size

        self.decoder_blocks = nn.ModuleList()
        self._out_channels = out_channels

        channels = in_channels
        for i in range(depth):
            if i >= len(skip_channels_list):
                logger.error(f"Index {i} OOB for skip_channels_list "
                             f"(len {len(skip_channels_list)}) ")
                break
            skip_ch = skip_channels_list[i]
            block = DecoderBlock(
                in_channels=channels,
                skip_channels=skip_ch,
                kernel_size=kernel_size,
                padding=padding,
                upsample_scale_factor=upsample_scale_factor,
                upsample_mode=upsample_mode,
                use_cbam=use_cbam
            )
            self.decoder_blocks.append(block)
            channels = block.out_channels

        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]
                ) -> torch.Tensor:
        """
        Forward pass through the decoder.
        Ensures output spatial dimensions match the highest-res skip
        connection.
        """
        if not skips and self.target_size is None:
            raise ValueError("Decoder requires skip connections or "
                             "target_size to determine output size.")

        # Process through decoder blocks
        num_iterations = len(self.decoder_blocks)
        if len(skips) != num_iterations:
            raise ValueError(
                f"Number of skips must match number of decoder blocks. "
                f"Got skips={len(skips)}, expected={num_iterations}."
            )

        for i in range(num_iterations):
            if i >= len(skips):
                logger.error(f"Logic error: Trying to access skips index {i}.")
                break
            current_skip = skips[i]
            x = self.decoder_blocks[i](x, [current_skip])

        # *** Apply final resize to match target_size if specified, ***
        # *** otherwise match highest-res skip connection. ***
        if self.target_size:
            target_h, target_w = self.target_size
            logger.debug("Decoder resizing output to target_size: "
                         f"{(target_h, target_w)}")
        elif skips:  # Fallback to highest-res skip if target_size is None
            target_h, target_w = skips[0].shape[2:]
            logger.debug("Decoder resizing output to highest-res skip size: "
                         f"{(target_h, target_w)}")
        else:
            # Should not happen due to check at the beginning, but as safeguard
            target_h, target_w = x.shape[2:]  # Keep current size
            logger.warning("Decoder could not determine target size, keeping "
                           "current spatial dimensions.")

        current_h, current_w = x.shape[2:]
        if (current_h != target_h) or (current_w != target_w):
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
            logger.debug("Decoder resized output from "
                         f"{(current_h, current_w)} to target "
                         f"{(target_h, target_w)}")

        x = self.final_conv(x)
        return x

    @property
    def out_channels(self) -> int:
        """Number of output channels (segmentation classes)."""
        return self._out_channels
