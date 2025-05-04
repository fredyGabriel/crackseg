import abc
import torch
import torch.nn as nn
from typing import List, Tuple


class EncoderBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for all encoder modules in the U-Net architecture.

    Encoders are responsible for downsampling the input image and extracting
    hierarchical features at different spatial resolutions. These features
    are often passed to the decoder via skip connections.
    """

    def __init__(self, in_channels: int):
        """
        Initializes the EncoderBase.

        Args:
            in_channels (int): Number of channels in the input tensor.
        """
        super().__init__()
        self.in_channels = in_channels

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                List[torch.Tensor]]:
        """
        Defines the forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor (e.g., batch of images) with shape
                              (batch_size, in_channels, height, width).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing:
                - Final output tensor of the encoder (bottleneck features).
                - List of intermediate feature maps (skip connections) from
                  different encoder stages, ordered from higher to lower
                  resolution.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_channels(self) -> int:
        """
        Number of channels in the final output tensor of the encoder.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def skip_channels(self) -> List[int]:
        """
        List of channel dimensions for each intermediate feature map
        (skip connection). The order should correspond to the list
        returned by the forward method.
        """
        raise NotImplementedError


class BottleneckBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for all bottleneck modules in the U-Net architecture.

    The bottleneck sits between the encoder and decoder, typically processing
    features at the lowest spatial resolution. It can perform further feature
    extraction or transformation before features are upsampled by the decoder.
    """

    def __init__(self, in_channels: int):
        """
        Initializes the BottleneckBase.

        Args:
            in_channels (int): Number of channels in the input tensor
                               (usually the output channels of the encoder).
        """
        super().__init__()
        self.in_channels = in_channels

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the bottleneck.

        Args:
            x (torch.Tensor): Input tensor from the encoder with shape
                              (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of the bottleneck, potentially with
                          different channel dimensions but often the same
                          spatial resolution as the input.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_channels(self) -> int:
        """
        Number of channels in the output tensor of the bottleneck.
        """
        raise NotImplementedError


class DecoderBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for all decoder modules in the U-Net architecture.

    Decoders are responsible for upsampling features from the bottleneck
    and merging them with corresponding features from the encoder via skip
    connections. The goal is to gradually reconstruct the segmentation map
    at the original input resolution.
    """

    def __init__(self, in_channels: int, skip_channels: List[int]):
        """
        Initializes the DecoderBase.

        Args:
            in_channels (int): Number of channels in the input tensor from the
                               bottleneck.
            skip_channels (List[int]): List of channel dimensions for skip
                                       connection tensors from the encoder,
                                       ordered based on implementation needs.
        """
        super().__init__()
        self.in_channels = in_channels
        # Store skips in a private attribute
        self._skip_channels = skip_channels

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]
                ) -> torch.Tensor:
        """
        Defines the forward pass for the decoder.

        Args:
            x (torch.Tensor): Input tensor from the bottleneck with shape
                              (batch_size, in_channels, height, width).
            skips (List[torch.Tensor]): List of intermediate feature maps
                                       (skip connections) from the encoder.
                                       Order should match `skip_channels` list.

        Returns:
            torch.Tensor: Output tensor (e.g., segmentation logits),
                          shape (batch_size, out_channels, H, W).
        """
        raise NotImplementedError

    @property
    def skip_channels(self) -> List[int]:
        """List of skip connection channels (order depends on implementation).
        """
        if hasattr(self, '_skip_channels') and self._skip_channels is not None:
            return self._skip_channels
        raise AttributeError(
            "DecoderBase has not properly initialized _skip_channels"
        )

    @property
    @abc.abstractmethod
    def out_channels(self) -> int:
        """
        Number of channels in the final output tensor of the decoder.
        Usually corresponds to the number of segmentation classes.
        """
        raise NotImplementedError


class UNetBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for U-Net architectures.

    Integrates encoder, bottleneck, and decoder components into a complete
    segmentation model. Ensures proper component compatibility and data flow.
    """

    def __init__(self, encoder: EncoderBase, bottleneck: BottleneckBase,
                 decoder: DecoderBase):
        """
        Initializes the UNetBase.

        Args:
            encoder (EncoderBase): Encoder component for feature extraction.
            bottleneck (BottleneckBase): Bottleneck for feature processing.
            decoder (DecoderBase): Decoder for segmentation map generation.
        """
        super().__init__()
        self._validate_components(encoder, bottleneck, decoder)
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

    def _validate_components(self,
                             encoder: EncoderBase,
                             bottleneck: BottleneckBase,
                             decoder: DecoderBase) -> None:
        """
        Validates compatibility between components.
        """
        if not isinstance(encoder, EncoderBase):
            raise TypeError("encoder must be an instance of EncoderBase")
        if not isinstance(bottleneck, BottleneckBase):
            raise TypeError("bottleneck must be an instance of BottleneckBase")
        if not isinstance(decoder, DecoderBase):
            raise TypeError("decoder must be an instance of DecoderBase")

        if encoder.out_channels != bottleneck.in_channels:
            raise ValueError(
                f"Encoder output channels ({encoder.out_channels}) must match "
                f"bottleneck input channels ({bottleneck.in_channels})")

        if bottleneck.out_channels != decoder.in_channels:
            raise ValueError(
                f"Bottleneck output channels ({bottleneck.out_channels}) must "
                f"match decoder input channels ({decoder.in_channels})")

        # Get skip channels from encoder and decoder
        encoder_skips = encoder.skip_channels
        decoder_skips = decoder.skip_channels
        decoder_skips_reversed = list(reversed(decoder_skips))

        # Check if skip channel counts match
        if len(encoder_skips) != len(decoder_skips):
            raise ValueError(
                f"Encoder skip channel count ({len(encoder_skips)}) must match"
                f" decoder skip channel count ({len(decoder_skips)})")

        # Check if individual skip channel values match
        mismatch = False
        mismatched_encoder = []
        mismatched_decoder = []

        for i, (enc_skip, dec_skip) in enumerate(
                zip(encoder_skips, decoder_skips_reversed)):
            if enc_skip != dec_skip:
                mismatch = True
                mismatched_encoder.append((i, enc_skip))
                mismatched_decoder.append((i, dec_skip))

        if mismatch:
            error_msg = (
                f"Encoder skip channels {encoder_skips} must match "
                f"reversed decoder skip channels {decoder_skips_reversed}.\n"
                f"Mismatched encoder indices: {mismatched_encoder},\n"
                f"decoder indices: {mismatched_decoder}"
            )
            raise ValueError(error_msg)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the U-Net model.

        Args:
            x (torch.Tensor): Input tensor (batch of images) with shape
                             (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor (segmentation logits) with shape
                         (batch_size, num_classes, height, width).
        """
        raise NotImplementedError

    @property
    def in_channels(self) -> int:
        """Number of input channels."""
        return self.encoder.in_channels

    @property
    def out_channels(self) -> int:
        """Number of output channels (segmentation classes)."""
        return self.decoder.out_channels
