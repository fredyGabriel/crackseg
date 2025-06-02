import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

# Import base classes
from src.model.base.abstract import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)

# Import reusable components
from src.model.components.convlstm import ConvLSTM, ConvLSTMConfig

logger = logging.getLogger(__name__)


@dataclass
class ConvLSTMBottleneckConfig:
    """Configuration for ConvLSTMBottleneck."""

    hidden_dim: int
    kernel_size: tuple[int, int] | list[tuple[int, int]]
    num_layers: int
    kernel_expected_dims: int
    num_dims_image: int
    bias: bool = True
    batch_first: bool = True
    return_all_layers: bool = False


# Definimos un bloque encoder simple
class SimpleEncoderBlock(nn.Module):
    """
    Bloque básico del encoder CNN.
    Consiste en dos capas convolucionales y una capa de pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
        use_pool: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool2d(pool_size)

        self._out_channels = out_channels
        self._skip_channels = [out_channels]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass a través del bloque encoder.

        Args:
            x: Tensor de entrada.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Tensor de salida después del pooling.
                - Lista con el tensor de skip connection (antes del pooling).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Guardar el skip connection antes del pooling
        skip = x

        # Aplicar pooling si está activado
        if self.use_pool:
            x = self.pool(x)

        return x, [skip]

    @property
    def out_channels(self) -> int:
        """Número de canales de salida del bloque."""
        return self._out_channels

    @property
    def skip_channels(self) -> list[int]:
        """Lista de canales de los skip connections."""
        return self._skip_channels


class CNNEncoder(EncoderBase):
    """
    CNN Encoder for UNet architecture.

    Consists of multiple encoder blocks with progressive downsampling
    and feature channel expansion.
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
        self._skip_channels: list[int] = []
        current_channels = in_channels

        for i in range(depth):
            out_channels = base_filters * (2**i)
            use_pool = True
            block = SimpleEncoderBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                pool_size=pool_size,
                use_pool=use_pool,
            )
            self.encoder_blocks.append(block)
            self._skip_channels.append(out_channels)
            current_channels = out_channels

        self._out_channels = current_channels

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Final feature map (output of the last block after pooling).
                - List of skip connection tensors (pre-pooled, high-to-low res)
        """
        skip_connections: list[torch.Tensor] = []
        output = x
        for block in self.encoder_blocks:
            output, skips = block(output)
            skip_connections.extend(skips)
        return output, skip_connections

    @property
    def out_channels(self) -> int:
        """Number of channels in the final output tensor (after last pool)."""
        return self._out_channels

    @property
    def skip_channels(self) -> list[int]:
        """List of channels for each skip connection (high-res to low-res)."""
        return self._skip_channels

    def get_feature_info(self) -> list[dict[str, Any]]:
        """
        Get information about feature maps produced by the encoder.

        Returns:
            List[Dict[str, Any]]: Information about each feature map,
                                 including channels and reduction factor.
        """
        feature_info: list[dict[str, Any]] = []

        # Calculate reduction factors for each stage
        # CNN encoder with pooling has reduction factor 2^stage
        pool_factor = 2  # Default pool_size is 2

        for i, channels in enumerate(self._skip_channels):
            reduction_factor = pool_factor ** (i + 1)
            feature_info.append(
                {
                    "channels": channels,
                    "reduction": reduction_factor,
                    "stage": i,
                }
            )

        # Add bottleneck info (final output)
        bottleneck_reduction = pool_factor ** len(self._skip_channels)
        feature_info.append(
            {
                "channels": self._out_channels,
                "reduction": bottleneck_reduction,
                "stage": len(self._skip_channels),
            }
        )

        return feature_info

    @property
    def feature_info(self) -> list[dict[str, Any]]:
        """Information about output features for each stage.

        Returns:
            List of dictionaries, each containing:
                - 'channels': Number of output channels
                - 'reduction': Spatial reduction factor from input
                - 'stage': Stage index
        """
        return self.get_feature_info()


class ConvLSTMBottleneck(BottleneckBase):
    """
    Bottleneck using ConvLSTM.
    """

    def __init__(self, in_channels: int, config: ConvLSTMBottleneckConfig):
        super().__init__(in_channels)
        self._out_channels = config.hidden_dim
        self.num_dims_image = config.num_dims_image

        convlstm_core_config = ConvLSTMConfig(
            hidden_dim=config.hidden_dim,
            kernel_size=config.kernel_size,
            num_layers=config.num_layers,
            kernel_expected_dims=config.kernel_expected_dims,
            batch_first=config.batch_first,
            bias=config.bias,
            return_all_layers=config.return_all_layers,
        )

        self.convlstm = ConvLSTM(
            input_dim=in_channels,
            config=convlstm_core_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # ConvLSTM expects input shape (batch, seq, channels, height, width)
        # Add sequence dimension if it doesn't exist
        if x.dim() == self.num_dims_image:
            x = x.unsqueeze(1)

        # Forward pass through ConvLSTM
        output, _ = self.convlstm(x)

        # Remove sequence dimension
        output = output.squeeze(1)
        return cast(torch.Tensor, output)

    @property
    def out_channels(self) -> int:
        """Number of channels in the output tensor."""
        return self._out_channels


class CNNConvLSTMUNet(UNetBase):
    """
    Complete U-Net architecture combining CNNEncoder, a Bottleneck, and
    CNNDecoder.

    While named CNNConvLSTMUNet, it can accept any BottleneckBase compliant
    module.
    """

    def __init__(
        self,
        encoder: EncoderBase,
        bottleneck: BottleneckBase,
        decoder: DecoderBase,
    ):
        # Validación flexible para el decoder
        if not (
            hasattr(decoder, "forward")
            and callable(getattr(decoder, "forward", None))
            and hasattr(decoder, "out_channels")
        ):
            raise TypeError(
                f"Expected DecoderBase, got {type(decoder)}. "
                "If using a wrapper, it must implement 'forward' and "
                "'out_channels'."
            )

        # Initialize the base class (performs further validation)
        super().__init__(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder
        )

        # Activación final para segmentación binaria
        self.final_activation = nn.Sigmoid()

        logger.info("CNNConvLSTMUNet inicializado con:")
        logger.info(f"  Encoder: {encoder}")
        logger.info(f"  Bottleneck: {bottleneck}")
        logger.info(f"  Decoder: {decoder}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the complete U-Net model.

        Args:
            x (torch.Tensor): Input tensor (batch of images).

        Returns:
            torch.Tensor: Output segmentation map.
        """
        assert self.encoder is not None, "Encoder is not initialized."
        assert self.bottleneck is not None, "Bottleneck is not initialized."
        assert self.decoder is not None, "Decoder is not initialized."
        encoder_output, skips = self.encoder(x)
        bottleneck_output = self.bottleneck(encoder_output)
        reversed_skips = list(reversed(skips))

        logger.debug(
            f"Encoder skip channels (HIGH->LOW): {[s.shape[1] for s in skips]}"
        )
        logger.debug(
            "Decoder skip channels (LOW->HIGH): %s",
            [s.shape[1] for s in reversed_skips],
        )

        decoder_output = self.decoder(bottleneck_output, reversed_skips)
        output = self.final_activation(decoder_output)
        return cast(torch.Tensor, output)


# Final architecture assembled.
# Next steps involve configuration, registration (Task 16.7),
# and comprehensive testing (Task 16.8).
