from typing import Any

import torch
from torch import nn

from src.model.base.abstract import EncoderBase

# Import the specific registry
from src.model.factory.registry_setup import encoder_registry


# No longer registering with the factory registry
# @encoder_registry.register("EncoderBlock")
class EncoderBlock(EncoderBase):
    """
    CNN Encoder block for U-Net architecture.

    Consists of two Conv2d layers (with BatchNorm and ReLU) and optional
    MaxPool2d. Returns both the output feature map and the pre-pooled feature
    map for skip connections.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
        use_pool: bool = True,
    ):
        """
        Initialize EncoderBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size. Default: 3.
            padding (int): Padding for convolutions. Default: 1.
            pool_size (int): Pooling size. Default: 2.
            use_pool (bool): Whether to use MaxPool2d. Default: True.
        """
        super().__init__(in_channels)
        self._out_channels = out_channels
        self.use_pool = use_pool
        self.pool_size = pool_size
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
        self.pool = nn.MaxPool2d(pool_size) if use_pool else None

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: (output, [skip_connection]
            )
                - output: Output tensor after convolutions and pooling (if used
                ).
                - [skip_connection]: List with the pre-pooled feature map for
                skip connections.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        skip = x  # Pre-pooled feature map for skip connection
        if self.use_pool:
            assert self.pool is not None
            x = self.pool(x)
        return x, [skip]

    @property
    def out_channels(self) -> int:
        """Number of output channels after convolutions."""
        return self._out_channels

    @property
    def skip_channels(self) -> list[int]:
        """List with the number of channels for each skip connection
        (single block)."""
        return [self._out_channels]

    def get_feature_info(self) -> list[dict[str, Any]]:
        """
        Get information about feature maps produced by the encoder block.

        Returns:
            List[Dict[str, Any]]: Information about each feature map,
                                 including channels and reduction factor.
        """
        # Import locally to avoid circular imports
        from src.model.encoder.feature_info_utils import (
            create_feature_info_entry,
        )

        # For a single encoder block, reduction depends on whether pooling
        # is used
        reduction_factor = self.pool_size if self.use_pool else 1

        return [
            create_feature_info_entry(
                channels=self._out_channels,
                reduction=reduction_factor,
                stage=0,
                name="encoder_block",
            )
        ]

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


# Registration
# @encoder_registry.register(name="CNNEncoderBlock")
class EncoderBlockAlias(EncoderBlock):
    # Optional: Alias for clarity in registry if needed
    pass


@encoder_registry.register("CNNEncoder")
class CNNEncoder(EncoderBase):
    """
    Standard CNN Encoder for U-Net style architectures.

    Composed of multiple EncoderBlocks with increasing feature channels
    and downsampling via pooling.
    """

    def __init__(  # noqa: PLR0913
        self,  # noqa: PLR0913
        in_channels: int,
        init_features: int = 64,
        depth: int = 4,
        pool_size: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the CNNEncoder.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            init_features (int): Number of feature channels in the first layer.
                                 Default: 64.
            depth (int): Number of encoder blocks (levels of downsampling).
                         Default: 4.
            pool_size (int): Pooling factor. Default: 2.
            kernel_size (int): Convolution kernel size. Default: 3.
            padding (int): Convolution padding. Default: 1.
        """
        super().__init__(in_channels)
        self.depth = depth
        self.pool_size = pool_size
        self.encoder_blocks = nn.ModuleList()
        self._skip_channels: list[int] = []

        channels = in_channels
        features = init_features
        for _ in range(depth):
            block = EncoderBlock(
                in_channels=channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                pool_size=pool_size,
                use_pool=True,  # Always pool in standard U-Net encoder blocks
            )
            self.encoder_blocks.append(block)
            self._skip_channels.append(features)  # Store pre-pooled channels
            channels = features
            features *= 2

        self._out_channels = channels  # Channels after last block (pre-pool)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Final feature map (output of the last block).
                - List of skip connection tensors (from higher to lower res).
        """
        skip_connections: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x, skips = block(x)
            # EncoderBlock returns list w/ one skip, take the first
            # element
            skip_connections.append(skips[0])

        # Return the final output and the collected skips (in correct order)
        # Skips are already collected from high-res to low-res
        return x, skip_connections

    @property
    def out_channels(self) -> int:
        """Number of channels in the final output tensor."""
        # This should be the number of channels *after* the last pooling step
        # which is the input channels to the bottleneck
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
        # Import locally to avoid circular imports
        from src.model.encoder.feature_info_utils import (
            build_feature_info_from_channels,
        )

        # CNN encoder with pooling has standard reduction progression
        return build_feature_info_from_channels(
            skip_channels=self._skip_channels,
            out_channels=self._out_channels,
            base_reduction=self.pool_size,  # First stage reduction
            reduction_factor=self.pool_size,  # Each stage mult. by pool_size
            stage_names=[f"stage_{i}" for i in range(self.depth)]
            + ["bottleneck"],
        )

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
