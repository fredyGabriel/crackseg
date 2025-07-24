import torch
from torch import nn

from crackseg.model.base.abstract import BottleneckBase
from crackseg.model.factory.registry_setup import bottleneck_registry


@bottleneck_registry.register("BottleneckBlock", force=True)
class BottleneckBlock(BottleneckBase):
    """
    CNN Bottleneck block for U-Net architecture.

    Consists of two Conv2d layers (with BatchNorm and ReLU) and a Dropout layer
    between them. Maintains spatial dimensions, increases channel depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.5,
    ):
        """
        Initialize BottleneckBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size. Default: 3.
            padding (int): Padding for convolutions. Default: 1.
            dropout (float): Dropout rate between convolutions. Default: 0.5.
        """
        super().__init__(in_channels)
        self._out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the bottleneck block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    @property
    def out_channels(self) -> int:
        """Number of output channels after convolutions."""
        return self._out_channels
