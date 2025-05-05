import torch
import torch.nn as nn
import torch.nn.functional as F

# Registry for attention modules (for future integration in factory/config)
from src.model.registry import Registry

attention_registry = Registry(nn.Module, "Attention")


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style).

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for the hidden layer in the MLP.
            Must be > 0 and < in_channels.

    Example:
        >>> attn = ChannelAttention(in_channels=64, reduction=16)
        >>> x = torch.randn(8, 64, 32, 32)
        >>> out = attn(x)
        >>> out.shape  # (8, 64, 32, 32)
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        if reduction <= 0 or reduction >= in_channels:
            raise ValueError(
                f"reduction must be > 0 and < in_channels "
                f"(got {reduction} for {in_channels})"
            )
        self.in_channels = in_channels
        self.reduction = reduction
        hidden_channels = in_channels // reduction
        # Shared MLP: two linear layers with ReLU in between
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        b, c, h, w = x.size()
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Global max pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        # Shared MLP for both
        avg_attn = self.mlp(avg_pool)
        max_attn = self.mlp(max_pool)
        # Sum and apply sigmoid
        attn = self.sigmoid(avg_attn + max_attn).view(b, c, 1, 1)
        # Scale input
        return x * attn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Applies attention across spatial dimensions using channel-wise pooling
    and a convolutional layer.

    Args:
        kernel_size (int): Convolution kernel size (must be odd, e.g., 7).

    Example:
        >>> attn = SpatialAttention(kernel_size=7)
        >>> x = torch.randn(8, 64, 32, 32)
        >>> out = attn(x)
        >>> out.shape  # (8, 64, 32, 32)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(
                "kernel_size must be odd and >= 1"
            )
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatial attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # Channel-wise average pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise max pooling
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        # Convolution and sigmoid
        attn = self.sigmoid(self.conv(pooled))
        # Scale input
        return x * attn


@attention_registry.register(name="CBAM")
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Sequentially applies channel and spatial attention to the input tensor.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for channel attention (default: 16).
        kernel_size (int): Kernel size for spatial attention (default: 7).

    Example:
        >>> cbam = CBAM(in_channels=64, reduction=16, kernel_size=7)
        >>> x = torch.randn(8, 64, 32, 32)
        >>> out = cbam(x)
        >>> out.shape  # (8, 64, 32, 32)
    """
    def __init__(self, in_channels: int, reduction: int = 16,
                 kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CBAM.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
