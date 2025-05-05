from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import BottleneckBase
from src.model.factory import bottleneck_registry


@bottleneck_registry.register(name="ASPPModule")
class ASPPModule(BottleneckBase):
    """Atrous Spatial Pyramid Pooling module for semantic segmentation
    networks.

    Applies multiple parallel atrous convolutions with different dilation rates
    to capture multi-scale contextual information. Includes:
    - Multiple parallel atrous convolutions with different dilation rates
    - A global average pooling branch to capture global context
    - A fusion layer to combine all branches
    - Dropout for regularization

    References:
        - DeepLabV3+: Encoder-Decoder with Atrous Separable
          Convolution
        - https://arxiv.org/abs/1802.02611
    """
    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        dilation_rates: List[int] = [1, 6, 12, 18],
        dropout_rate: float = 0.1,
        output_stride: int = 16
    ) -> None:
        """Initialize the ASPP module.

        Args:
            in_channels: Number of input channels
            output_channels: Number of output channels for each branch
            dilation_rates: List of dilation rates for parallel atrous convs
            dropout_rate: Dropout probability for regularization
            output_stride: Output stride of the network, affects dilations

        Raises:
            ValueError: If in_channels or output_channels are not positive
            ValueError: If dilation_rates is empty
            ValueError: If dropout_rate is not in range [0,1]
        """
        super().__init__(in_channels)
        if in_channels <= 0 or output_channels <= 0:
            raise ValueError("Channel dimensions must be positive integers")
        if not dilation_rates:
            raise ValueError("At least one dilation rate must be provided")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")

        # Guardar parámetros
        self._output_channels = output_channels
        self._dilation_rates = dilation_rates
        if output_stride != 16:
            self._dilation_rates = [
                rate * (16 // output_stride) for rate in dilation_rates
            ]
        self._dropout_rate = dropout_rate
        self._output_stride = output_stride

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolution branches
        self.branches = nn.ModuleList()
        for rate in self._dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        output_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels,
                output_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        # Nota: la expansión espacial se hará en el forward (próxima subtarea)

        # Final 1x1 projection after concatenation
        self.project = nn.Sequential(
            nn.Conv2d(
                output_channels * (len(self._dilation_rates) + 2),
                output_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else \
            nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ASPP module.

        Args:
            x: Input tensor of shape [batch, in_channels, height, width]
        Returns:
            Processed tensor of shape [batch, output_channels, height, width]
        """
        # Collect outputs from all branches
        outputs = [branch(x) for branch in self.branches]
        outputs.append(self.conv_1x1(x))
        # Global pooling branch: pool, upsample to input size
        pool = self.global_pool(x)
        pool_upsampled = F.interpolate(
            pool, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        outputs.append(pool_upsampled)
        # Concatenate along channel dimension
        x_cat = torch.cat(outputs, dim=1)
        # Project and apply dropout if training
        x_proj = self.project(x_cat)
        if self.training and self._dropout_rate > 0:
            x_proj = self.dropout(x_proj)
        return x_proj

    @property
    def out_channels(self) -> int:
        """Return the number of output channels.

        Returns:
            Number of output channels from this module
        """
        return self._output_channels
