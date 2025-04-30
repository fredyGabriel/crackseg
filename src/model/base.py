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
