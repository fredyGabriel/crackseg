from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class EncoderBase(nn.Module, ABC):
    """Base class for all encoder implementations.

    Encoders extract multi-scale features from input images, typically
    following a hierarchical structure where each stage produces features
    at different spatial resolutions.
    """

    def __init__(self, in_channels: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Defines the forward pass for the encoder."""
        raise NotImplementedError

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of channels in the final output tensor of the encoder."""
        raise NotImplementedError

    @property
    @abstractmethod
    def skip_channels(self) -> list[int]:
        """Channel dimensions for each skip connection feature map."""
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_info(self) -> list[dict[str, Any]]:
        """Information about output features for each stage."""
        raise NotImplementedError

    @property
    def feature_reduction(self) -> list[int]:
        """Spatial reduction factors for each feature stage."""
        return [info["reduction"] for info in self.feature_info]

    def get_stages(self) -> list[nn.Module]:
        """Get encoder stages as separate modules if available."""
        return []
