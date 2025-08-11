from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BottleneckBase(nn.Module, ABC):
    """Base class for bottleneck implementations."""

    def __init__(self, in_channels: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the bottleneck."""
        raise NotImplementedError

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of channels in the bottleneck output tensor."""
        raise NotImplementedError
