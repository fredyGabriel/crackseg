from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class DecoderBase(nn.Module, ABC):
    """Base class for all decoder implementations."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: list[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self._skip_channels = skip_channels

    @abstractmethod
    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        """Defines the forward pass for the decoder."""
        raise NotImplementedError

    @property
    def skip_channels(self) -> list[int]:
        """Decoder-expected skip channels order (low->high resolution)."""
        if hasattr(self, "_skip_channels"):
            return self._skip_channels
        raise AttributeError(
            "DecoderBase has not properly initialized _skip_channels"
        )

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of channels in the final output tensor of the decoder."""
        raise NotImplementedError
