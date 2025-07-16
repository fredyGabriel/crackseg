"""
Core interfaces for the loss system architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch


class ILossComponent(Protocol):
    """
    Protocol defining the interface for all loss components.
    Both leaf losses and combinators must implement this interface.
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between prediction and target."""
        ...

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Make the component callable."""
        ...


class ILossCombinator(ABC):
    """
    Abstract base class for loss combinators.
    Defines how multiple loss components can be combined.
    """

    @abstractmethod
    def __init__(self, components: list[ILossComponent], **kwargs: Any):
        """Initialize with a list of loss components."""
        pass

    @abstractmethod
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Combine multiple loss components."""
        pass


class ILossFactory(Protocol):
    """
    Protocol for loss factory implementations.
    Defines the contract for creating loss components from configuration.
    """

    def create_from_config(self, config: dict[str, Any]) -> ILossComponent:
        """Create a loss component from configuration."""
        ...

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate that a configuration is valid."""
        ...


class ILossRegistry(Protocol):
    """
    Protocol for loss registry implementations.
    Defines how loss implementations are registered and retrieved.
    """

    def register(self, name: str, loss_class: type, **metadata: Any) -> None:
        """Register a loss implementation."""
        ...

    def get(self, name: str) -> type:
        """Retrieve a registered loss implementation."""
        ...

    def instantiate(self, name: str, **params: Any) -> ILossComponent:
        """Create an instance of a registered loss."""
        ...

    def list_available(self) -> list[str]:
        """List all available registered losses."""
        ...
