"""
Unit tests for basic recursive loss factory instantiation. This module
tests the core instantiation functionality: - Simple loss creation -
Parameter passing - Basic error handling
"""

from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from crackseg.training.losses.factory.recursive_factory import (
    RecursiveFactoryError,
    RecursiveLossFactory,
)
from crackseg.training.losses.interfaces.loss_interface import (
    ILossComponent,
)
from crackseg.training.losses.registry.clean_registry import (
    CleanLossRegistry,
)
from crackseg.training.losses.registry.enhanced_registry import (
    EnhancedLossRegistry,
)


class MockLoss(nn.Module, ILossComponent):
    """Mock loss component for testing."""

    def __init__(self, value: float = 0.5, **kwargs: Any) -> None:
        super().__init__()  # type: ignore[misc]
        self.value = value
        self.params = kwargs

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # Create a tensor that depends on pred to ensure gradient flow
        # Use pred.mean() * 0 + self.value to maintain gradient connection
        return pred.mean() * 0 + self.value


@pytest.fixture
def mock_registry() -> CleanLossRegistry:
    """Create a mock registry with test losses."""
    registry = CleanLossRegistry()

    def dice_loss_factory(**params: Any) -> MockLoss:
        return MockLoss(value=0.3, **params)

    def bce_loss_factory(**params: Any) -> MockLoss:
        return MockLoss(value=0.7, **params)

    def focal_loss_factory(**params: Any) -> MockLoss:
        return MockLoss(value=0.9, **params)

    registry.register_factory("dice_loss", dice_loss_factory)
    registry.register_factory("bce_loss", bce_loss_factory)
    registry.register_factory("focal_loss", focal_loss_factory)

    return registry


@pytest.fixture
def factory(mock_registry: CleanLossRegistry) -> RecursiveLossFactory:
    """Create a recursive factory with mock registry."""
    factory: RecursiveLossFactory = RecursiveLossFactory()
    factory.registry = cast(EnhancedLossRegistry, mock_registry)
    return factory


@pytest.fixture
def sample_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample prediction and target tensors."""
    pred: torch.Tensor = torch.randn(4, 1, 32, 32, requires_grad=True)
    target: torch.Tensor = torch.randint(
        0, 2, (4, 1, 32, 32), dtype=torch.float32
    )
    return pred, target


class TestBasicLossInstantiation:
    """Test basic loss instantiation functionality."""

    def test_create_simple_loss(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test creating a simple loss from configuration."""
        config: dict[str, Any] = {
            "name": "dice_loss",
            "params": {"smooth": 1.0},
        }

        loss = factory.create_from_config(config)
        pred, target = sample_tensors

        # Test that loss can be computed
        result = loss(pred, target)
        assert isinstance(result, torch.Tensor)
        assert result.requires_grad
        assert (
            abs(result.item() - 0.3) < 1e-6
        )  # MockLoss value with floating point tolerance

    def test_create_loss_with_parameters(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test creating loss with various parameters."""
        config: dict[str, Any] = {
            "name": "bce_loss",
            "params": {"reduction": "mean", "weight": 0.8},
        }

        loss = factory.create_from_config(config)
        pred, target = sample_tensors

        result = loss(pred, target)
        assert isinstance(result, torch.Tensor)
        assert (
            abs(result.item() - 0.7) < 1e-6
        )  # MockLoss value with floating point tolerance

        # Check that parameters were passed correctly
        if isinstance(loss, MockLoss):
            assert loss.params["reduction"] == "mean"
            assert loss.params["weight"] == 0.8

    def test_invalid_loss_name(self, factory: RecursiveLossFactory) -> None:
        """Test error handling for invalid loss names."""
        config: dict[str, Any] = {"name": "nonexistent_loss"}

        with pytest.raises(RecursiveFactoryError) as exc_info:
            factory.create_from_config(config)

        assert "nonexistent_loss" in str(exc_info.value)

    def test_empty_configuration(self, factory: RecursiveLossFactory) -> None:
        """Test error handling for empty configuration."""
        config: dict[str, Any] = {}

        with pytest.raises(RecursiveFactoryError):
            factory.create_from_config(config)
