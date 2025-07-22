"""
Unit tests for recursive loss factory error handling. This module
tests error conditions and edge cases: - Invalid configurations -
Malformed parameters - Error recovery scenarios
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

    def __init__(self, value: float = 0.5, **kwargs: Any):
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
        return MockLoss(value=0.4, **params)

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


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_invalid_combinator_type(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test error handling for invalid combinator types."""
        config: dict[str, Any] = {
            "type": "invalid_type",
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        with pytest.raises(RecursiveFactoryError) as exc_info:
            factory.create_from_config(config)

        assert "invalid_type" in str(exc_info.value)

    def test_mismatched_weights_and_components(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test error handling for mismatched weights and components."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.5, 0.3],  # Only 2 weights
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
                {"name": "focal_loss"},  # But 3 components
            ],
        }

        with pytest.raises(RecursiveFactoryError):
            factory.create_from_config(config)

    def test_empty_components_list(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test error handling for empty components list."""
        config: dict[str, Any] = {"type": "sum", "components": []}

        with pytest.raises(RecursiveFactoryError):
            factory.create_from_config(config)

    def test_negative_weights(self, factory: RecursiveLossFactory) -> None:
        """Test error handling for negative weights."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.5, -0.3],  # Negative weight
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        with pytest.raises(RecursiveFactoryError):
            factory.create_from_config(config)

    def test_zero_sum_weights(self, factory: RecursiveLossFactory) -> None:
        """Test error handling for zero sum weights."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.0, 0.0],  # Zero sum
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        with pytest.raises(RecursiveFactoryError):
            factory.create_from_config(config)

    def test_malformed_nested_config(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test error handling for malformed nested configurations."""
        config: dict[str, Any] = {
            "type": "sum",
            "components": [
                {"name": "dice_loss"},
                {
                    # Missing 'type' or 'name'
                    "components": [{"name": "bce_loss"}]
                },
            ],
        }

        with pytest.raises(RecursiveFactoryError):
            factory.create_from_config(config)
