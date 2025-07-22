"""
Unit tests for recursive loss factory configuration validation. This
module tests configuration validation functionality: - Valid
configuration detection - Configuration summaries - Validation
utilities
"""

from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from crackseg.training.losses.factory.recursive_factory import (
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


class TestConfigurationValidation:
    """Test configuration validation functionality."""

    def test_validate_valid_config(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test validation of valid configurations."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        assert factory.validate_config(config) is True

    def test_validate_invalid_config(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test validation of invalid configurations."""
        config: dict[str, Any] = {
            "type": "invalid_type",
            "components": [{"name": "dice_loss"}],
        }

        assert factory.validate_config(config) is False

    def test_config_summary(self, factory: RecursiveLossFactory) -> None:
        """Test configuration summary functionality."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.5, 0.5],
            "components": [
                {"name": "dice_loss"},
                {
                    "type": "product",
                    "components": [
                        {"name": "bce_loss"},
                        {"name": "focal_loss"},
                    ],
                },
            ],
        }

        summary = factory.get_config_summary(config)

        assert summary["valid"] is True
        assert (
            summary["total_nodes"] == 5
        )  # 1 root + 1 leaf + 1 combinator + 2 leaves
        assert summary["leaf_count"] == 3
        assert summary["combinator_count"] == 2
        assert summary["max_depth"] == 3
        assert "sum" in summary["combinator_types"]
        assert "product" in summary["combinator_types"]
        assert "dice_loss" in summary["loss_types"]
        assert "bce_loss" in summary["loss_types"]
        assert "focal_loss" in summary["loss_types"]
