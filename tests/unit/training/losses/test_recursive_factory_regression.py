"""
Unit tests for recursive loss factory regression scenarios. This
module tests known working configurations: - Standard segmentation
losses - Complex multi-task losses - Production-ready configurations
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


class TestRegressionTests:
    """Regression tests for known configurations."""

    def test_standard_segmentation_loss(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test a standard segmentation loss configuration."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.8, 0.2],
            "components": [
                {"name": "dice_loss", "params": {"smooth": 1.0}},
                {"name": "bce_loss", "params": {"reduction": "mean"}},
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        expected = 0.8 * 0.3 + 0.2 * 0.7  # 0.24 + 0.14 = 0.38
        assert abs(result.item() - expected) < 1e-6

    def test_complex_multitask_loss(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test a complex multi-task loss configuration."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.6, 0.3, 0.1],
            "components": [
                {
                    "type": "sum",
                    "weights": [0.7, 0.3],
                    "components": [
                        {"name": "dice_loss"},
                        {"name": "bce_loss"},
                    ],
                },
                {
                    "type": "product",
                    "components": [
                        {"name": "focal_loss"},
                        {"name": "dice_loss"},
                    ],
                },
                {"name": "bce_loss"},
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)

        # Calculate expected result step by step
        first_sum = 0.7 * 0.3 + 0.3 * 0.7  # 0.21 + 0.21 = 0.42
        product = 0.4 * 0.3  # 0.12
        third_comp = 0.7
        expected = 0.6 * first_sum + 0.3 * product + 0.1 * third_comp
        # = 0.6 * 0.42 + 0.3 * 0.12 + 0.1 * 0.7
        # = 0.252 + 0.036 + 0.07 = 0.358

        assert abs(result.item() - expected) < 1e-6

    def test_production_ready_configuration(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test a production-ready configuration with all features."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.4, 0.4, 0.2],
            "components": [
                {"name": "dice_loss", "params": {"smooth": 1.0}},
                {
                    "type": "product",
                    "components": [
                        {"name": "bce_loss", "params": {"reduction": "mean"}},
                        {"name": "focal_loss", "params": {"alpha": 0.25}},
                    ],
                },
                {
                    "type": "sum",
                    "weights": [0.8, 0.2],
                    "components": [
                        {"name": "focal_loss"},
                        {"name": "dice_loss"},
                    ],
                },
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        # Should execute without errors
        result = combinator(pred, target)
        assert isinstance(result, torch.Tensor)
        assert result.requires_grad
        assert not torch.isnan(result)
        assert not torch.isinf(result)
