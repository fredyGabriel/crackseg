"""
Unit tests for recursive loss factory combinations. This module tests
loss combination functionality: - Simple combinations (sum, product) -
Nested combinations - Weight handling and normalization
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


class TestSimpleCombinations:
    """Test simple loss combinations (sum, product)."""

    def test_weighted_sum_combination(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test creating a weighted sum combination."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        expected = 0.6 * 0.3 + 0.4 * 0.7  # 0.18 + 0.28 = 0.46
        assert abs(result.item() - expected) < 1e-6

    def test_equal_weights_sum(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test sum combination with equal weights (auto-generated)."""
        config: dict[str, Any] = {
            "type": "sum",
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
                {"name": "focal_loss"},
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        # Equal weights: 1/3 each
        expected = (0.3 + 0.7 + 0.9) / 3  # 1.9 / 3 ≈ 0.6333
        assert abs(result.item() - expected) < 1e-6

    def test_product_combination(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test creating a product combination."""
        config: dict[str, Any] = {
            "type": "product",
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        expected = 0.3 * 0.7  # 0.21
        assert abs(result.item() - expected) < 1e-6

    def test_weight_normalization(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that weights are properly normalized."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [3.0, 6.0, 1.0],  # Should normalize to [0.3, 0.6, 0.1]
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
                {"name": "focal_loss"},
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        expected = (
            0.3 * 0.3 + 0.6 * 0.7 + 0.1 * 0.9
        )  # 0.09 + 0.42 + 0.09 = 0.60
        assert abs(result.item() - expected) < 1e-6


class TestNestedCombinations:
    """Test complex nested loss combinations."""

    def test_nested_sum_and_product(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test a combination with nested sum and product."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.7, 0.3],
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

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        # First component: dice_loss = 0.3
        # Second component: bce_loss * focal_loss = 0.7 * 0.9 = 0.63
        # Final: 0.7 * 0.3 + 0.3 * 0.63 = 0.21 + 0.189 = 0.399
        expected = 0.7 * 0.3 + 0.3 * (0.7 * 0.9)
        assert abs(result.item() - expected) < 1e-6

    def test_deeply_nested_configuration(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test a deeply nested configuration (3 levels)."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.5, 0.5],
            "components": [
                {
                    "type": "product",
                    "components": [
                        {"name": "dice_loss"},
                        {"name": "bce_loss"},
                    ],
                },
                {
                    "type": "sum",
                    "components": [
                        {"name": "focal_loss"},
                        {
                            "type": "product",
                            "components": [
                                {"name": "dice_loss"},
                                {"name": "focal_loss"},
                            ],
                        },
                    ],
                },
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        # First component: dice_loss * bce_loss = 0.3 * 0.7 = 0.21
        # Second component (sum with equal weights):
        #   - focal_loss = 0.9
        #   - dice_loss * focal_loss = 0.3 * 0.9 = 0.27
        #   - Equal weights: (0.9 + 0.27) / 2 = 0.585
        # Final: 0.5 * 0.21 + 0.5 * 0.585 = 0.105 + 0.2925 = 0.3975
        first_comp = 0.3 * 0.7
        second_comp = (0.9 + 0.3 * 0.9) / 2
        expected = 0.5 * first_comp + 0.5 * second_comp
        assert abs(result.item() - expected) < 1e-6

    def test_multiple_levels_same_type(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test multiple levels of the same combinator type."""
        config: dict[str, Any] = {
            "type": "sum",
            "components": [
                {
                    "type": "sum",
                    "components": [
                        {"name": "dice_loss"},
                        {"name": "bce_loss"},
                    ],
                },
                {
                    "type": "sum",
                    "components": [
                        {"name": "focal_loss"},
                        {"name": "dice_loss"},
                    ],
                },
            ],
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        # First nested sum: (0.3 + 0.7) / 2 = 0.5
        # Second nested sum: (0.9 + 0.3) / 2 = 0.6
        # Final sum: (0.5 + 0.6) / 2 = 0.55
        first_sum = (0.3 + 0.7) / 2
        second_sum = (0.9 + 0.3) / 2
        expected = (first_sum + second_sum) / 2
        assert abs(result.item() - expected) < 1e-6
