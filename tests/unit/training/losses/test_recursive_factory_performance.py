"""
Unit tests for recursive loss factory performance and gradient flow.
This module tests performance characteristics and gradient flow: -
Memory usage optimization - Deep nesting performance - Gradient
propagation - Large configuration handling
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


class TestPerformanceAndMemory:
    """Test performance characteristics and memory usage."""

    def test_large_nested_configuration(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test performance with a large nested configuration."""
        # Create a configuration with many components
        base_components = [
            {"name": "dice_loss"},
            {"name": "bce_loss"},
            {"name": "focal_loss"},
        ]

        config: dict[str, Any] = {
            "type": "sum",
            "components": base_components * 10,  # 30 components total
        }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        # Should complete without timeout or memory issues
        result = combinator(pred, target)
        assert isinstance(result, torch.Tensor)

    def test_deep_nesting_levels(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test performance with deep nesting levels."""
        # Create a deeply nested configuration (5 levels)
        config: dict[str, Any] = {"name": "dice_loss"}
        for _ in range(5):
            config = {
                "type": "sum",
                "components": [config, {"name": "bce_loss"}],
            }

        combinator = factory.create_from_config(config)
        pred, target = sample_tensors

        result = combinator(pred, target)
        assert isinstance(result, torch.Tensor)

    def test_repeated_instantiation(
        self,
        factory: RecursiveLossFactory,
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test performance of repeated instantiation."""
        config: dict[str, Any] = {
            "type": "sum",
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        # Create multiple instances
        from crackseg.training.losses.interfaces.loss_interface import (
            ILossComponent,
        )

        combinators: list[ILossComponent] = []
        for _ in range(10):
            combinator = factory.create_from_config(config)
            combinators.append(combinator)

        pred, target = sample_tensors

        # All should produce the same result
        results: list[torch.Tensor] = [
            combinator(pred, target) for combinator in combinators
        ]
        for result in results[1:]:
            assert abs(result.item() - results[0].item()) < 1e-6


class TestGradientFlow:
    """Test gradient flow through combinations."""

    def test_gradients_through_sum(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test that gradients flow correctly through sum combinators."""
        config: dict[str, Any] = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        loss_fn = factory.create_from_config(config)
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()

        result = loss_fn(pred, target)
        result.backward()

        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))

        # Expected: 0.6 * 0.3 + 0.4 * 0.7 = 0.18 + 0.28 = 0.46
        expected = 0.6 * 0.3 + 0.4 * 0.7
        assert abs(result.item() - expected) < 1e-6

        # pred.mean() * 0 + constant
        # This is mathematically correct for constant loss functions

    def test_gradients_through_product(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test that gradients flow correctly through product combinators."""
        config: dict[str, Any] = {
            "type": "product",
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        loss_fn = factory.create_from_config(config)
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()

        result = loss_fn(pred, target)
        result.backward()

        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))

        # Expected: 0.3 * 0.7 = 0.21
        expected = 0.3 * 0.7
        assert abs(result.item() - expected) < 1e-6

        # pred.mean() * 0 + constant
        # This is mathematically correct for constant loss functions

    def test_gradients_through_nested_combination(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test gradient flow through nested combinations."""
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

        loss_fn = factory.create_from_config(config)
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()

        result = loss_fn(pred, target)
        result.backward()

        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))

        # Expected: 0.5 * 0.3 + 0.5 * (0.7 * 0.9) = 0.15 + 0.315 = 0.465
        expected = 0.5 * 0.3 + 0.5 * (0.7 * 0.9)
        assert abs(result.item() - expected) < 1e-6

        # pred.mean() * 0 + constant
        # This is mathematically correct for constant loss functions
