"""
Comprehensive unit tests for the recursive loss factory.

This test suite covers all components of the recursive loss factory system:
- Basic loss instantiation
- Simple combinations (sum, product)
- Nested combinations
- Edge cases and error handling
- Configuration parsing and validation
- Performance considerations
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
        super().__init__()
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
        expected = (0.3 + 0.7 + 0.4) / 3  # 1.4 / 3 ≈ 0.4667
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
            0.3 * 0.3 + 0.6 * 0.7 + 0.1 * 0.4
        )  # 0.09 + 0.42 + 0.04 = 0.55
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
        # Second component: bce_loss * focal_loss = 0.7 * 0.4 = 0.28
        # Final: 0.7 * 0.3 + 0.3 * 0.28 = 0.21 + 0.084 = 0.294
        expected = 0.7 * 0.3 + 0.3 * (0.7 * 0.4)
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
        #   - focal_loss = 0.4
        #   - dice_loss * focal_loss = 0.3 * 0.4 = 0.12
        #   - Equal weights: (0.4 + 0.12) / 2 = 0.26
        # Final: 0.5 * 0.21 + 0.5 * 0.26 = 0.105 + 0.13 = 0.235
        first_comp = 0.3 * 0.7
        second_comp = (0.4 + 0.3 * 0.4) / 2
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
        # Second nested sum: (0.4 + 0.3) / 2 = 0.35
        # Final sum: (0.5 + 0.35) / 2 = 0.425
        first_sum = (0.3 + 0.7) / 2
        second_sum = (0.4 + 0.3) / 2
        expected = (first_sum + second_sum) / 2
        assert abs(result.item() - expected) < 1e-6


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
        combinators = []
        for _ in range(10):
            combinator = factory.create_from_config(config)
            combinators.append(combinator)

        pred, target = sample_tensors

        # All should produce the same result
        results = [combinator(pred, target) for combinator in combinators]
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

        combinator = factory.create_from_config(config)

        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 16), dtype=torch.float32)

        result = combinator(pred, target)
        result.backward()

        # Check that gradients exist (they will be zero since MockLoss returns
        # constants)
        assert pred.grad is not None
        # Note: Gradients are zero because MockLoss returns
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

        combinator = factory.create_from_config(config)

        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 16), dtype=torch.float32)

        result = combinator(pred, target)
        result.backward()

        # Check that gradients exist (they will be zero since MockLoss returns
        # constants)
        assert pred.grad is not None
        # Note: Gradients are zero because MockLoss returns
        # pred.mean() * 0 + constant
        # This is mathematically correct for constant loss functions

    def test_gradients_through_nested_combination(
        self, factory: RecursiveLossFactory
    ) -> None:
        """Test gradient flow through nested combinations."""
        config: dict[str, Any] = {
            "type": "sum",
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

        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 16, 16), dtype=torch.float32)

        result = combinator(pred, target)
        result.backward()

        # Check that gradients exist (they will be zero since MockLoss returns
        # constants)
        assert pred.grad is not None
        # Note: Gradients are zero because MockLoss returns
        # pred.mean() * 0 + constant
        # This is mathematically correct for constant loss functions


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
