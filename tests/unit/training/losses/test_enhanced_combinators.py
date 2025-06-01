"""
Comprehensive unit tests for enhanced loss combinators.

This test suite covers all advanced features of the enhanced combinators:
- Enhanced weighted sum combinator functionality
- Enhanced product combinator functionality
- Numerical stability features
- Edge case handling
- Performance monitoring
- Advanced combinator features
"""

import pytest
import torch
import torch.nn as nn

from src.training.losses.combinators.base_combinator import (
    CombinatorFactory,
    NumericalStabilityError,
    ValidationError,
    handle_zero_weights,
    normalize_weights,
    validate_component_compatibility,
)
from src.training.losses.combinators.enhanced_product import (
    EnhancedProductCombinator,
    create_product_combinator,
)
from src.training.losses.combinators.enhanced_weighted_sum import (
    EnhancedWeightedSumCombinator,
    create_weighted_sum_combinator,
)
from src.training.losses.interfaces.loss_interface import ILossComponent


class MockLossComponent(nn.Module, ILossComponent):
    """Mock loss component for testing."""

    def __init__(self, value: float = 0.5, device: str = "cpu") -> None:
        super().__init__()
        self.value = value
        self.device_name = device

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(
            self.value,
            requires_grad=True,
            device=pred.device,
            dtype=pred.dtype,
        )


class BadLossComponent:
    """Non-torch module loss component for testing warnings."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(self.value, requires_grad=True, device=pred.device)


@pytest.fixture
def mock_components() -> list[ILossComponent]:
    """Create mock loss components for testing."""
    return [
        MockLossComponent(0.3),
        MockLossComponent(0.7),
        MockLossComponent(0.4),
    ]


@pytest.fixture
def sample_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample prediction and target tensors."""
    pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    return pred, target


class TestEnhancedWeightedSumCombinator:
    """Test enhanced weighted sum combinator functionality."""

    def test_basic_weighted_sum(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test basic weighted sum functionality."""
        weights = [0.5, 0.3, 0.2]
        combinator = EnhancedWeightedSumCombinator(mock_components, weights)

        pred, target = sample_tensors
        result = combinator(pred, target)

        expected = (
            0.5 * 0.3 + 0.3 * 0.7 + 0.2 * 0.4
        )  # 0.15 + 0.21 + 0.08 = 0.44
        assert abs(result.item() - expected) < 1e-6

    def test_equal_weights_auto_generation(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test automatic equal weight generation."""
        combinator = EnhancedWeightedSumCombinator(mock_components)

        pred, target = sample_tensors
        result = combinator(pred, target)

        # Should use equal weights: 1/3 each
        expected = (0.3 + 0.7 + 0.4) / 3  # 1.4 / 3 ≈ 0.4667
        assert abs(result.item() - expected) < 1e-6

    def test_weight_normalization(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test automatic weight normalization."""
        weights = [3.0, 6.0, 1.0]  # Should normalize to [0.3, 0.6, 0.1]
        combinator = EnhancedWeightedSumCombinator(
            mock_components, weights, auto_normalize=True
        )

        pred, target = sample_tensors
        result = combinator(pred, target)

        expected = (
            0.3 * 0.3 + 0.6 * 0.7 + 0.1 * 0.4
        )  # 0.09 + 0.42 + 0.04 = 0.55
        assert abs(result.item() - expected) < 1e-6

    def test_zero_weight_handling(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test zero weight handling with epsilon replacement."""
        weights = [0.5, 0.0, 0.5]  # Middle weight is zero
        combinator = EnhancedWeightedSumCombinator(
            mock_components, weights, handle_zero_weights_flag=True
        )

        pred, target = sample_tensors
        result = combinator(pred, target)

        # Zero weight should be replaced with epsilon and normalized
        # Original: [0.5, 1e-8, 0.5] -> normalized
        # Since 1e-8 is very small, result should be approximately
        # 0.5 * 0.3 + 0.5 * 0.4
        assert isinstance(result, torch.Tensor)
        assert result.requires_grad

    def test_weight_update(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test dynamic weight updating."""
        initial_weights = [0.5, 0.3, 0.2]
        combinator = EnhancedWeightedSumCombinator(
            mock_components, initial_weights
        )

        # Test initial result
        pred, target = sample_tensors
        initial_result = combinator(pred, target)

        # Update weights
        new_weights = [0.2, 0.3, 0.5]
        combinator.update_weights(new_weights)

        # Test updated result
        updated_result = combinator(pred, target)

        # Results should be different
        assert abs(initial_result.item() - updated_result.item()) > 1e-6

        # Check new weights are correct
        expected_new = (
            0.2 * 0.3 + 0.3 * 0.7 + 0.5 * 0.4
        )  # 0.06 + 0.21 + 0.2 = 0.47
        assert abs(updated_result.item() - expected_new) < 1e-6

    def test_weight_statistics(
        self, mock_components: list[ILossComponent]
    ) -> None:
        """Test weight statistics calculation."""
        weights = [0.1, 0.7, 0.2]
        combinator = EnhancedWeightedSumCombinator(mock_components, weights)

        stats = combinator.get_weight_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "sum" in stats
        assert "entropy" in stats

        # Check values are reasonable
        assert abs(stats["sum"] - 1.0) < 1e-6  # Normalized weights sum to 1
        assert (
            abs(stats["min"] - 0.1) < 1e-6
        )  # Account for floating point precision
        assert abs(stats["max"] - 0.7) < 1e-6

    def test_dominant_component_setting(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test setting a dominant component."""
        combinator = EnhancedWeightedSumCombinator(mock_components)

        # Set component 1 as dominant with 80% weight
        combinator.set_dominant_component(1, dominance=0.8)

        pred, target = sample_tensors
        result = combinator(pred, target)

        # Component 1 has value 0.7, others get (1-0.8)/2 = 0.1 each
        expected = (
            0.1 * 0.3 + 0.8 * 0.7 + 0.1 * 0.4
        )  # 0.03 + 0.56 + 0.04 = 0.63
        assert abs(result.item() - expected) < 1e-6

    def test_weight_balancing(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test weight balancing to equal distribution."""
        weights = [0.9, 0.05, 0.05]  # Imbalanced weights
        combinator = EnhancedWeightedSumCombinator(mock_components, weights)

        # Balance weights
        combinator.balance_weights()

        pred, target = sample_tensors
        result = combinator(pred, target)

        # Should now use equal weights: 1/3 each
        expected = (0.3 + 0.7 + 0.4) / 3
        assert abs(result.item() - expected) < 1e-6

    def test_invalid_weight_update(
        self, mock_components: list[ILossComponent]
    ) -> None:
        """Test error handling for invalid weight updates."""
        combinator = EnhancedWeightedSumCombinator(mock_components)

        # Wrong number of weights
        with pytest.raises(ValueError):
            combinator.update_weights(
                [0.5, 0.5]
            )  # Only 2 weights for 3 components

        # Invalid dominance value
        with pytest.raises(ValueError):
            combinator.set_dominant_component(0, dominance=1.5)  # > 1.0

        # Invalid component index
        with pytest.raises(ValueError):
            combinator.set_dominant_component(
                5, dominance=0.8
            )  # Index out of range


class TestEnhancedProductCombinator:
    """Test enhanced product combinator functionality."""

    def test_basic_product(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test basic product functionality."""
        combinator = EnhancedProductCombinator(mock_components)
        pred, target = sample_tensors
        result = combinator(pred, target)
        expected = 0.3 * 0.7 * 0.4  # 0.084
        assert abs(result.item() - expected) < 1e-6

    def test_log_space_computation(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test log-space computation for numerical stability."""
        combinator = EnhancedProductCombinator(
            mock_components, use_log_space=True
        )
        pred, target = sample_tensors
        result = combinator(pred, target)
        expected = 0.3 * 0.7 * 0.4  # 0.084
        assert abs(result.item() - expected) < 1e-6

    def test_direct_computation(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test direct computation mode."""
        combinator = EnhancedProductCombinator(
            mock_components, use_log_space=False
        )
        pred, target = sample_tensors
        result = combinator(pred, target)
        expected = 0.3 * 0.7 * 0.4  # 0.084
        assert abs(result.item() - expected) < 1e-6

    def test_zero_handling(
        self, sample_tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test handling of zero values."""
        components: list[ILossComponent] = [
            MockLossComponent(0.0),  # Zero value
            MockLossComponent(0.5),
            MockLossComponent(0.3),
        ]
        combinator = EnhancedProductCombinator(
            components, zero_epsilon=1e-6, use_log_space=True
        )
        pred, target = sample_tensors
        result = combinator(pred, target)
        assert result.item() > 0
        assert not torch.isnan(result)
        assert not torch.isinf(result)

    def test_very_small_values(
        self, sample_tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test handling of very small values that could cause underflow."""
        components: list[ILossComponent] = [
            MockLossComponent(1e-10),
            MockLossComponent(1e-8),
            MockLossComponent(1e-6),
        ]
        combinator = EnhancedProductCombinator(components, use_log_space=True)
        pred, target = sample_tensors
        result = combinator(pred, target)
        assert not torch.isnan(result)
        assert not torch.isinf(result)
        assert result.item() > 0

    def test_product_statistics(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test product-specific statistics."""
        combinator = EnhancedProductCombinator(mock_components)
        pred, target = sample_tensors
        _ = combinator(pred, target)
        stats = combinator.get_product_statistics()
        assert "zero_replacements" in stats
        assert "stability_warnings" in stats
        assert "use_log_space" in stats
        assert "zero_epsilon" in stats
        assert "stability_threshold" in stats

    def test_component_range_analysis(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test component output range analysis."""
        combinator = EnhancedProductCombinator(mock_components)
        pred, target = sample_tensors
        _ = combinator(pred, target)
        component_outputs = [comp(pred, target) for comp in mock_components]
        analysis = combinator.analyze_component_ranges(component_outputs)
        assert len(analysis) == 3  # One analysis per component
        for i in range(3):
            comp_analysis = analysis[f"component_{i}"]
            assert "min" in comp_analysis
            assert "max" in comp_analysis
            assert "mean" in comp_analysis
            assert "std" in comp_analysis
            assert "zeros" in comp_analysis
            assert "near_zeros" in comp_analysis
            assert "shape" in comp_analysis

    def test_geometric_mean_equivalent(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test geometric mean equivalent calculation."""
        combinator = EnhancedProductCombinator(mock_components)
        pred, target = sample_tensors
        _ = combinator(pred, target)
        component_outputs = [comp(pred, target) for comp in mock_components]
        geom_mean = combinator.get_geometric_mean_equivalent(component_outputs)
        expected_geom_mean = (0.3 * 0.7 * 0.4) ** (1 / 3)
        assert abs(geom_mean.item() - expected_geom_mean) < 1e-6

    def test_log_space_mode_switching(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test switching between log-space and direct modes."""
        combinator = EnhancedProductCombinator(
            mock_components, use_log_space=False
        )
        pred, target = sample_tensors
        direct_result = combinator(pred, target)
        combinator.set_log_space_mode(True)
        log_result = combinator(pred, target)
        assert abs(direct_result.item() - log_result.item()) < 1e-6

    def test_epsilon_adjustment(
        self, mock_components: list[ILossComponent]
    ) -> None:
        """Test adjusting zero epsilon value."""
        combinator = EnhancedProductCombinator(mock_components)
        new_epsilon = 1e-10
        combinator.set_zero_epsilon(new_epsilon)
        assert combinator.zero_epsilon == new_epsilon
        with pytest.raises(ValueError):
            combinator.set_zero_epsilon(-1e-8)  # Negative epsilon


class TestBaseCombinatorValidation:
    """Test base combinator validation features."""

    def test_component_validation(
        self, sample_tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test component validation during initialization."""
        valid_components: list[ILossComponent] = [MockLossComponent(0.5)]
        EnhancedWeightedSumCombinator(valid_components)
        with pytest.raises(ValidationError):
            EnhancedWeightedSumCombinator([])
        with pytest.raises(ValidationError):
            EnhancedWeightedSumCombinator("not_a_list")  # type: ignore
        with pytest.raises(ValidationError):
            EnhancedWeightedSumCombinator(["not_callable"])  # type: ignore

    def test_input_validation(
        self, mock_components: list[ILossComponent]
    ) -> None:
        """Test input validation during forward pass."""
        combinator = EnhancedWeightedSumCombinator(
            mock_components, validate_inputs=True
        )
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        combinator(pred, target)  # Should work
        with pytest.raises(ValidationError):
            combinator("not_tensor", target)  # type: ignore
        with pytest.raises(ValidationError):
            combinator(pred, "not_tensor")  # type: ignore
        with pytest.raises(ValidationError):
            combinator(
                pred, torch.randn(2, 3, 16, 16)
            )  # Different spatial size
        if torch.cuda.is_available():
            pred_gpu = pred.cuda()
            with pytest.raises(ValidationError):
                combinator(pred_gpu, target)  # target is on CPU

    def test_numerical_stability_checking(
        self, sample_tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test numerical stability checking."""

        # Create a component that returns NaN
        class NaNComponent(nn.Module):
            def forward(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return torch.full_like(pred, float("nan"))

        # Create a component that returns Inf
        class InfComponent(nn.Module):
            def forward(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return torch.full_like(pred, float("inf"))

        # Test NaN detection
        combinator = EnhancedWeightedSumCombinator(
            [NaNComponent()], numerical_stability_check=True
        )
        pred, target = sample_tensors

        with pytest.raises(NumericalStabilityError):
            combinator(pred, target)

        # Test Inf detection
        combinator = EnhancedWeightedSumCombinator(
            [InfComponent()], numerical_stability_check=True
        )

        with pytest.raises(NumericalStabilityError):
            combinator(pred, target)

    def test_gradient_clipping(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test gradient clipping functionality."""
        combinator = EnhancedWeightedSumCombinator(
            mock_components, gradient_clipping=1.0
        )

        pred, target = sample_tensors
        result = combinator(pred, target)
        result.backward()

        # Check that gradient clipping was applied (statistics should show
        # clips)
        combinator.get_statistics()
        # Note: Gradient clipping is applied via hooks, so we mainly test that
        # it doesn't error

    def test_statistics_and_monitoring(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test statistics collection and monitoring."""
        combinator = EnhancedWeightedSumCombinator(mock_components)

        pred, target = sample_tensors

        # Initial statistics
        initial_stats = combinator.get_statistics()
        assert initial_stats["forward_count"] == 0

        # After forward pass
        _ = combinator(pred, target)
        updated_stats = combinator.get_statistics()
        assert updated_stats["forward_count"] == 1

        # Test statistics reset
        combinator.reset_statistics()
        reset_stats = combinator.get_statistics()
        assert reset_stats["forward_count"] == 0

    def test_validation_mode_control(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test enabling/disabling validation modes."""
        combinator = EnhancedWeightedSumCombinator(
            mock_components,
            validate_inputs=True,
            numerical_stability_check=True,
        )

        # Test disabling validation
        combinator.set_validation_mode(False)
        combinator.set_numerical_stability_check(False)

        pred, target = sample_tensors
        combinator(pred, target)  # Should work even with validation disabled


class TestCombinatorFactory:
    """Test combinator factory functionality."""

    def test_factory_weighted_sum_creation(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        weights = [0.5, 0.3, 0.2]
        combinator = CombinatorFactory.create_weighted_sum(
            mock_components, weights
        )
        pred, target = sample_tensors
        result = combinator(pred, target)
        expected = 0.5 * 0.3 + 0.3 * 0.7 + 0.2 * 0.4
        assert abs(result.item() - expected) < 1e-6

    def test_factory_product_creation(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        combinator = CombinatorFactory.create_product(mock_components)
        pred, target = sample_tensors
        result = combinator(pred, target)
        expected = 0.3 * 0.7 * 0.4
        assert abs(result.item() - expected) < 1e-6

    def test_factory_from_config(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        sum_config = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": mock_components[:2],
        }
        sum_combinator = CombinatorFactory.create_from_config(sum_config)
        pred, target = sample_tensors
        sum_result = sum_combinator(pred, target)
        expected_sum = 0.6 * 0.3 + 0.4 * 0.7
        assert abs(sum_result.item() - expected_sum) < 1e-6
        product_config = {
            "type": "product",
            "components": mock_components[:2],
        }
        product_combinator = CombinatorFactory.create_from_config(
            product_config
        )
        product_result = product_combinator(pred, target)
        expected_product = 0.3 * 0.7
        assert abs(product_result.item() - expected_product) < 1e-6
        with pytest.raises(ValueError):
            CombinatorFactory.create_from_config({"type": "invalid"})


class TestUtilityFunctions:
    """Test utility functions for combinators."""

    def test_handle_zero_weights(self) -> None:
        """Test zero weight handling utility."""
        weights = [0.5, 0.0, 0.3, 0.0]
        processed = handle_zero_weights(weights)
        assert all(w > 0 for w in processed)
        assert processed[0] == 0.5
        assert processed[2] == 0.3
        assert processed[1] == 1e-8
        assert processed[3] == 1e-8

    def test_normalize_weights(self) -> None:
        """Test weight normalization utility."""
        weights = [3.0, 6.0, 1.0]
        normalized = normalize_weights(weights)
        assert abs(sum(normalized) - 1.0) < 1e-6
        expected = [0.3, 0.6, 0.1]
        for actual, exp in zip(normalized, expected, strict=False):
            assert abs(actual - exp) < 1e-6
        zero_weights = [0.0, 0.0, 0.0]
        normalized_zeros = normalize_weights(zero_weights)
        assert abs(sum(normalized_zeros) - 1.0) < 1e-6

    def test_component_compatibility_validation(
        self, sample_tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test component compatibility validation."""
        pred, target = sample_tensors
        compatible_components: list[ILossComponent] = [
            MockLossComponent(0.5),
            MockLossComponent(0.3),
        ]
        result = validate_component_compatibility(
            compatible_components, pred, target
        )
        assert result is True

        class IncompatibleComponent:
            def __call__(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return torch.ones_like(pred)

        incompatible_components: list[ILossComponent] = [
            MockLossComponent(0.5),
            IncompatibleComponent(),  # type: ignore
        ]
        with pytest.raises(ValidationError):
            validate_component_compatibility(
                incompatible_components, pred, target
            )


class TestConvenienceFunctions:
    """Test convenience functions for creating combinators."""

    def test_create_weighted_sum_combinator(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        weights = [0.4, 0.4, 0.2]
        combinator = create_weighted_sum_combinator(mock_components, weights)
        pred, target = sample_tensors
        result = combinator(pred, target)
        expected = 0.4 * 0.3 + 0.4 * 0.7 + 0.2 * 0.4
        assert abs(result.item() - expected) < 1e-6

    def test_create_product_combinator(
        self,
        mock_components: list[ILossComponent],
        sample_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        combinator = create_product_combinator(mock_components)
        pred, target = sample_tensors
        result = combinator(pred, target)
        expected = 0.3 * 0.7 * 0.4
        assert abs(result.item() - expected) < 1e-6


class TestReductionModes:
    """Test different reduction modes."""

    def test_mean_reduction(
        self, mock_components: list[ILossComponent]
    ) -> None:
        combinator = EnhancedWeightedSumCombinator(
            mock_components, reduction="mean"
        )
        pred = torch.randn(4, 3, 16, 16, requires_grad=True)
        target = torch.randn(4, 3, 16, 16)
        result = combinator(pred, target)
        assert result.dim() == 0
        assert isinstance(result.item(), float)

    def test_sum_reduction(
        self, mock_components: list[ILossComponent]
    ) -> None:
        combinator = EnhancedWeightedSumCombinator(
            mock_components, reduction="sum"
        )
        pred = torch.randn(4, 3, 16, 16, requires_grad=True)
        target = torch.randn(4, 3, 16, 16)
        result = combinator(pred, target)
        assert result.dim() == 0
        assert isinstance(result.item(), float)

    def test_none_reduction(
        self, mock_components: list[ILossComponent]
    ) -> None:
        combinator = EnhancedWeightedSumCombinator(
            mock_components, reduction="none"
        )
        pred = torch.randn(4, 3, 16, 16, requires_grad=True)
        target = torch.randn(4, 3, 16, 16)
        result = combinator(pred, target)
        assert result.dim() == 0
        assert isinstance(result, torch.Tensor)

    def test_invalid_reduction(
        self, mock_components: list[ILossComponent]
    ) -> None:
        combinator = EnhancedWeightedSumCombinator(mock_components)
        combinator.reduction = "invalid"
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randn(2, 3, 16, 16)
        with pytest.raises(ValueError):
            combinator(pred, target)
