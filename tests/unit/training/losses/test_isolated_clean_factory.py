"""
Isolated unit tests for the clean recursive loss factory architecture.
This test file imports only the loss modules to avoid circular dependency
issues in other parts.
"""

from typing import Any

import pytest
import torch

from src.training.losses.combinators.product import ProductCombinator
from src.training.losses.combinators.weighted_sum import WeightedSumCombinator
from src.training.losses.factory.config_parser import ConfigParsingError
from src.training.losses.factory.config_validator import ConfigValidator
from src.training.losses.factory.recursive_factory import RecursiveLossFactory

# Import only loss-related modules to avoid circular dependencies from model
# components
from src.training.losses.registry.clean_registry import CleanLossRegistry


@pytest.fixture
def sample_pred_target() -> tuple[torch.Tensor, torch.Tensor]:
    """Sample prediction and target tensors for testing."""
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    return pred, target


@pytest.fixture
def mock_registry() -> CleanLossRegistry:
    """Create a mock registry with a simple test loss for isolated testing."""
    registry: CleanLossRegistry = CleanLossRegistry()

    # Register a simple mock loss for testing
    def mock_dice_loss(**params: Any) -> torch.nn.Module:
        class MockDiceLoss(torch.nn.Module):
            def __init__(self, smooth: float = 1.0) -> None:
                super().__init__()  # type: ignore[override]
                self.smooth = smooth

            def forward(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.5, requires_grad=True)

        return MockDiceLoss(**params)

    def mock_bce_loss(**params: Any) -> torch.nn.Module:
        class MockBCELoss(torch.nn.Module):
            def __init__(self, reduction: str = "mean") -> None:
                super().__init__()  # type: ignore[override]
                self.reduction = reduction

            def forward(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.3, requires_grad=True)

        return MockBCELoss(**params)

    registry.register_factory("dice_loss", mock_dice_loss, tags=["test"])  # type: ignore[arg-type]
    registry.register_factory("bce_loss", mock_bce_loss, tags=["test"])  # type: ignore[arg-type]

    return registry


class TestIsolatedCleanFactory:
    """Test the clean factory in complete isolation."""

    def test_registry_basic_functionality(
        self, mock_registry: CleanLossRegistry
    ) -> None:
        """Test basic registry functionality."""
        assert mock_registry.is_registered("dice_loss")
        assert mock_registry.is_registered("bce_loss")
        assert "dice_loss" in mock_registry.list_available()

        # Test instantiation
        loss_fn = mock_registry.instantiate("dice_loss", smooth=1.0)  # type: ignore
        assert loss_fn is not None

    def test_weighted_sum_combinator(
        self,
        mock_registry: CleanLossRegistry,
        sample_pred_target: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test WeightedSumCombinator in isolation."""
        # Create individual loss components
        dice_loss = mock_registry.instantiate("dice_loss", smooth=1.0)  # type: ignore
        bce_loss = mock_registry.instantiate("bce_loss", reduction="mean")  # type: ignore

        # Create combinator
        combinator = WeightedSumCombinator(
            [dice_loss, bce_loss], weights=[0.6, 0.4]
        )

        # Test forward pass
        pred, target = sample_pred_target
        result = combinator(pred, target)

        assert isinstance(result, torch.Tensor)
        assert result.requires_grad

        # Check that weights are normalized
        weights = combinator.get_component_weights()
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_product_combinator(
        self,
        mock_registry: CleanLossRegistry,
        sample_pred_target: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test ProductCombinator in isolation."""
        # Create individual loss components
        dice_loss = mock_registry.instantiate("dice_loss", smooth=1.0)  # type: ignore
        bce_loss = mock_registry.instantiate("bce_loss", reduction="mean")  # type: ignore

        # Create combinator
        combinator = ProductCombinator([dice_loss, bce_loss])

        # Test forward pass
        pred, target = sample_pred_target
        result = combinator(pred, target)

        assert isinstance(result, torch.Tensor)
        assert result.requires_grad
        assert combinator.get_num_components() == 2

    def test_config_validator(self, mock_registry: CleanLossRegistry) -> None:
        """Test configuration validator in isolation."""
        validator = ConfigValidator(mock_registry)

        # Valid leaf config
        valid_leaf = {"name": "dice_loss", "params": {"smooth": 1.0}}
        validator.validate(valid_leaf)  # Should not raise

        # Valid combinator config
        valid_combinator = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": [
                {"name": "dice_loss", "params": {"smooth": 1.0}},
                {"name": "bce_loss", "params": {"reduction": "mean"}},
            ],
        }
        validator.validate(valid_combinator)  # Should not raise

        # Invalid config - missing fields
        with pytest.raises(ConfigParsingError):
            validator.validate({"params": {}})

    def test_recursive_factory_isolated(
        self,
        mock_registry: CleanLossRegistry,
        sample_pred_target: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test RecursiveLossFactory using the mock registry."""
        factory = RecursiveLossFactory()
        factory.registry = mock_registry  # type: ignore[assignment]
        factory.validator = ConfigValidator(mock_registry)

        # Simple leaf loss
        leaf_config = {"name": "dice_loss", "params": {"smooth": 1.0}}
        loss_fn = factory.create_from_config(leaf_config)

        pred, target = sample_pred_target
        result = loss_fn(pred, target)
        assert isinstance(result, torch.Tensor)

        # Nested combination
        nested_config = {
            "type": "sum",
            "weights": [0.7, 0.3],
            "components": [
                {"name": "dice_loss", "params": {"smooth": 1.0}},
                {
                    "type": "product",
                    "components": [
                        {"name": "dice_loss"},
                        {"name": "bce_loss"},
                    ],
                },
            ],
        }

        nested_loss = factory.create_from_config(nested_config)
        nested_result = nested_loss(pred, target)
        assert isinstance(nested_result, torch.Tensor)

    def test_configuration_summary(
        self, mock_registry: CleanLossRegistry
    ) -> None:
        """Test configuration summary generation."""
        factory = RecursiveLossFactory()
        factory.registry = mock_registry  # type: ignore[assignment]
        factory.validator = ConfigValidator(mock_registry)

        config = {
            "type": "sum",
            "components": [
                {"name": "dice_loss"},
                {
                    "type": "product",
                    "components": [
                        {"name": "bce_loss"},
                        {"name": "dice_loss"},
                    ],
                },
            ],
        }

        summary = factory.get_config_summary(config)
        assert summary["valid"]
        assert summary["depth"] == 3
        assert summary["leaf_count"] == 3
        assert summary["combinator_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
