"""
Unit tests for the clean recursive loss factory architecture.
"""

import pytest
import torch

from crackseg.training.losses.factory import factory


@pytest.fixture
def sample_pred_target():
    """Sample prediction and target tensors for testing."""
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    return pred, target


class TestCleanRecursiveFactory:
    """Test the clean recursive loss factory implementation."""

    def test_simple_leaf_loss_creation(
        self,
        sample_pred_target: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test creating a simple leaf loss."""
        config = {
            "name": "dice_loss",
            "params": {"smooth": 1.0},
        }

        loss_fn = factory.create_from_config(config)
        pred, target = sample_pred_target

        result = loss_fn(pred, target)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_weighted_sum_combination(
        self,
        sample_pred_target: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test creating a weighted sum combination."""
        config = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": [
                {"name": "dice_loss", "params": {"smooth": 1.0}},
                {"name": "bce_loss", "params": {"reduction": "mean"}},
            ],
        }

        loss_fn = factory.create_from_config(config)
        pred, target = sample_pred_target

        result = loss_fn(pred, target)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {"name": "dice_loss", "params": {"smooth": 1.0}}
        assert factory.validate_config(valid_config)

        # Invalid config - missing required fields
        invalid_config: dict[str, object] = {"params": {}}
        assert not factory.validate_config(invalid_config)
