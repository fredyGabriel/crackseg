from typing import Any

import pytest
import torch

from src.training.losses.recursive_factory import parse_loss_config


@pytest.fixture
def sample_pred_target() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fixture that returns a pair of prediction and target tensors for testing.
    """
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    return pred, target


def test_simple_leaf_loss(
    sample_pred_target: tuple[torch.Tensor, torch.Tensor],
) -> None:
    config = {"name": "dice_loss", "params": {"smooth": 1.0}}
    loss_fn = parse_loss_config(config)
    pred, target = sample_pred_target
    result = loss_fn(pred, target)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0


def test_weighted_sum_of_leaves(
    sample_pred_target: tuple[torch.Tensor, torch.Tensor],
) -> None:
    config = {
        "type": "sum",
        "weights": [0.6, 0.4],
        "components": [
            {"name": "dice_loss", "params": {"smooth": 1.0}},
            {"name": "bce_loss", "params": {"reduction": "mean"}},
        ],
    }
    loss_fn = parse_loss_config(config)
    pred, target = sample_pred_target
    result = loss_fn(pred, target)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0


def test_nested_combination(
    sample_pred_target: tuple[torch.Tensor, torch.Tensor],
) -> None:
    config = {
        "type": "sum",
        "weights": [0.7, 0.3],
        "components": [
            {"name": "dice_loss", "params": {"smooth": 1.0}},
            {
                "type": "product",
                "components": [
                    {
                        "name": "focal_loss",
                        "params": {"alpha": 0.25, "gamma": 2.0},
                    },
                    {"name": "bce_loss", "params": {"reduction": "mean"}},
                ],
            },
        ],
    }
    loss_fn = parse_loss_config(config)
    pred, target = sample_pred_target
    result = loss_fn(pred, target)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0


def test_invalid_combination_type() -> None:
    config = {"type": "unknown", "components": []}
    with pytest.raises(ValueError, match="Tipo de combinación no soportado"):
        parse_loss_config(config)


def test_missing_required_fields() -> None:
    config: dict[str, Any] = {"params": {}}
    with pytest.raises(ValueError, match="falta 'type' o 'name'"):
        parse_loss_config(config)


def test_weights_length_mismatch() -> None:
    config = {
        "type": "sum",
        "weights": [0.5],
        "components": [
            {"name": "dice_loss"},
            {"name": "bce_loss"},
        ],
    }
    with pytest.raises(ValueError, match="The number of weights must match"):
        parse_loss_config(config)


def test_weights_sum_zero() -> None:
    config = {
        "type": "sum",
        "weights": [0.0, 0.0],
        "components": [
            {"name": "dice_loss"},
            {"name": "bce_loss"},
        ],
    }
    with pytest.raises(
        ValueError, match="The sum of the weights must be positive"
    ):
        parse_loss_config(config)
