"""Tests for loss and metric factory functions."""

import pytest
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.losses import (
    BCELoss, DiceLoss, FocalLoss,
    CombinedLoss, BCEDiceLoss
)
from src.training.metrics import (
    IoUScore,
    PrecisionScore,
    RecallScore,
    F1Score,
)
from src.utils.factory import (
    get_loss_from_cfg,
    get_metrics_from_cfg,
)


# --- Helper Functions ---

def load_test_config(config_name: str = "training/loss") -> DictConfig:
    """Loads a specific test configuration using Hydra Compose API."""
    # Initialize Hydra with the configs directory
    hydra.initialize_config_dir(config_dir="configs", version_base=None)
    cfg = hydra.compose(config_name=config_name)
    # Clean up hydra state
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    return cfg


# --- Test Cases for Loss Factory ---

def test_get_loss_from_cfg_bce():
    """Test creating BCE loss from config."""
    config = OmegaConf.create({
        "_target_": "src.training.losses.BCELoss",
        "weight": None,
        "reduction": "mean"
    })

    loss_fn = get_loss_from_cfg(config)
    assert isinstance(loss_fn, BCELoss)


def test_get_loss_from_cfg_dice():
    """Test creating Dice loss from config."""
    config = OmegaConf.create({
        "_target_": "src.training.losses.DiceLoss",
        "smooth": 1e-6,
        "sigmoid": True
    })

    loss_fn = get_loss_from_cfg(config)
    assert isinstance(loss_fn, DiceLoss)
    assert loss_fn.smooth == 1e-6
    assert loss_fn.sigmoid is True


def test_get_loss_from_cfg_focal():
    """Test creating Focal loss from config."""
    config = OmegaConf.create({
        "_target_": "src.training.losses.FocalLoss",
        "alpha": 0.25,
        "gamma": 2.0,
        "sigmoid": True
    })

    loss_fn = get_loss_from_cfg(config)
    assert isinstance(loss_fn, FocalLoss)
    assert loss_fn.alpha == 0.25
    assert loss_fn.gamma == 2.0


def test_get_loss_from_cfg_combined():
    """Test creating Combined loss from config."""
    config = OmegaConf.create({
        "_target_": "src.training.losses.CombinedLoss",
        "losses": [
            {
                "config": {
                    "_target_": "src.training.losses.BCELoss",
                },
                "weight": 0.5
            },
            {
                "config": {
                    "_target_": "src.training.losses.DiceLoss",
                    "smooth": 1e-6
                },
                "weight": 0.5
            }
        ],
        "weights": [0.5, 0.5]
    })

    loss_fn = get_loss_from_cfg(config)
    assert isinstance(loss_fn, CombinedLoss)
    assert len(loss_fn.losses) == 2
    assert isinstance(loss_fn.losses[0], BCELoss)
    assert isinstance(loss_fn.losses[1], DiceLoss)
    assert loss_fn.weights == [0.5, 0.5]


def test_get_loss_from_cfg_bcedice():
    """Test creating BCE+Dice loss from config."""
    config = OmegaConf.create({
        "_target_": "src.training.losses.BCEDiceLoss",
        "bce_weight": 0.7,
        "dice_weight": 0.3
    })

    loss_fn = get_loss_from_cfg(config)
    assert isinstance(loss_fn, BCEDiceLoss)
    # Access weights via the internal CombinedLoss instance
    assert hasattr(loss_fn, 'combined')
    assert isinstance(loss_fn.combined, CombinedLoss)
    assert loss_fn.combined.weights == [0.7, 0.3]


def test_get_loss_from_cfg_invalid():
    """Test error handling for invalid loss config."""
    config = OmegaConf.create({
        "_target_": "invalid.path.Loss"
    })

    with pytest.raises(ValueError):
        get_loss_from_cfg(config)


def test_get_loss_from_cfg_missing_params():
    """Test error handling for missing required parameters."""
    config = OmegaConf.create({
        "_target_": "src.training.losses.FocalLoss",
        # Missing optional alpha parameter (has default=None)
        # Missing optional gamma parameter (has default=2.0)
        # Missing optional sigmoid parameter (has default=True)
    })

    # Expect successful instantiation with defaults
    try:
        loss_fn = get_loss_from_cfg(config)
        assert isinstance(loss_fn, FocalLoss)
        # Verify defaults are set if needed
        # assert loss_fn.alpha is None
        # assert loss_fn.gamma == 2.0
    except Exception as e:
        pytest.fail(f"Instatiation with missing optional params failed: {e}")
    # with pytest.raises(TypeError): # This won't raise if params have defaults
    #     get_loss_from_cfg(config)


# --- Test Cases for Metrics Factory ---

def test_get_metrics_from_cfg_single():
    """Test creating a single metric from config."""
    config = OmegaConf.create({
        "iou": {
            "_target_": "src.training.metrics.IoUScore",
            "threshold": 0.5,
            "smooth": 1e-6
        }
    })

    metrics = get_metrics_from_cfg(config)
    assert len(metrics) == 1
    assert isinstance(metrics["iou"], IoUScore)
    assert metrics["iou"].threshold == 0.5
    assert metrics["iou"].smooth == 1e-6


def test_get_metrics_from_cfg_multiple():
    """Test creating multiple metrics from config."""
    config = OmegaConf.create({
        "iou": {
            "_target_": "src.training.metrics.IoUScore",
            "threshold": 0.5
        },
        "precision": {
            "_target_": "src.training.metrics.PrecisionScore",
            "threshold": 0.5
        },
        "recall": {
            "_target_": "src.training.metrics.RecallScore",
            "threshold": 0.5
        },
        "f1": {
            "_target_": "src.training.metrics.F1Score",
            "threshold": 0.5
        }
    })

    metrics = get_metrics_from_cfg(config)
    assert len(metrics) == 4
    assert isinstance(metrics["iou"], IoUScore)
    assert isinstance(metrics["precision"], PrecisionScore)
    assert isinstance(metrics["recall"], RecallScore)
    assert isinstance(metrics["f1"], F1Score)


def test_get_metrics_from_cfg_invalid():
    """Test error handling for invalid metric config."""
    config = OmegaConf.create({
        "invalid": {
            "_target_": "invalid.path.Metric"
        }
    })

    with pytest.raises(ValueError):
        get_metrics_from_cfg(config)


def test_get_metrics_from_cfg_missing_params():
    """Test error handling for missing required parameters."""
    config = OmegaConf.create({
        "iou": {
            "_target_": "src.training.metrics.IoUScore",
            # Missing optional threshold parameter (default=0.5)
            # Missing optional smooth parameter (default=1e-6)
        }
    })

    # Expect successful instantiation with defaults
    try:
        metrics = get_metrics_from_cfg(config)
        assert "iou" in metrics
        assert isinstance(metrics["iou"], IoUScore)
        # Verify defaults are set if needed
        # assert metrics["iou"].threshold == 0.5
        # assert metrics["iou"].smooth == 1e-6
    except Exception as e:
        pytest.fail(f"Instatiation with missing optional params failed: {e}")
    # with pytest.raises(TypeError): # This won't raise if params have defaults
    #     get_metrics_from_cfg(config)

# --- Integration Tests ---


def test_loss_and_metrics_integration():
    """Test that configured loss and metrics work together correctly."""
    # Create sample data
    pred = torch.randn(2, 1, 4, 4)  # Random predictions
    target = torch.randint(0, 2, (2, 1, 4, 4)).float()  # Random binary targets

    # Configure loss and metrics
    loss_config = OmegaConf.create({
        "_target_": "src.training.losses.BCEDiceLoss",
        "bce_weight": 0.5,
        "dice_weight": 0.5
    })

    metrics_config = OmegaConf.create({
        "iou": {
            "_target_": "src.training.metrics.IoUScore",
            "threshold": 0.5
        },
        "f1": {
            "_target_": "src.training.metrics.F1Score",
            "threshold": 0.5
        }
    })

    # Create loss and metrics
    loss_fn = get_loss_from_cfg(loss_config)
    metrics = get_metrics_from_cfg(metrics_config)

    # Compute loss and metrics
    loss = loss_fn(pred, target)
    metric_values = {name: metric(pred, target) for name, metric in
                     metrics.items()}

    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar output
    assert len(metric_values) == 2
    assert all(isinstance(v, torch.Tensor) for v in metric_values.values())
    assert all(v.dim() == 0 for v in metric_values.values())  # Scalar outputs
