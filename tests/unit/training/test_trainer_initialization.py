# ruff: noqa: PLR2004
# ruff: noqa: PLR0913
"""Tests for Trainer class initialization and configuration."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from crackseg.training.trainer import Trainer, TrainingComponents
from crackseg.utils.logging import NoOpLogger


@dataclass
class TrainerMocks:
    """Dataclass to group trainer-related mocks."""

    model: MagicMock
    dataloader: MagicMock
    loss_fn: MagicMock
    metrics_dict: MagicMock
    logger_instance: MagicMock


@pytest.fixture
def trainer_mocks_fixture() -> TrainerMocks:
    """Provides an instance of TrainerMocks with mocked components."""
    model = MagicMock(spec=torch.nn.Module)
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    model.to.return_value = model

    # Create a proper mock for DataLoader
    dataloader = MagicMock()
    dataloader.__len__ = MagicMock(return_value=10)
    dataloader.__iter__ = MagicMock(
        return_value=iter(
            [
                (torch.randn(2, 3, 4, 4), torch.randn(2, 1, 4, 4))
                for _ in range(10)
            ]
        )
    )

    loss_fn = MagicMock(return_value=torch.tensor(0.5))
    metrics_dict = {
        "iou": MagicMock(return_value=0.8),
        "f1": MagicMock(return_value=0.7),
    }
    logger_instance = MagicMock(spec=NoOpLogger)

    return TrainerMocks(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        metrics_dict=metrics_dict,
        logger_instance=logger_instance,
    )


@pytest.fixture
def base_trainer_cfg() -> DictConfig:
    """Provides a base configuration for Trainer tests."""
    cfg_dict = {
        "training": {
            "epochs": 10,
            "optimizer": {"name": "Adam", "lr": 1e-3},
            "lr_scheduler": None,
            "amp": {"enabled": False},
            "gradient_accumulation_steps": 1,
        },
        "device": "cpu",
        "checkpointing": {"save_every": 5, "save_top_k": 3},
        "logging": {"log_every": 1},
        "early_stopping": {"enabled": False},
    }
    return OmegaConf.create(cfg_dict)


# Initialization Tests
@patch(
    "crackseg.training.trainer.get_device", return_value=torch.device("cpu")
)
@patch("crackseg.training.trainer.create_lr_scheduler")
@patch("crackseg.training.trainer.create_optimizer")
def test_trainer_initialization(
    mock_create_optimizer: MagicMock,
    mock_create_scheduler: MagicMock,
    mock_get_device: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
) -> None:
    """Test if the Trainer class can be initialized correctly."""
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.param_groups = [{"lr": 1e-3}]
    mock_create_optimizer.return_value = mock_optimizer
    mock_create_scheduler.return_value = None

    try:
        components = TrainingComponents(
            model=trainer_mocks_fixture.model,
            train_loader=trainer_mocks_fixture.dataloader,
            val_loader=trainer_mocks_fixture.dataloader,
            loss_fn=trainer_mocks_fixture.loss_fn,
            metrics_dict=trainer_mocks_fixture.metrics_dict,
        )
        trainer = Trainer(
            components=components,
            cfg=base_trainer_cfg,
            logger_instance=trainer_mocks_fixture.logger_instance,
        )

        assert trainer.model == trainer_mocks_fixture.model
        assert trainer.train_loader == trainer_mocks_fixture.dataloader
        assert trainer.optimizer is mock_optimizer
        assert trainer.scheduler is None

        trainer_mocks_fixture.model.to.assert_called_once_with(
            torch.device("cpu")
        )
        mock_create_optimizer.assert_called_once()
        if base_trainer_cfg.training.lr_scheduler:
            mock_create_scheduler.assert_called_once()
        else:
            mock_create_scheduler.assert_called_once()
            assert trainer.scheduler is None
        mock_get_device.assert_called_once_with("cpu")

    except Exception as e:
        pytest.fail(f"Trainer initialization failed: {e}")


def test_validate_trainer_config_valid() -> None:
    """Test that a valid trainer config passes validation."""
    from crackseg.training.trainer import validate_trainer_config

    config = {
        "training": {"epochs": 10, "optimizer": {"name": "Adam"}},
        "device": "cpu",
    }
    # Should not raise any exception
    validate_trainer_config(OmegaConf.create(config))


def test_validate_trainer_config_missing_field() -> None:
    """Test that trainer config validation catches missing required fields."""
    from crackseg.training.trainer import validate_trainer_config

    config = {"device": "cpu"}  # Missing 'training' section
    with pytest.raises((KeyError, AttributeError)):
        validate_trainer_config(OmegaConf.create(config))


def test_validate_trainer_config_invalid_type() -> None:
    """Test that trainer config validation catches invalid field types."""
    from crackseg.training.trainer import validate_trainer_config

    config = {
        "training": {
            "epochs": "invalid_string",
            "optimizer": {"name": "Adam"},
        },
        "device": "cpu",
    }
    with pytest.raises((TypeError, ValueError)):
        validate_trainer_config(OmegaConf.create(config))


def test_validate_trainer_config_invalid_optimizer() -> None:
    """Test that trainer config validation catches invalid optimizer names."""
    from crackseg.training.trainer import validate_trainer_config

    config = {
        "training": {"epochs": 10, "optimizer": {"name": "InvalidOptimizer"}},
        "device": "cpu",
    }
    with pytest.raises(ValueError):
        validate_trainer_config(OmegaConf.create(config))
