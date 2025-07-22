# ruff: noqa: PLR2004
# ruff: noqa: PLR0913
"""Tests for Trainer class training loop functionality."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig

from crackseg.training.batch_processing import train_step
from crackseg.training.trainer import Trainer, TrainingComponents

from .test_trainer_initialization import TrainerMocks


@patch(
    "crackseg.training.trainer.get_device", return_value=torch.device("cpu")
)
@patch("crackseg.training.trainer.create_lr_scheduler")
@patch("crackseg.training.trainer.create_optimizer")
@patch("crackseg.training.trainer.handle_epoch_checkpointing")
@patch("crackseg.training.trainer.Trainer._step_scheduler")
@patch(
    "crackseg.training.trainer.Trainer.validate",
    return_value={"loss": 0.4, "iou": 0.8},
)
@patch("crackseg.training.trainer.Trainer._train_epoch", return_value=0.5)
def test_trainer_train_loop(
    mock_train_epoch: MagicMock,
    mock_validate: MagicMock,
    mock_step_scheduler: MagicMock,
    mock_handle_checkpoint: MagicMock,
    mock_create_optimizer: MagicMock,
    mock_create_scheduler: MagicMock,
    mock_get_device: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
) -> None:
    """Test the training loop runs for specified epochs."""
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.param_groups = [{"lr": 1e-3}]
    mock_create_optimizer.return_value = mock_optimizer
    mock_create_scheduler.return_value = None

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

    trainer.train()

    assert mock_train_epoch.call_count == base_trainer_cfg.training.epochs
    assert mock_validate.call_count == base_trainer_cfg.training.epochs


@pytest.fixture
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Provides a dummy batch for testing."""
    inputs = torch.randn(2, 3, 4, 4)
    targets = torch.randn(2, 1, 4, 4)
    return inputs, targets


def test_train_step_computes_loss_and_backward(
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test that train_step correctly computes loss and performs backward
    pass."""
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

    mock_optimizer = MagicMock()
    trainer.optimizer = mock_optimizer

    # Mock model forward pass
    mock_output = torch.randn(2, 1, 4, 4, requires_grad=True)
    trainer_mocks_fixture.model.return_value = mock_output

    # Mock loss function
    mock_loss = torch.tensor(0.5, requires_grad=True)
    trainer_mocks_fixture.loss_fn.return_value = mock_loss

    metrics = train_step(
        model=trainer_mocks_fixture.model,
        batch=dummy_batch,
        optimizer=mock_optimizer,
        loss_fn=trainer_mocks_fixture.loss_fn,
        device=torch.device("cpu"),
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    trainer_mocks_fixture.model.assert_called_once()
    trainer_mocks_fixture.loss_fn.assert_called_once()


@patch("torch.cuda.autocast")
def test_train_step_amp_cuda(
    mock_autocast: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test train_step with AMP enabled on CUDA."""
    # Enable AMP in config
    base_trainer_cfg.training.amp = {"enabled": True}

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

    mock_optimizer = MagicMock()
    trainer.optimizer = mock_optimizer

    # Mock scaler for AMP
    # mock_scaler = MagicMock()  # Not used, removed to avoid linter warning

    # Mock model forward pass
    mock_output = torch.randn(2, 1, 4, 4, requires_grad=True)
    trainer_mocks_fixture.model.return_value = mock_output

    # Mock loss function
    mock_loss = torch.tensor(0.5, requires_grad=True)
    trainer_mocks_fixture.loss_fn.return_value = mock_loss

    metrics = train_step(
        model=trainer_mocks_fixture.model,
        batch=dummy_batch,
        optimizer=mock_optimizer,
        loss_fn=trainer_mocks_fixture.loss_fn,
        device=torch.device("cuda"),
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )

    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_train_step_raises_on_forward_error(
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test that train_step properly handles forward pass errors."""
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

    mock_optimizer = MagicMock()
    trainer.optimizer = mock_optimizer

    # Make model forward pass raise an exception
    trainer_mocks_fixture.model.side_effect = RuntimeError(
        "Forward pass failed"
    )

    with pytest.raises(RuntimeError, match="Forward pass failed"):
        train_step(
            model=trainer_mocks_fixture.model,
            batch=dummy_batch,
            optimizer=mock_optimizer,
            loss_fn=trainer_mocks_fixture.loss_fn,
            device=torch.device("cpu"),
            metrics_dict=trainer_mocks_fixture.metrics_dict,
        )


@patch("crackseg.training.trainer.Trainer._train_epoch")
def test_epoch_level_logging(
    mock_train_epoch: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
) -> None:
    """Test that epoch-level logging is performed correctly."""
    mock_train_epoch.return_value = 0.5

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

    # Mock validate to return some metrics
    trainer.validate = MagicMock(
        return_value={"val_loss": 0.4, "val_iou": 0.8}
    )

    trainer.train()

    # Check that logging methods were called
    assert trainer_mocks_fixture.logger_instance.log_scalar.call_count > 0


@patch("crackseg.training.trainer.train_step")
def test_batch_level_logging(
    mock_train_step: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
) -> None:
    """Test that batch-level logging is performed correctly."""
    mock_train_step.return_value = {"loss": 0.5, "iou": 0.8}

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

    # Train for one epoch to test batch logging
    trainer._train_epoch(epoch=1)

    # Check that batch-level logging was performed
    logger = trainer_mocks_fixture.logger_instance
    assert logger.log_scalar.call_count > 0
