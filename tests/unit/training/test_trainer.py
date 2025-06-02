# ruff: noqa: PLR2004
# ruff: noqa: PLR0913
"""Tests for the Trainer class."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from pytest import LogCaptureFixture, MonkeyPatch

from src.training.batch_processing import train_step, val_step

# Assuming the necessary modules exist in these paths
from src.training.trainer import Trainer, TrainingComponents
from src.utils.logging import NoOpLogger  # Use NoOpLogger for testing
from src.utils.training.early_stopping import EarlyStopping

# --- Mocks and Fixtures ---


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

    loss_fn = MagicMock(spec=torch.nn.Module)
    loss_fn.return_value = torch.tensor(0.5, requires_grad=True)

    metric_mock = MagicMock()
    metric_mock.return_value = torch.tensor(0.8)
    metrics_dict = MagicMock()
    logger_instance = MagicMock()

    return TrainerMocks(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        metrics_dict=metrics_dict,
        logger_instance=logger_instance,
    )


@pytest.fixture
def base_trainer_cfg() -> DictConfig:
    """Basic Hydra config for the trainer."""
    return OmegaConf.create(
        {
            "training": {
                "epochs": 2,
                "device": "cpu",
                "use_amp": False,
                "gradient_accumulation_steps": 1,
                "verbose": False,
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-3},
                "lr_scheduler": None,
                "scheduler": None,
                "early_stopping": {
                    "_target_": (
                        "src.utils.training.early_stopping.EarlyStopping"
                    ),
                    "monitor": "val_loss",
                    "patience": 2,
                    "mode": "min",
                    "min_delta": 0.001,
                    "verbose": True,
                },
                # Add other necessary fields if init requires them
            }
            # Add other top-level keys if needed by mocks (e.g., logging)
        }
    )


@pytest.fixture
def mock_logger_instance() -> NoOpLogger:
    """Mock logger instance."""
    return NoOpLogger()  # Use NoOpLogger, requires no setup


@pytest.fixture
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a dummy batch of images and masks."""
    images = torch.randn(2, 3, 4, 4)
    masks = torch.randn(2, 1, 4, 4)
    return (images, masks)


@pytest.fixture
def dummy_data_loader() -> (
    torch.utils.data.DataLoader[dict[str, torch.Tensor]]
):
    # Returns a DataLoader with 2 batches of dummy data
    class DummyDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "image": torch.randn(3, 4, 4),
                "mask": torch.randn(1, 4, 4),
            }

    return torch.utils.data.DataLoader(DummyDataset(), batch_size=2)


@pytest.fixture
def dummy_loss() -> torch.nn.MSELoss:
    return torch.nn.MSELoss()


@pytest.fixture
def dummy_metrics() -> dict[str, Any]:
    return {}


# --- Test Cases ---


@patch("src.training.trainer.get_device", return_value=torch.device("cpu"))
@patch("src.training.trainer.create_lr_scheduler")
@patch("src.training.trainer.create_optimizer")
def test_trainer_initialization(
    mock_create_optimizer: MagicMock,
    mock_create_scheduler: MagicMock,
    mock_get_device: MagicMock,
    trainer_mocks_fixture: TrainerMocks,  # Use the new fixture
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
            # Still called, returns None
            mock_create_scheduler.assert_called_once()
            assert trainer.scheduler is None
        mock_get_device.assert_called_once_with("cpu")

    except Exception as e:
        pytest.fail(f"Trainer initialization failed: {e}")


@patch("src.training.trainer.get_device", return_value=torch.device("cpu"))
@patch("src.training.trainer.create_lr_scheduler")
@patch("src.training.trainer.create_optimizer")
@patch("src.training.trainer.handle_epoch_checkpointing")
@patch(
    "src.training.trainer.Trainer._step_scheduler"
)  # Corrected patch target
@patch(
    "src.training.trainer.Trainer.validate",
    return_value={"loss": 0.4, "iou": 0.8},
)
@patch("src.training.trainer.Trainer._train_epoch", return_value=0.5)
def test_trainer_train_loop(
    mock_train_epoch: MagicMock,
    mock_validate: MagicMock,
    mock_step_scheduler: MagicMock,  # Corrected order
    mock_handle_checkpoint: MagicMock,  # Corrected order
    mock_create_optimizer: MagicMock,
    mock_create_scheduler: MagicMock,
    mock_get_device: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
) -> None:
    """Test the main training loop orchestration in train()."""
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.param_groups = [{"lr": 1e-3}]
    mock_create_optimizer.return_value = mock_optimizer
    mock_scheduler_instance = MagicMock()
    mock_create_scheduler.return_value = mock_scheduler_instance
    mock_handle_checkpoint.return_value = float("inf")

    test_cfg = base_trainer_cfg.copy()
    test_cfg.training.epochs = 3
    test_cfg.training.lr_scheduler = OmegaConf.create(
        {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1}
    )

    components = TrainingComponents(
        model=trainer_mocks_fixture.model,
        train_loader=trainer_mocks_fixture.dataloader,
        val_loader=trainer_mocks_fixture.dataloader,
        loss_fn=trainer_mocks_fixture.loss_fn,
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )
    trainer = Trainer(
        components=components,
        cfg=test_cfg,
        logger_instance=trainer_mocks_fixture.logger_instance,
    )

    final_results = trainer.train()

    mock_create_optimizer.assert_called_once()
    mock_create_scheduler.assert_called_once()
    assert mock_train_epoch.call_count == test_cfg.training.epochs
    assert mock_validate.call_count == test_cfg.training.epochs
    if trainer.scheduler:
        assert mock_step_scheduler.call_count == test_cfg.training.epochs
    assert final_results == {"loss": 0.4, "iou": 0.8}
    for i in range(1, test_cfg.training.epochs + 1):
        mock_train_epoch.assert_any_call(i)
    assert mock_handle_checkpoint.call_count == test_cfg.training.epochs


@patch("src.training.trainer.get_device", return_value=torch.device("cpu"))
@patch("src.training.trainer.create_lr_scheduler")
@patch("src.training.trainer.create_optimizer")
def test_train_step_computes_loss_and_backward(
    mock_create_optimizer: MagicMock,
    mock_create_scheduler: MagicMock,
    mock_get_device: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    base_trainer_cfg.training.gradient_accumulation_steps = 2
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_create_optimizer.return_value = mock_optimizer
    mock_create_scheduler.return_value = None
    mock_scaler = MagicMock()
    mock_scaled_loss = MagicMock()
    mock_scaler.scale.return_value = mock_scaled_loss
    trainer_mocks_fixture.model.return_value = torch.ones(2, 1, 4, 4)
    trainer_mocks_fixture.loss_fn.return_value = torch.tensor(
        0.8, requires_grad=True
    )

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
    trainer.scaler = mock_scaler

    result = train_step(
        model=trainer_mocks_fixture.model,
        batch=dummy_batch,
        loss_fn=trainer_mocks_fixture.loss_fn,
        optimizer=mock_optimizer,
        device=mock_get_device(),
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )

    trainer_mocks_fixture.model.assert_called_once()
    trainer_mocks_fixture.loss_fn.assert_called_once()
    assert float(result["loss"]) == pytest.approx(0.8)


@pytest.mark.cuda
@patch("src.training.trainer.get_device", return_value=torch.device("cuda:0"))
@patch("src.training.trainer.create_lr_scheduler")
@patch("src.training.trainer.create_optimizer")
def test_train_step_amp_cuda(
    mock_create_optimizer: MagicMock,
    mock_create_scheduler: MagicMock,
    mock_get_device: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    base_trainer_cfg.training.use_amp = True
    base_trainer_cfg.training.device = "cuda:0"
    base_trainer_cfg.training.gradient_accumulation_steps = 1
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_create_optimizer.return_value = mock_optimizer
    mock_create_scheduler.return_value = None
    mock_scaler = MagicMock()
    mock_scaled_loss = MagicMock()
    mock_scaler.scale.return_value = mock_scaled_loss
    trainer_mocks_fixture.model.return_value = torch.ones(
        2, 1, 4, 4, device="cuda:0"
    )
    trainer_mocks_fixture.loss_fn.return_value = torch.tensor(
        0.5, device="cuda:0", requires_grad=True
    )

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
    trainer.scaler = mock_scaler

    result = train_step(
        model=trainer_mocks_fixture.model,
        batch=dummy_batch,
        loss_fn=trainer_mocks_fixture.loss_fn,
        optimizer=mock_optimizer,
        device=mock_get_device(),
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )

    trainer_mocks_fixture.model.assert_called_once()
    trainer_mocks_fixture.loss_fn.assert_called_once()
    assert float(result["loss"]) == pytest.approx(0.5)


@patch("src.training.trainer.get_device", return_value=torch.device("cpu"))
@patch("src.training.trainer.create_lr_scheduler")
@patch("src.training.trainer.create_optimizer")
def test_train_step_raises_on_forward_error(
    mock_create_optimizer: MagicMock,
    mock_create_scheduler: MagicMock,
    mock_get_device: MagicMock,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    base_trainer_cfg.training.gradient_accumulation_steps = 1
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_create_optimizer.return_value = mock_optimizer
    mock_create_scheduler.return_value = None
    mock_scaler = MagicMock()
    mock_scaler.scale.return_value = MagicMock()
    trainer_mocks_fixture.model.side_effect = RuntimeError("Forward error")

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
    trainer.scaler = mock_scaler

    with pytest.raises(RuntimeError, match="Forward error"):
        train_step(
            model=trainer_mocks_fixture.model,
            batch=dummy_batch,
            loss_fn=trainer_mocks_fixture.loss_fn,
            optimizer=mock_optimizer,
            device=mock_get_device(),
            metrics_dict=trainer_mocks_fixture.metrics_dict,
        )


def test_epoch_level_logging(
    dummy_data_loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    dummy_loss: torch.nn.MSELoss,
    dummy_metrics: dict[str, Any],
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create(
        {
            "training": {
                "epochs": 2,
                "device": "cpu",
                "use_amp": False,
                "gradient_accumulation_steps": 1,
                "checkpoint_dir": str(checkpoint_dir),
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
                "lr_scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 1,
                    "gamma": 0.5,
                },
                "scheduler": None,
                "verbose": False,
            }
        }
    )
    model = torch.nn.Conv2d(3, 1, 1)
    logger = MagicMock()
    exp_manager = MagicMock()
    exp_manager.get_path.return_value = str(checkpoint_dir)
    logger.experiment_manager = exp_manager

    components = TrainingComponents(
        model=model,
        train_loader=dummy_data_loader,
        val_loader=dummy_data_loader,
        loss_fn=dummy_loss,
        metrics_dict=dummy_metrics,
    )
    trainer = Trainer(components=components, cfg=cfg, logger_instance=logger)
    trainer.train()
    assert logger.log_scalar.call_count > 0


def test_batch_level_logging(
    dummy_data_loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    dummy_loss: torch.nn.MSELoss,
    dummy_metrics: dict[str, Any],
    tmp_path: Path,
) -> None:
    log_interval = 2
    num_epochs = 2
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create(
        {
            "training": {
                "epochs": num_epochs,
                "device": "cpu",
                "use_amp": False,
                "gradient_accumulation_steps": 1,
                "checkpoint_dir": str(checkpoint_dir),
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
                "lr_scheduler": None,
                "scheduler": None,
                "verbose": False,
                "log_interval_batches": log_interval,
            }
        }
    )
    model = torch.nn.Conv2d(3, 1, 1)
    logger = MagicMock()
    exp_manager = MagicMock()
    exp_manager.get_path.return_value = str(checkpoint_dir)
    logger.experiment_manager = exp_manager

    components = TrainingComponents(
        model=model,
        train_loader=dummy_data_loader,
        val_loader=dummy_data_loader,
        loss_fn=dummy_loss,
        metrics_dict=dummy_metrics,
    )
    trainer = Trainer(components=components, cfg=cfg, logger_instance=logger)
    trainer.train()
    assert logger.log_scalar.call_count > 0


def test_val_step_returns_metrics(
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    components = TrainingComponents(
        model=trainer_mocks_fixture.model,
        train_loader=trainer_mocks_fixture.dataloader,
        val_loader=trainer_mocks_fixture.dataloader,
        loss_fn=trainer_mocks_fixture.loss_fn,
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )
    Trainer(
        components=components,
        cfg=base_trainer_cfg,
        logger_instance=trainer_mocks_fixture.logger_instance,
    )
    metrics = val_step(
        model=trainer_mocks_fixture.model,
        batch=dummy_batch,
        loss_fn=trainer_mocks_fixture.loss_fn,
        device=torch.device("cpu"),
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "iou" in metrics
    assert "f1" in metrics
    assert all(isinstance(float(v), float) for v in metrics.values())


def test_validate_aggregates_metrics(
    trainer_mocks_fixture: TrainerMocks, base_trainer_cfg: DictConfig
) -> None:
    """Test that validate correctly averages and returns metrics with val_
    prefix."""
    components = TrainingComponents(
        model=trainer_mocks_fixture.model,
        train_loader=MagicMock(),  # Dummy DataLoader
        val_loader=trainer_mocks_fixture.dataloader,  # Use mocked dataloader
        loss_fn=trainer_mocks_fixture.loss_fn,
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )
    trainer = Trainer(
        components=components,
        cfg=base_trainer_cfg,
        logger_instance=trainer_mocks_fixture.logger_instance,
    )
    val_metrics = trainer.validate(epoch=1)
    assert isinstance(val_metrics, dict)
    assert "val_loss" in val_metrics
    assert "val_iou" in val_metrics
    assert "val_f1" in val_metrics
    assert all(isinstance(v, float) for v in val_metrics.values())


def test_step_scheduler_reduce_on_plateau(monkeypatch: MonkeyPatch) -> None:
    scheduler = MagicMock()
    scheduler.__class__.__name__ = "ReduceLROnPlateau"
    optimizer = MagicMock()
    optimizer.param_groups = [{"lr": 0.01}]
    logger = MagicMock()
    metrics = {"val_loss": 0.5}
    from src.utils.training.scheduler_helper import step_scheduler_helper

    lr = step_scheduler_helper(
        scheduler=scheduler,
        optimizer=optimizer,
        monitor_metric="val_loss",
        metrics=metrics,
        logger=logger,
    )
    scheduler.step.assert_called_once_with(0.5)
    logger.info.assert_called()
    assert lr == 0.01

    scheduler.reset_mock()
    logger.reset_mock()
    lr = step_scheduler_helper(
        scheduler=scheduler,
        optimizer=optimizer,
        monitor_metric="val_loss",
        metrics={},
        logger=logger,
    )
    scheduler.step.assert_not_called()
    logger.warning.assert_called()
    assert lr == 0.01


def test_step_scheduler_other_scheduler() -> None:
    scheduler = MagicMock()
    optimizer = MagicMock()
    optimizer.param_groups = [{"lr": 0.02}]
    logger = MagicMock()
    from src.utils.training.scheduler_helper import step_scheduler_helper

    lr = step_scheduler_helper(
        scheduler=scheduler,
        optimizer=optimizer,
        monitor_metric="val_loss",
        metrics={"val_loss": 0.1},
        logger=logger,
    )
    scheduler.step.assert_called_once()
    logger.info.assert_called()
    assert lr == 0.02


def test_trainer_early_stopping(
    monkeypatch: MonkeyPatch,
    trainer_mocks_fixture: TrainerMocks,
    base_trainer_cfg: DictConfig,
) -> None:
    """
    Test that Trainer stops training when early_stopper.step returns True.
    """
    test_cfg = base_trainer_cfg.copy()
    test_cfg.training.epochs = 5

    class DummyEarlyStopping(EarlyStopping):
        def __init__(self) -> None:
            super().__init__(
                patience=7, min_delta=0.0, mode="min", verbose=True
            )
            self.calls = 0
            self.enabled = True

        def step(self, value: float) -> bool:
            self.calls += 1
            return self.calls == 3

        def __call__(self, value: float | None) -> bool:
            return self.step(value or 0.0)

    early_stopper = DummyEarlyStopping()

    def fake_validate(self: Any, epoch: int) -> dict[str, float]:
        return {"val_loss": 0.5}

    monkeypatch.setattr("src.training.trainer.Trainer.validate", fake_validate)

    components = TrainingComponents(
        model=trainer_mocks_fixture.model,
        train_loader=trainer_mocks_fixture.dataloader,
        val_loader=trainer_mocks_fixture.dataloader,
        loss_fn=trainer_mocks_fixture.loss_fn,
        metrics_dict=trainer_mocks_fixture.metrics_dict,
    )
    trainer = Trainer(
        components=components,
        cfg=test_cfg,
        logger_instance=trainer_mocks_fixture.logger_instance,
        early_stopper=early_stopper,
    )
    trainer.early_stopper = early_stopper  # Ensure it's used
    monkeypatch.setattr(trainer, "_train_epoch", lambda epoch: 0.0)

    with patch(
        "src.training.trainer.handle_epoch_checkpointing",
        return_value=float("inf"),
    ):
        trainer.train()
    assert early_stopper.calls == 3


def test_format_metrics() -> None:
    from src.utils.logging.training import format_metrics

    metrics = {"loss": 0.1234, "iou": 0.5678}
    formatted = format_metrics(metrics)
    assert "Loss: 0.1234" in formatted
    assert "Iou: 0.5678" in formatted
    assert "|" in formatted


def test_log_validation_results(caplog: LogCaptureFixture) -> None:
    from src.utils.logging.training import log_validation_results

    class DummyLogger:
        def __init__(self) -> None:
            self.last_msg: str | None = None

        def info(self, msg: str) -> None:
            self.last_msg = msg

    logger = DummyLogger()
    metrics = {"val_loss": 0.1, "val_iou": 0.9}
    log_validation_results(logger, 5, metrics)
    assert logger.last_msg is not None and logger.last_msg.startswith(
        "Epoch 5 | Validation Results | "
    )
    assert (
        logger.last_msg is not None and "Val_loss: 0.1000" in logger.last_msg
    )
    assert logger.last_msg is not None and "Val_iou: 0.9000" in logger.last_msg


def test_amp_autocast_context_manager() -> None:
    from src.utils.training.amp_utils import amp_autocast

    with amp_autocast(False):
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2
        assert y.item() == 2.0
    with amp_autocast(True):
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2
        assert y.item() == 2.0


def test_optimizer_step_with_accumulation_no_amp() -> None:
    from src.utils.training.amp_utils import optimizer_step_with_accumulation

    optimizer = MagicMock()
    scaler = None
    loss = torch.tensor(1.0, requires_grad=True)
    grad_accum_steps = 2
    optimizer_step_with_accumulation(
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
        grad_accum_steps=grad_accum_steps,
        batch_idx=0,
        use_amp=False,
    )
    optimizer.step.assert_not_called()
    optimizer_step_with_accumulation(
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
        grad_accum_steps=grad_accum_steps,
        batch_idx=1,
        use_amp=False,
    )
    optimizer.step.assert_called_once()
    optimizer.zero_grad.assert_called()


def test_optimizer_step_with_accumulation_amp() -> None:
    from src.utils.training.amp_utils import optimizer_step_with_accumulation

    optimizer = MagicMock()
    scaler = MagicMock()
    loss = torch.tensor(1.0, requires_grad=True)
    grad_accum_steps = 2
    optimizer_step_with_accumulation(
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
        grad_accum_steps=grad_accum_steps,
        batch_idx=0,
        use_amp=True,
    )
    scaler.scale.assert_called_with(loss / grad_accum_steps)
    scaler.step.assert_not_called()
    optimizer_step_with_accumulation(
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
        grad_accum_steps=grad_accum_steps,
        batch_idx=1,
        use_amp=True,
    )
    scaler.step.assert_called_once_with(optimizer)
    scaler.update.assert_called()
    optimizer.zero_grad.assert_called()


def test_validate_trainer_config_valid() -> None:
    from src.training.config_validation import validate_trainer_config

    class DummyCfg:
        epochs = 5
        device = "cpu"
        optimizer = {"_target_": "torch.optim.Adam", "lr": 0.001}
        gradient_accumulation_steps = 1

    validate_trainer_config(DummyCfg())


def test_validate_trainer_config_missing_field() -> None:
    from src.training.config_validation import validate_trainer_config

    class DummyCfg:
        device = "cpu"
        optimizer = {"_target_": "torch.optim.Adam", "lr": 0.001}
        gradient_accumulation_steps = 1

    with pytest.raises(ValueError):
        validate_trainer_config(DummyCfg())


def test_validate_trainer_config_invalid_type() -> None:
    from src.training.config_validation import validate_trainer_config

    class DummyCfg:
        epochs = "five"
        device = "cpu"
        optimizer = {"_target_": "torch.optim.Adam", "lr": 0.001}
        gradient_accumulation_steps = 1

    with pytest.raises(TypeError):
        validate_trainer_config(DummyCfg())


def test_validate_trainer_config_invalid_optimizer() -> None:
    from src.training.config_validation import validate_trainer_config

    class DummyCfg:
        epochs = 5
        device = "cpu"
        optimizer = {"lr": 0.001}
        gradient_accumulation_steps = 1

    with pytest.raises(ValueError):
        validate_trainer_config(DummyCfg())
