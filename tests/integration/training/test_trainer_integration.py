"""Integration tests for the Trainer class."""

import os
import pathlib
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from crackseg.training.trainer import Trainer, TrainingComponents
from crackseg.utils.logging import NoOpLogger


def get_dummy_data_loader(
    num_batches: int = 4,
    batch_size: int = 2,
    shape: tuple[int, ...] = (3, 4, 4),
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    class DummyDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return num_batches * batch_size

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {"image": torch.randn(*shape), "mask": torch.randn(1, 4, 4)}

    return torch.utils.data.DataLoader(DummyDataset(), batch_size=batch_size)


def get_dummy_model() -> torch.nn.Module:
    return torch.nn.Conv2d(3, 1, 1)


def get_dummy_loss() -> torch.nn.Module:
    return torch.nn.MSELoss()


def get_dummy_metrics() -> dict[str, Any]:
    return {}


def integration_test_trainer_checkpoint_resume(
    tmp_path: pathlib.Path, use_amp: bool = False, grad_accum_steps: int = 1
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(
        {
            "training": {
                "epochs": 2,
                "device": "cpu",
                "use_amp": use_amp,
                "gradient_accumulation_steps": grad_accum_steps,
                "checkpoint_dir": str(checkpoint_dir),
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 1,
                    "gamma": 0.5,
                },
                "verbose": False,
                "log_interval_batches": 0,
            }
        }
    )

    model = get_dummy_model()
    train_loader = get_dummy_data_loader()
    val_loader = get_dummy_data_loader()
    loss_fn = get_dummy_loss()
    metrics = get_dummy_metrics()
    logger = NoOpLogger()

    components = TrainingComponents(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics,
    )

    trainer = Trainer(
        components=components,
        cfg=cfg,
        logger_instance=logger,
    )
    trainer.train()

    # Verify that checkpoints were saved
    # (in outputs/checkpoints, not in tmp_path)
    actual_checkpoint_dir = "outputs/checkpoints"
    os.makedirs(actual_checkpoint_dir, exist_ok=True)
    last_ckpt = os.path.join(actual_checkpoint_dir, "checkpoint_last.pth")
    assert os.path.exists(
        last_ckpt
    ), "No checkpoint saved in outputs/\
checkpoints"

    # Load from checkpoint and continue training
    cfg.training["checkpoint_load_path"] = last_ckpt
    cfg.training["epochs"] = 3  # Train one more epoch

    model2 = get_dummy_model()
    components2 = TrainingComponents(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics,
    )
    trainer2 = Trainer(
        components=components2,
        cfg=cfg,
        logger_instance=logger,
    )
    trainer2.train()

    assert (
        trainer2.start_epoch == 3  # noqa: PLR2004
    ), f"Training did not continue from the \
correct epoch: {trainer2.start_epoch}"  # noqa: PLR2004
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=False):
        assert not torch.equal(
            p1, p2
        ), "Model weights did not change \
after resuming and training"

    # Clean up the checkpoint directory
    shutil.rmtree(actual_checkpoint_dir, ignore_errors=True)


def test_trainer_integration_checkpoint_resume_cpu(
    tmp_path: pathlib.Path,
) -> None:
    integration_test_trainer_checkpoint_resume(
        tmp_path, use_amp=False, grad_accum_steps=1
    )


def test_trainer_integration_checkpoint_resume_amp(
    tmp_path: pathlib.Path,
) -> None:
    integration_test_trainer_checkpoint_resume(
        tmp_path, use_amp=True, grad_accum_steps=1
    )


def test_trainer_integration_checkpoint_resume_accum(
    tmp_path: pathlib.Path,
) -> None:
    integration_test_trainer_checkpoint_resume(
        tmp_path, use_amp=False, grad_accum_steps=2
    )


@pytest.fixture
def mock_components() -> TrainingComponents:
    """Fixture to create mock components for the Trainer."""
    model = nn.Linear(10, 2)
    train_dataset = TensorDataset(
        torch.randn(20, 10), torch.randint(0, 2, (20,))
    )
    val_dataset = TensorDataset(
        torch.randn(10, 10), torch.randint(0, 2, (10,))
    )
    train_loader = DataLoader(train_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=2)
    loss_fn = nn.CrossEntropyLoss()

    class MockMetric:
        def reset(self) -> None:
            pass

        def update(self, y_pred: Any, y_true: Any) -> None:
            pass

        def compute(self) -> float:
            return 0.5

    metrics_dict = {"iou": MockMetric()}

    return TrainingComponents(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics_dict,
    )


@pytest.fixture
def mock_config() -> DictConfig:
    """Fixture to create a mock OmegaConf configuration."""
    return OmegaConf.create(
        {
            "training": {
                "epochs": 2,
                "device": "cpu",
                "use_amp": False,
                "gradient_accumulation_steps": 1,
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-3},
                "checkpoints": {"save_best": {"enabled": False}},
                "early_stopping": {"enabled": False},
            },
            "experiment": {"dir": "/tmp/mock_experiment"},
        }
    )


def test_trainer_runs_without_errors(
    mock_components: TrainingComponents,
    mock_config: DictConfig,
    tmp_path: Path,
):
    """
    Tests that the Trainer can initialize and run a basic training loop
    without raising exceptions.
    """
    mock_config.experiment.dir = str(tmp_path)

    try:
        trainer = Trainer(components=mock_components, cfg=mock_config)
        final_metrics = trainer.train()

        assert isinstance(final_metrics, dict)
        assert "val_loss" in final_metrics
        assert final_metrics["val_loss"] > 0

    except Exception as e:
        pytest.fail(f"Trainer failed to run: {e}")
