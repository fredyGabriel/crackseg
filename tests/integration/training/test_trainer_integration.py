import os
import pathlib
import shutil
from typing import Any

import torch
from omegaconf import OmegaConf

from src.training.trainer import Trainer, TrainingComponents
from src.utils.logging import NoOpLogger


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
