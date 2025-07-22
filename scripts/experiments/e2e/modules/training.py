"""Training functions for end-to-end pipeline testing."""

from collections.abc import Callable
from typing import Any

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from .dataclasses import TrainingRunArgs
from .utils import get_metrics_from_cfg

NO_CHANNEL_DIM = 3


def _initialize_training_components(cfg: Any, device: torch.device) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    Any,
    dict[str, Any],
    Any,
]:
    """
    Initializes model, loss, optimizer, scheduler, metrics, and AMP scaler.
    """
    from crackseg.model.factory import create_unet
    from crackseg.training.factory import create_lr_scheduler
    from crackseg.utils.factory import get_loss_fn, get_optimizer
    from crackseg.utils.logging import get_logger

    logger = get_logger("E2ETestTraining")

    logger.info("Creating model and training components...")
    model = create_unet(cfg.model).to(device)
    loss_fn = get_loss_fn(cfg.training.loss)
    # Si loss_fn no es nn.Module, lo envuelvo en nn.Module para
    # cumplir el tipado estricto
    if not isinstance(loss_fn, torch.nn.Module):

        class LossWrapper(torch.nn.Module):
            def __init__(self, fn: Callable[..., Any]) -> None:
                super().__init__()
                self.fn = fn

            def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
                return self.fn(*args, **kwargs)

        loss_fn = LossWrapper(loss_fn)
    optimizer = get_optimizer(list(model.parameters()), cfg.training.optimizer)
    lr_scheduler = create_lr_scheduler(optimizer, cfg.training.scheduler)
    metrics_dict = get_metrics_from_cfg(cfg.evaluation.metrics)
    use_amp = cfg.training.get("amp_enabled", False)
    scaler = (
        torch.cuda.amp.GradScaler()
        if use_amp and device.type == "cuda"
        else None
    )
    logger.info(f"AMP Enabled: {scaler is not None}")
    logger.info("Training components initialized.")
    return model, loss_fn, optimizer, lr_scheduler, metrics_dict, scaler


def _run_train_epoch(
    args: TrainingRunArgs, train_loader: DataLoader[Any]
) -> tuple[float, dict[str, float]]:
    """Runs a single training epoch and returns loss and metrics."""
    from crackseg.utils.logging import get_logger

    logger = get_logger("E2ETestTraining")

    args.model.train()
    epoch_loss = 0.0
    epoch_metrics = dict.fromkeys(args.metrics_dict.keys(), 0.0)

    for batch_idx, batch in enumerate(train_loader):
        inputs, targets = (
            batch["image"].to(args.device),
            batch["mask"].to(args.device),
        )
        if inputs.shape[-1] == NO_CHANNEL_DIM:
            inputs = inputs.permute(0, 3, 1, 2)
        if len(targets.shape) == NO_CHANNEL_DIM:
            targets = targets.unsqueeze(1)

        args.optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=(args.scaler is not None)):
            outputs = args.model(inputs)
            loss = args.loss_fn(outputs, targets)

        if args.scaler:
            args.scaler.scale(loss).backward()
            args.scaler.step(args.optimizer)
            args.scaler.update()
        else:
            loss.backward()
            args.optimizer.step()

        epoch_loss += loss.item()
        with torch.no_grad():
            for k, metric_fn in args.metrics_dict.items():
                epoch_metrics[k] += metric_fn(outputs, targets).item()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            logger.info(
                f"Train Batch: {batch_idx + 1}/{len(train_loader)}, Loss: "
                f"{loss.item():.4f}"
            )

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_metrics = {
        k: v / len(train_loader) for k, v in epoch_metrics.items()
    }
    return avg_epoch_loss, avg_epoch_metrics


def _run_val_epoch(
    args: TrainingRunArgs, val_loader: DataLoader[Any]
) -> tuple[float, dict[str, float]]:
    """Runs a single validation epoch and returns loss and metrics."""
    args.model.eval()
    epoch_loss = 0.0
    epoch_metrics = dict.fromkeys(args.metrics_dict.keys(), 0.0)

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = (
                batch["image"].to(args.device),
                batch["mask"].to(args.device),
            )
            if inputs.shape[-1] == NO_CHANNEL_DIM:
                inputs = inputs.permute(0, 3, 1, 2)
            if len(targets.shape) == NO_CHANNEL_DIM:
                targets = targets.unsqueeze(1)
            outputs = args.model(inputs)
            loss = args.loss_fn(outputs, targets)
            epoch_loss += loss.item()
            for k, metric_fn in args.metrics_dict.items():
                epoch_metrics[k] += metric_fn(outputs, targets).item()

    avg_epoch_loss = epoch_loss / len(val_loader)
    avg_epoch_metrics = {
        k: v / len(val_loader) for k, v in epoch_metrics.items()
    }
    return avg_epoch_loss, avg_epoch_metrics
