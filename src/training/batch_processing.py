"""Batch processing helpers for training and validation steps."""

from typing import Any

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer


def train_step(  # noqa: PLR0913
    *,
    model: Module,
    batch: tuple | dict[str, torch.Tensor],
    loss_fn: Any,
    optimizer: Optimizer,
    device: torch.device,
    metrics_dict: dict[str, Any] | None = None,
    cfg: DictConfig | None = None,
) -> dict[str, torch.Tensor | float]:
    """Performs a single training step on a batch and returns metrics.
    Returns 'loss' as a tensor for backward compatibility.
    Does NOT call backward or optimizer.step (handled externally)."""
    model.train()

    num_dims_mask_pre_unsqueeze = 3  # Default value
    if (
        cfg
        and hasattr(cfg, "data")
        and hasattr(cfg.data, "num_dims_mask_pre_unsqueeze")
    ):
        num_dims_mask_pre_unsqueeze = cfg.data.num_dims_mask_pre_unsqueeze

    if isinstance(batch, tuple):
        images, masks = batch
    else:
        images = batch["image"]
        masks = batch["mask"]
    images, masks = images.to(device), masks.to(device)
    if len(masks.shape) == num_dims_mask_pre_unsqueeze:
        masks = masks.unsqueeze(1)
    outputs = model(images)
    loss = loss_fn(outputs, masks)
    metrics = {"loss": loss}  # Keep as tensor
    if metrics_dict:
        with torch.no_grad():
            for name, metric_fn in metrics_dict.items():
                metrics[name] = metric_fn(outputs, masks)
    return metrics


def val_step(  # noqa: PLR0913
    *,
    model: Module,
    batch: tuple | dict[str, torch.Tensor],
    loss_fn: Any,
    device: torch.device,
    metrics_dict: dict[str, Any] | None = None,
    cfg: DictConfig | None = None,
) -> dict[str, torch.Tensor | float]:
    """Performs a single validation step on a batch and returns metrics.
    Returns 'loss' as a tensor for compatibility with training helpers."""
    model.eval()

    num_dims_mask_pre_unsqueeze = 3  # Default value
    if (
        cfg
        and hasattr(cfg, "data")
        and hasattr(cfg.data, "num_dims_mask_pre_unsqueeze")
    ):
        num_dims_mask_pre_unsqueeze = cfg.data.num_dims_mask_pre_unsqueeze

    with torch.no_grad():
        if isinstance(batch, tuple):
            images, masks = batch
        else:
            images = batch["image"]
            masks = batch["mask"]
        images, masks = images.to(device), masks.to(device)
        if len(masks.shape) == num_dims_mask_pre_unsqueeze:
            masks = masks.unsqueeze(1)
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        metrics = {"loss": loss}  # Keep as tensor
        if metrics_dict:
            for name, metric_fn in metrics_dict.items():
                metrics[name] = metric_fn(outputs, masks)
    return metrics
