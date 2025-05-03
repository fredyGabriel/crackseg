"""Batch processing helpers for training and validation steps."""
from typing import Dict, Any, Union
import torch
from torch.nn import Module
from torch.optim import Optimizer


def train_step(
    *,
    model: Module,
    batch: Union[tuple, Dict[str, torch.Tensor]],
    loss_fn: Any,
    optimizer: Optimizer,
    device: torch.device,
    metrics_dict: Dict[str, Any] = None
) -> Dict[str, float]:
    """Performs a single training step on a batch and returns metrics.
    Returns 'loss' as a tensor for backward compatibility.
    Does NOT call backward or optimizer.step (handled externally)."""
    model.train()
    if isinstance(batch, tuple):
        images, masks = batch
    else:
        images = batch['image']
        masks = batch['mask']
    images, masks = images.to(device), masks.to(device)
    if len(masks.shape) == 3:
        masks = masks.unsqueeze(1)
    outputs = model(images)
    loss = loss_fn(outputs, masks)
    metrics = {"loss": loss}  # Keep as tensor
    if metrics_dict:
        with torch.no_grad():
            for name, metric_fn in metrics_dict.items():
                metrics[name] = metric_fn(outputs, masks)
    return metrics


def val_step(
    *,
    model: Module,
    batch: Union[tuple, Dict[str, torch.Tensor]],
    loss_fn: Any,
    device: torch.device,
    metrics_dict: Dict[str, Any] = None
) -> Dict[str, float]:
    """Performs a single validation step on a batch and returns metrics.
    Returns 'loss' as a tensor for compatibility with training helpers."""
    model.eval()
    with torch.no_grad():
        if isinstance(batch, tuple):
            images, masks = batch
        else:
            images = batch['image']
            masks = batch['mask']
        images, masks = images.to(device), masks.to(device)
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        metrics = {"loss": loss}  # Keep as tensor
        if metrics_dict:
            for name, metric_fn in metrics_dict.items():
                metrics[name] = metric_fn(outputs, masks)
    return metrics
