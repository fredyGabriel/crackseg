"""Handles the main training and evaluation loops."""

import torch
from torch.amp import GradScaler
from omegaconf import DictConfig
from typing import Dict, Any
import logging

# Placeholder imports - replace with actual logger and metric functions later
# from src.utils.loggers import BaseLogger
# from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    # logger_instance: BaseLogger | None = None, # Add later
    cfg: DictConfig | None = None,
    epoch: int = 0,
) -> float:
    """Runs a single training epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for the training set.
        loss_fn: The loss function.
        optimizer: The optimizer.
        device: The device to run training on.
        scaler: Optional GradScaler for mixed-precision training.
        # logger_instance: Optional logger for recording metrics.
        cfg: Optional configuration object (for grad_accum, log_interval etc.).
        epoch: Current epoch number (for logging).

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    # Extract relevant config params if available
    grad_accum_steps = 1
    log_interval = 50
    if cfg:
        grad_accum_steps = cfg.training.get("gradient_accumulation_steps", 1)
        log_interval = cfg.training.get("log_interval", 50)

    optimizer.zero_grad()  # Reset gradients at the start

    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            # Scale loss for accumulation
            loss = loss / grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step only after accumulating gradients
        is_update_step = (batch_idx + 1) % grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == num_batches
        if is_update_step or is_last_batch:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()  # Reset gradients after step

        # Unscale loss for logging
        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss

        is_log_step = (batch_idx + 1) % log_interval == 0
        if is_log_step or is_last_batch:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{num_batches} | "
                f"Loss: {batch_loss:.4f}"
            )
            # --- Log batch loss ---
            # if logger_instance:
            #     global_step = epoch * num_batches + batch_idx
            #     logger_instance.log_scalar(
            #         "train/batch_loss", batch_loss, global_step
            #     )

        # --- Calculate and log batch metrics (optional) ---
        # log_batch_metrics = cfg.training.get("log_batch_metrics", False) if cfg else False
        # if logger_instance and log_batch_metrics:
        #     with torch.no_grad():
        #         # metrics = calculate_metrics(outputs, masks, threshold=...)
        #         for name, value in metrics.items():
        #              logger_instance.log_scalar(
        #                  f"train/batch_{name}", value.item(), global_step
        #              )

    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} | Average Training Loss: {avg_loss:.4f}")
    # --- Log epoch loss ---
    # if logger_instance:
    #    logger_instance.log_scalar("train/epoch_loss", avg_loss, epoch)

    return avg_loss


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    metrics_dict: Dict[str, Any],  # Expects instantiated metrics
    # logger_instance: BaseLogger | None = None, # Add later
    cfg: DictConfig | None = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """Runs evaluation on a dataset.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the validation/test set.
        loss_fn: The loss function.
        device: The device to run evaluation on.
        metrics_dict: Dictionary of instantiated metric objects
            {name: metric_fn}.
        # logger_instance: Optional logger for recording metrics.
        cfg: Optional configuration object.
        epoch: Current epoch number (for logging).


    Returns:
        Dictionary containing average loss and metric values.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    # Initialize metric accumulators (or use a dedicated class)
    metric_values = {name: 0.0 for name in metrics_dict}
    amp_enabled = cfg.training.amp_enabled if cfg and hasattr(
        cfg, 'training') else False

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            total_loss += loss.item()

            # Update metrics
            for name, metric_fn in metrics_dict.items():
                # Assume metric returns scalar tensor
                metric_val = metric_fn(outputs, masks)
                metric_values[name] += metric_val.item()

    avg_loss = total_loss / num_batches
    avg_metrics = {name: val / num_batches for name, val in
                   metric_values.items()}

    log_str = f"Epoch {epoch} | Validation Loss: {avg_loss:.4f}"
    for name, value in avg_metrics.items():
        log_str += f" | {name}: {value:.4f}"
    logger.info(log_str)

    # --- Log epoch metrics ---
    # if logger_instance:
    #     logger_instance.log_scalar("val/epoch_loss", avg_loss, epoch)
    #     for name, value in avg_metrics.items():
    #         logger_instance.log_scalar(f"val/{name}", value, epoch)

    results = {"val_loss": avg_loss}
    results.update({f"val_{k}": v for k, v in avg_metrics.items()})
    return results
