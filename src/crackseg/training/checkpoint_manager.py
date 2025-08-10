"""Checkpoint management for training (consolidated)."""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig
from torch import optim

from crackseg.utils import load_checkpoint

log = logging.getLogger(__name__)


def handle_checkpointing_and_resume(
    cfg: DictConfig,
    model: Any,
    optimizer: optim.Optimizer,
    device: Any,
    experiment_logger: Any,
) -> tuple[int, float | None]:
    log.info("Handling checkpointing and resume...")
    start_epoch = 0
    best_metric_value: float | None = None

    training_cfg = None
    if "experiments" in cfg:
        for exp_name in cfg.experiments:
            exp_config = cfg.experiments[exp_name]
            if hasattr(exp_config, "training"):
                training_cfg = exp_config.training
                break
    if training_cfg is None and hasattr(cfg, "training"):
        training_cfg = cfg.training

    checkpoint_path = (
        getattr(training_cfg, "checkpoint_dir", None) if training_cfg else None
    )
    if checkpoint_path:
        log.info("Resuming from checkpoint: %s", checkpoint_path)
        try:
            checkpoint = load_checkpoint(checkpoint_path, device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                log.info("Model state loaded from checkpoint")
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                log.info("Optimizer state loaded from checkpoint")
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                log.info("Resuming from epoch: %s", start_epoch)
            if "best_metric_value" in checkpoint:
                best_metric_value = checkpoint["best_metric_value"]
                log.info(
                    "Best metric value from checkpoint: %s", best_metric_value
                )
            log.info("Checkpoint loaded successfully")
        except Exception as e:
            log.error("Error loading checkpoint: %s", str(e))
            log.warning("Starting training from scratch")
            start_epoch = 0
            best_metric_value = None
    else:
        log.info("No checkpoint specified, starting from scratch")

    return start_epoch, best_metric_value
