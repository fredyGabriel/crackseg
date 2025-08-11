"""Training setup for training (consolidated)."""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import optim

from crackseg.utils.factory import (
    get_loss_fn,
    get_metrics_from_cfg,
    get_optimizer,
)

log = logging.getLogger(__name__)


def setup_training_components(
    cfg: DictConfig, model: Any
) -> tuple[dict[str, Any], optim.Optimizer, Any]:
    log.info("Setting up training components...")
    training_cfg = None
    evaluation_cfg = None
    if "experiments" in cfg:
        for exp_name in cfg.experiments:
            exp_config = cfg.experiments[exp_name]
            if hasattr(exp_config, "training"):
                training_cfg = exp_config.training
            if hasattr(exp_config, "evaluation"):
                evaluation_cfg = exp_config.evaluation
            if training_cfg and evaluation_cfg:
                break
    if training_cfg is None and hasattr(cfg, "training"):
        training_cfg = cfg.training
    if evaluation_cfg is None and hasattr(cfg, "evaluation"):
        evaluation_cfg = cfg.evaluation

    if evaluation_cfg and hasattr(evaluation_cfg, "metrics"):
        metrics_dict = get_metrics_from_cfg(evaluation_cfg.metrics)
    else:
        metrics_dict = get_metrics_from_cfg(
            ["iou", "dice", "precision", "recall", "f1"]
        )
    log.info("Configured metrics: %s", list(metrics_dict.keys()))

    if training_cfg and hasattr(training_cfg, "optimizer"):
        optimizer = get_optimizer(model.parameters(), training_cfg.optimizer)
    else:
        optimizer = get_optimizer(
            model.parameters(), {"_target_": "torch.optim.Adam", "lr": 0.001}
        )
    log.info("Configured optimizer: %s", type(optimizer).__name__)

    if training_cfg and hasattr(training_cfg, "loss"):
        loss_fn = get_loss_fn(training_cfg.loss)
    else:
        default_loss_cfg = OmegaConf.create(
            {"_target_": "crackseg.training.losses.BCEDiceLoss"}
        )
        loss_fn = get_loss_fn(default_loss_cfg)
    log.info("Configured loss function: %s", type(loss_fn).__name__)

    return metrics_dict, optimizer, loss_fn
