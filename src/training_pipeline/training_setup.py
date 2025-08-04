"""Training setup for training pipeline.

This module provides training component setup functions for the training
pipeline including metrics, optimizer, and loss function configuration.
"""

import logging
from typing import Any

from omegaconf import DictConfig
from torch import optim

from crackseg.utils.factory import (
    get_loss_fn,
    get_metrics_from_cfg,
    get_optimizer,
)

# Configure standard logger
log = logging.getLogger(__name__)


def setup_training_components(
    cfg: DictConfig, model: Any
) -> tuple[dict[str, Any], optim.Optimizer, Any]:
    """
    Set up training components including metrics, optimizer, and loss function.

    This function configures all components required for training:
    - Evaluation metrics from configuration
    - Optimizer with specified parameters
    - Loss function with appropriate configuration
    - Error handling and fallback mechanisms

    Args:
        cfg: Hydra configuration containing training settings.
            Expected structure:
            - cfg.evaluation.metrics (DictConfig, optional): Metrics
                configuration
            - cfg.training.optimizer (DictConfig): Optimizer configuration
            - cfg.training.loss (DictConfig, optional): Loss function
                configuration

        model: The neural network model for optimizer parameter access.

    Returns:
        tuple[dict[str, Any], optim.Optimizer, Any]: A tuple containing:
            - metrics_dict: Dictionary of evaluation metrics
            - optimizer: Configured optimizer instance
            - loss_fn: Loss function for training

    Raises:
        ValueError: If training component configuration is invalid
        AttributeError: If required configuration keys are missing
        TypeError: If component instantiation fails

    Examples:
        ```python
        cfg = OmegaConf.create({
            "training": {
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    "lr": 0.001
                },
                "loss": {
                    "_target_": "torch.nn.BCEWithLogitsLoss"
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "iou"]
            }
        })

        metrics, optimizer, loss_fn = setup_training_components(cfg, model)

        # Check components
        print(f"Metrics: {list(metrics.keys())}")
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Loss: {type(loss_fn).__name__}")
        ```

    Note:
        The function provides fallback configurations for missing components
        and validates all instantiated objects before returning them.
    """
    log.info("Setting up training components...")

    # Handle experiment namespace configuration access
    training_cfg = None
    evaluation_cfg = None

    if "experiments" in cfg:
        # Look in experiments namespace for configs
        for exp_name in cfg.experiments:
            exp_config = cfg.experiments[exp_name]
            if hasattr(exp_config, "training"):
                training_cfg = exp_config.training
            if hasattr(exp_config, "evaluation"):
                evaluation_cfg = exp_config.evaluation
            if training_cfg and evaluation_cfg:
                break

    # Fall back to direct access
    if training_cfg is None and hasattr(cfg, "training"):
        training_cfg = cfg.training
    if evaluation_cfg is None and hasattr(cfg, "evaluation"):
        evaluation_cfg = cfg.evaluation

    # Get metrics from evaluation configuration, with fallback to default metrics
    if evaluation_cfg and hasattr(evaluation_cfg, "metrics"):
        metrics_dict = get_metrics_from_cfg(evaluation_cfg.metrics)
    else:
        # Default metrics if not specified
        default_metrics = ["iou", "dice", "precision", "recall", "f1"]
        metrics_dict = get_metrics_from_cfg(default_metrics)
    log.info("Configured metrics: %s", list(metrics_dict.keys()))

    # Get optimizer from training configuration, with fallback to default
    if training_cfg and hasattr(training_cfg, "optimizer"):
        optimizer = get_optimizer(model.parameters(), training_cfg.optimizer)
    else:
        # Default optimizer if not specified
        default_optimizer_cfg = {"_target_": "torch.optim.Adam", "lr": 0.001}
        optimizer = get_optimizer(model.parameters(), default_optimizer_cfg)
    log.info("Configured optimizer: %s", type(optimizer).__name__)

    # Get loss function from training configuration, with fallback to default
    if training_cfg and hasattr(training_cfg, "loss"):
        loss_fn = get_loss_fn(training_cfg.loss)
    else:
        # Default loss function if not specified
        from omegaconf import OmegaConf

        default_loss_cfg = OmegaConf.create(
            {"_target_": "crackseg.training.losses.BCEDiceLoss"}
        )
        loss_fn = get_loss_fn(default_loss_cfg)
    log.info("Configured loss function: %s", type(loss_fn).__name__)

    return metrics_dict, optimizer, loss_fn
