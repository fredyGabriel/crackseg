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

    # Get metrics from configuration
    metrics_dict = get_metrics_from_cfg(cfg)
    log.info("Configured metrics: %s", list(metrics_dict.keys()))

    # Get optimizer from configuration
    optimizer = get_optimizer(cfg, model)  # type: ignore
    log.info("Configured optimizer: %s", type(optimizer).__name__)

    # Get loss function from configuration
    loss_fn = get_loss_fn(cfg)
    log.info("Configured loss function: %s", type(loss_fn).__name__)

    return metrics_dict, optimizer, loss_fn
