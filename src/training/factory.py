"""Factory functions for creating training components."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.nn import Module

# Removed incorrect imports of create_loss, create_metric
# from src.training.losses import create_loss
# from src.training.metrics import create_metric


def create_loss_fn(cfg: DictConfig) -> Module:
    """Creates a loss function based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration dictionary for the loss function.
                         Expected keys: _target_, etc.

    Returns:
        Module: The instantiated loss function.
    """
    if not cfg or not cfg.get('_target_'):
        raise ValueError("Loss configuration must include '_target_'")

    # Use Hydra's instantiate to create the loss function
    try:
        return instantiate(cfg)
    except:  # noqa: E722
        # If instantiation fails and we're in a test (mock is set up),
        # the exception will be caught by the test framework
        raise


def create_optimizer(model_params, cfg: DictConfig) -> Optimizer:
    """Creates an optimizer based on the provided configuration.

    Args:
        model_params: Parameters of the model to optimize.
        cfg (DictConfig): Configuration dictionary for the optimizer.
                          Expected keys: _target_, lr, etc.

    Returns:
        Optimizer: The instantiated optimizer.
    """
    if not cfg or not cfg.get('_target_'):
        raise ValueError("Optimizer configuration must include '_target_'")

    # Use Hydra's instantiate to create the optimizer
    try:
        return instantiate(cfg, params=model_params)
    except:  # noqa: E722
        # If instantiation fails and we're in a test (mock is set up),
        # the exception will be caught by the test framework
        raise


def create_lr_scheduler(optimizer: Optimizer, cfg: DictConfig
                        ) -> _LRScheduler | None:
    """Creates a learning rate scheduler based on the provided configuration.

    Args:
        optimizer (Optimizer): The optimizer instance.
        cfg (DictConfig): Configuration dictionary for the scheduler.
                          Expected keys: _target_, etc. Can be None or empty.

    Returns:
        _LRScheduler | None: The instantiated scheduler, or None if no
                             scheduler is configured.
    """
    if not cfg or not cfg.get('_target_'):
        return None  # No scheduler configured

    # Use Hydra's instantiate to create the scheduler
    try:
        return instantiate(cfg, optimizer=optimizer)
    except:  # noqa: E722
        # If instantiation fails and we're in a test (mock is set up),
        # the exception will be caught by the test framework
        raise
