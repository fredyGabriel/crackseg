"""Factory functions for creating training components."""

from typing import Union, Optional

from torch.optim import Optimizer, Adam, SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import _LRScheduler, StepLR, ReduceLROnPlateau
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


def create_optimizer(
    model_params,
    cfg: Union[DictConfig, str, dict]
) -> Optimizer:
    """Creates an optimizer based on the provided configuration.

    Args:
        model_params: Parameters of the model to optimize.
        cfg: Configuration for the optimizer. Can be:
             - DictConfig with _target_ key
             - String with optimizer name (adam, sgd, etc.)
             - Dictionary with type key and other params

    Returns:
        Optimizer: The instantiated optimizer.
    """
    if isinstance(cfg, str):
        # Handle string case - map to common optimizers with default params
        optimizer_name = cfg.lower()

        if optimizer_name == 'adam':
            return Adam(model_params, lr=0.001)
        elif optimizer_name == 'sgd':
            return SGD(model_params, lr=0.01)
        elif optimizer_name == 'adamw':
            return AdamW(model_params, lr=0.001)
        elif optimizer_name == 'rmsprop':
            return RMSprop(model_params, lr=0.001)
        else:
            raise ValueError(
                f"Unsupported optimizer name: {optimizer_name}. "
                "Supported names: adam, sgd, adamw, rmsprop"
            )

    elif isinstance(cfg, dict) and not hasattr(cfg, 'get'):
        # Handle plain dictionary case
        if 'type' in cfg:
            optimizer_type = cfg.pop('type').lower()

            if optimizer_type == 'adam':
                return Adam(model_params, **cfg)
            elif optimizer_type == 'sgd':
                return SGD(model_params, **cfg)
            elif optimizer_type == 'adamw':
                return AdamW(model_params, **cfg)
            elif optimizer_type == 'rmsprop':
                return RMSprop(model_params, **cfg)
            else:
                raise ValueError(f"Unsupported optimizer type: \
{optimizer_type}")
        else:
            raise ValueError("Dictionary optimizer config must contain 'type' \
key")

    # Handle DictConfig case with _target_ attribute
    elif hasattr(cfg, 'get') and cfg.get('_target_'):
        # Use Hydra's instantiate to create the optimizer
        try:
            return instantiate(cfg, params=model_params)
        except:  # noqa: E722
            # If instantiation fails and we're in a test (mock is set up),
            # the exception will be caught by the test framework
            raise

    else:
        raise ValueError(
            f"Invalid optimizer configuration: {cfg}. "
            "Must be string, DictConfig with _target_, or dict with type"
        )


def create_lr_scheduler(
    optimizer: Optimizer,
    cfg: Union[DictConfig, str, dict, None]
) -> Optional[_LRScheduler]:
    """Creates a learning rate scheduler based on the provided configuration.

    Args:
        optimizer: The optimizer instance.
        cfg: Configuration for the scheduler. Can be:
             - DictConfig with _target_ key
             - String with scheduler name (step_lr, reduce_lr_on_plateau, etc.)
             - Dictionary with type key and other params
             - None (no scheduler)

    Returns:
        Optional[_LRScheduler]: The instantiated scheduler, or None if no
                                scheduler is configured.
    """
    if cfg is None:
        return None  # No scheduler configured

    if isinstance(cfg, str):
        # Handle string case - map to common schedulers with default params
        scheduler_name = cfg.lower()

        if scheduler_name == 'step_lr':
            return StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_name == 'reduce_lr_on_plateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                     patience=10)
        elif scheduler_name == 'none' or scheduler_name == '':
            return None
        else:
            raise ValueError(
                f"Unsupported scheduler name: {scheduler_name}. "
                "Supported names: step_lr, reduce_lr_on_plateau, none"
            )

    elif isinstance(cfg, dict) and not hasattr(cfg, 'get'):
        # Handle plain dictionary case
        if 'type' in cfg:
            scheduler_type = cfg.pop('type').lower()

            if scheduler_type == 'step_lr':
                return StepLR(optimizer, **cfg)
            elif scheduler_type == 'reduce_lr_on_plateau':
                return ReduceLROnPlateau(optimizer, **cfg)
            elif scheduler_type == 'none':
                return None
            else:
                raise ValueError(f"Unsupported scheduler type: \
{scheduler_type}")
        else:
            raise ValueError("Dictionary scheduler config must contain 'type' \
key")

    # Handle DictConfig case with _target_ attribute
    elif hasattr(cfg, 'get') and cfg.get('_target_'):
        # Use Hydra's instantiate to create the scheduler
        try:
            return instantiate(cfg, optimizer=optimizer)
        except:  # noqa: E722
            # If instantiation fails and we're in a test (mock is set up),
            # the exception will be caught by the test framework
            raise

    else:
        raise ValueError(
            f"Invalid scheduler configuration: {cfg}. "
            "Must be string, DictConfig with _target_, or dict with type, or"
            "None"
        )
