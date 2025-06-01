"""Factory functions for creating objects from configuration."""

from collections.abc import Callable as TypingCallable
from importlib import import_module
from typing import Any, TypeVar, cast

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn

# Import specific losses for type checking if needed, or rely on name
from src.training.losses import BCEDiceLoss, CombinedLoss
from src.utils.exceptions import ConfigError
from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def import_class(class_path: str) -> type:
    """Import a class from a string path.

    Args:
        class_path: Full path to the class (e.g., 'torch.nn.CrossEntropyLoss')

    Returns:
        The imported class
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = import_module(module_path)
        return cast(type, getattr(module, class_name))
    except Exception as e:
        raise ConfigError(
            f"Failed to import class '{class_path}'", details=str(e)
        ) from e


def get_optimizer(
    model_params: list[nn.Parameter],
    optimizer_cfg: DictConfig | str | dict[str, Any],
) -> torch.optim.Optimizer:
    """Create an optimizer from config.

    Args:
        model_params: Model parameters to optimize
        optimizer_cfg: Optimizer configuration or string name

    Returns:
        Configured optimizer instance
    """
    try:
        if isinstance(optimizer_cfg, str):
            # Handle string case - map to common optimizers
            optimizer_name = optimizer_cfg.lower()

            if optimizer_name == "adam":
                return torch.optim.Adam(model_params, lr=0.001)
            elif optimizer_name == "sgd":
                return torch.optim.SGD(model_params, lr=0.01)
            elif optimizer_name == "adamw":
                return torch.optim.AdamW(model_params, lr=0.001)
            elif optimizer_name == "rmsprop":
                return torch.optim.RMSprop(model_params, lr=0.001)
            else:
                raise ConfigError(
                    f"Unsupported optimizer name: {optimizer_name}",
                    details="Supported names: adam, sgd, adamw, rmsprop",
                )

        elif (
            hasattr(optimizer_cfg, "__getitem__")
            and "_target_" in optimizer_cfg
        ):
            optimizer_class = import_class(str(optimizer_cfg["_target_"]))
            optimizer_params = {
                str(k): v for k, v in optimizer_cfg.items() if k != "_target_"
            }
            return cast(
                torch.optim.Optimizer,
                optimizer_class(model_params, **optimizer_params),
            )

        else:
            raise ConfigError(
                f"Unsupported optimizer config type: {type(optimizer_cfg)}",
                details="Must be string, DictConfig with _target_, or dict \
with type",
            )

    except Exception as e:
        raise ConfigError("Failed to create optimizer", details=str(e)) from e


def get_loss_fn(loss_cfg: DictConfig) -> TypingCallable[..., object]:
    """Create a loss function from config."""
    try:
        target_path = loss_cfg.get("_target_")
        if not target_path:
            raise ConfigError("Loss configuration missing '_target_' key.")

        loss_class = import_class(target_path)

        # --- Special Handling ---
        if loss_class is CombinedLoss:
            inner_loss_configs = loss_cfg.losses
            weights = loss_cfg.get("weights")
            if (
                not isinstance(inner_loss_configs, list | ListConfig)
                or not inner_loss_configs
            ):
                raise ConfigError(
                    "CombinedLoss requires a non-empty list or ListConfig "
                    "under 'losses'."
                )

            # Convertir cada item a dict[str, Any] para evitar Unknown
            losses_config: list[dict[str, Any]] = []
            for item in cast(list[Any], inner_loss_configs):
                # Forzar tipado explícito para el linter
                if isinstance(item, dict):
                    losses_config.append(cast(dict[str, Any], item))
                else:
                    item_obj = cast(object, item)  # Para el linter
                    if hasattr(item_obj, "__dict__"):
                        # Para dataclasses/configs
                        losses_config.append(dict(item_obj.__dict__))
                    else:
                        raise ConfigError(
                            "Each item in CombinedLoss losses must be a dict "
                            "or convertible to dict."
                        )
            combined_params = {}
            if weights is not None:
                combined_params["weights"] = weights
            return CombinedLoss(losses_config=losses_config, **combined_params)  # type: ignore

        elif loss_class is BCEDiceLoss:
            params = {
                str(k): v for k, v in loss_cfg.items() if k != "_target_"
            }
            return cast(TypingCallable[..., object], BCEDiceLoss(**params))

        # --- Default Handling ---
        else:
            params = {
                str(k): v for k, v in loss_cfg.items() if k != "_target_"
            }
            return cast(TypingCallable[..., object], loss_class(**params))

    except ConfigError as e:
        raise e  # Re-raise specific config errors
    except Exception as e:
        raise ConfigError(
            f"Failed to create loss function from config: {loss_cfg}",
            details=str(e),
        ) from e


def get_metrics_from_cfg(
    metrics_cfg: DictConfig,
) -> dict[str, TypingCallable[..., object]]:
    """Create metric functions from config.

    Args:
        metrics_cfg: Metrics configuration dictionary

    Returns:
        Dictionary mapping metric names to their functions
    """
    metrics = {}
    for name, cfg in metrics_cfg.items():
        try:
            metric_class = import_class(cfg._target_)
            metric_params = {k: v for k, v in cfg.items() if k != "_target_"}
            metrics[str(name)] = metric_class(**metric_params)
        except Exception as e:
            raise ConfigError(
                f"Failed to create metric {name!r}", details=str(e)
            ) from e
    return cast(dict[str, TypingCallable[..., object]], metrics)
