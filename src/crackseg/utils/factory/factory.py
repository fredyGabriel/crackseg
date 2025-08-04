"""Factory functions for creating objects from configuration."""

from collections.abc import Callable as TypingCallable
from importlib import import_module
from typing import Any, TypeVar, cast

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn

# Import specific losses using lazy loading to avoid circular dependencies
# from crackseg.training.losses import BCEDiceLoss, CombinedLoss, FocalDiceLoss
from crackseg.utils.core.exceptions import ConfigError
from crackseg.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _get_loss_class(class_name: str):
    """Lazy import of loss classes to avoid circular dependencies.

    Args:
        class_name: Name of the loss class to import

    Returns:
        The loss class

    Raises:
        AttributeError: If the class doesn't exist in the losses module
    """
    import crackseg.training.losses as losses_module

    return getattr(losses_module, class_name)


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

        elif hasattr(optimizer_cfg, "__getitem__") and "type" in optimizer_cfg:
            # Handle 'type' format as alias for '_target_'
            optimizer_type = str(optimizer_cfg["type"]).lower()

            # Map optimizer types to classes
            optimizer_mapping = {
                "adam": torch.optim.Adam,
                "sgd": torch.optim.SGD,
                "adamw": torch.optim.AdamW,
                "rmsprop": torch.optim.RMSprop,
            }

            if optimizer_type not in optimizer_mapping:
                raise ConfigError(
                    f"Unsupported optimizer type: {optimizer_type}",
                    details="Supported types: adam, sgd, adamw, rmsprop",
                )

            optimizer_class = optimizer_mapping[optimizer_type]
            optimizer_params = {
                str(k): v for k, v in optimizer_cfg.items() if k != "type"
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
        # Handle 'type' format as alias for '_target_'
        if hasattr(loss_cfg, "__getitem__") and "type" in loss_cfg:
            loss_type = str(loss_cfg["type"]).lower()

            # Map loss types to classes
            loss_mapping = {
                "bce_dice": "crackseg.training.losses.BCEDiceLoss",
                "focal_dice": "crackseg.training.losses.focal_dice_loss.FocalDiceLoss",
                "dice": "crackseg.training.losses.DiceLoss",
                "bce": "torch.nn.BCEWithLogitsLoss",
                "cross_entropy": "torch.nn.CrossEntropyLoss",
            }

            if loss_type not in loss_mapping:
                raise ConfigError(
                    f"Unsupported loss type: {loss_type}",
                    details="Supported types: bce_dice, focal_dice, dice, bce, cross_entropy",
                )

            # Create a new config with _target_ instead of type
            loss_cfg_with_target = DictConfig(loss_cfg)
            loss_cfg_with_target["_target_"] = loss_mapping[loss_type]
            del loss_cfg_with_target["type"]

            # Continue with the standard processing
            loss_cfg = loss_cfg_with_target

        target_path = loss_cfg.get("_target_")
        if not target_path:
            raise ConfigError("Loss configuration missing '_target_' key.")

        loss_class = import_class(target_path)

        # --- Special Handling ---
        if loss_class is _get_loss_class("CombinedLoss"):
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
            CombinedLoss = _get_loss_class("CombinedLoss")
            return CombinedLoss(losses_config=losses_config, **combined_params)

        elif loss_class is _get_loss_class("BCEDiceLoss"):
            # BCEDiceLoss expects a config parameter with BCEDiceLossConfig
            if "config" in loss_cfg:
                # If config is provided, use it directly
                config_params = {}
                # Access values directly without resolving interpolations
                config_obj = loss_cfg.config
                for key in config_obj.keys():
                    if key != "_target_":
                        try:
                            value = config_obj[key]
                            # Handle interpolation values by providing defaults
                            if (
                                isinstance(value, str)
                                and value.startswith("${")
                                and value.endswith("}")
                            ):
                                # Extract the key and provide default value
                                key_name = value[2:-1]  # Remove ${ and }
                                if "thresholds.loss_weight" in key_name:
                                    config_params[str(key)] = (
                                        0.5  # Default value
                                    )
                                elif "thresholds" in key_name:
                                    config_params[str(key)] = (
                                        0.5  # Default value
                                    )
                                else:
                                    config_params[str(key)] = (
                                        value  # Keep as string if unknown
                                    )
                            else:
                                config_params[str(key)] = value
                        except Exception:
                            # If we can't access the value due to interpolation, use default
                            config_params[str(key)] = 0.5

                BCEDiceLossConfig = _get_loss_class("BCEDiceLossConfig")
                BCEDiceLoss = _get_loss_class("BCEDiceLoss")
                config = BCEDiceLossConfig(**config_params)
                return cast(
                    TypingCallable[..., object], BCEDiceLoss(config=config)
                )
            else:
                # If no config provided, use default parameters
                BCEDiceLoss = _get_loss_class("BCEDiceLoss")
                return cast(TypingCallable[..., object], BCEDiceLoss())

        elif loss_class is _get_loss_class("FocalDiceLoss"):
            # FocalDiceLoss expects a config parameter with FocalDiceLossConfig
            if "config" in loss_cfg:
                # If config is provided, use it directly
                config_params = {
                    str(k): v
                    for k, v in loss_cfg.config.items()
                    if k != "_target_"
                }
                FocalDiceLossConfig = _get_loss_class("FocalDiceLossConfig")
                FocalDiceLoss = _get_loss_class("FocalDiceLoss")
                config = FocalDiceLossConfig(**config_params)
                return cast(
                    TypingCallable[..., object], FocalDiceLoss(config=config)
                )
            else:
                # If no config provided, use default parameters
                FocalDiceLoss = _get_loss_class("FocalDiceLoss")
                return cast(TypingCallable[..., object], FocalDiceLoss())

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
    metrics_cfg: DictConfig | list[str],
) -> dict[str, TypingCallable[..., object]]:
    """Create metric functions from config.

    Args:
        metrics_cfg: Metrics configuration - either a dictionary with full
                     configs or a list of metric names

    Returns:
        Dictionary mapping metric names to their functions
    """
    metrics = {}

    # Handle list of metric names (simplified format)
    if isinstance(metrics_cfg, list | ListConfig):
        # Import default metric classes
        from crackseg.training.metrics import (
            F1Score,
            IoUScore,
            PrecisionScore,
            RecallScore,
        )

        # Mapping of metric names to classes
        metric_mapping = {
            "accuracy": IoUScore,  # Use IoU as proxy for accuracy
            "dice": F1Score,  # F1 is equivalent to Dice for binary segment.
            "f1": F1Score,
            "iou": IoUScore,
            "precision": PrecisionScore,
            "recall": RecallScore,
        }

        for metric_name in metrics_cfg:
            metric_name_str = str(metric_name).lower()
            if metric_name_str in metric_mapping:
                metrics[metric_name_str] = metric_mapping[metric_name_str]()
            else:
                logger.warning(f"Unknown metric name: {metric_name}")

    # Handle dictionary format (full configuration)
    else:
        for name, cfg in metrics_cfg.items():
            try:
                metric_class = import_class(cfg._target_)
                metric_params = {
                    k: v for k, v in cfg.items() if k != "_target_"
                }
                metrics[str(name)] = metric_class(**metric_params)
            except Exception as e:
                raise ConfigError(
                    f"Failed to create metric {name!r}", details=str(e)
                ) from e

    return cast(dict[str, TypingCallable[..., object]], metrics)
