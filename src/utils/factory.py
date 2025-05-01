import hydra
import torch  # Needed for BCELoss weight tensor conversion
from omegaconf import DictConfig, ListConfig
from typing import Dict

# Imports are handled by hydra.utils.instantiate based on _target_
from src.training.losses import SegmentationLoss, CombinedLoss
from src.training.metrics import Metric


def get_loss_from_cfg(cfg: DictConfig) -> SegmentationLoss:
    """
    Instantiates a loss function based on the provided Hydra configuration.

    Handles standard losses and the CombinedLoss case which requires
    recursive instantiation of nested loss configurations.

    Args:
        cfg: The Hydra configuration object for the loss function.
             Expected to have a '_target_' field pointing to the class
             and other parameters matching the class __init__.

    Returns:
        An instantiated loss function object.

    Raises:
        ValueError: If the configuration is invalid or the target class
                    cannot be instantiated.
        TypeError: If the instantiated object is not a SegmentationLoss.
    """
    target_class_path = cfg.get("_target_")
    if not target_class_path:
        raise ValueError(f"Configuration missing '_target_' field: {cfg}")

    try:
        if target_class_path == "src.training.losses.CombinedLoss":
            nested_losses = []
            nested_weights = []
            if not hasattr(cfg, 'losses') or \
               not isinstance(cfg.losses, (list, ListConfig)):
                raise ValueError("CombinedLoss config requires a 'losses' \
list.")

            for item_cfg_dict in cfg.losses:
                if not isinstance(item_cfg_dict, DictConfig):
                    item_cfg_dict = DictConfig(item_cfg_dict)

                if 'config' not in item_cfg_dict or 'weight' not in \
                        item_cfg_dict:
                    # Split the long error message line
                    err_msg = (
                        "Each item in CombinedLoss.losses must have "
                        "'config' and 'weight'."
                    )
                    raise ValueError(err_msg)

                nested_loss_cfg = item_cfg_dict.config
                weight = item_cfg_dict.weight

                nested_loss_instance = get_loss_from_cfg(nested_loss_cfg)
                nested_losses.append(nested_loss_instance)
                nested_weights.append(weight)

            # Manually instantiate CombinedLoss
            instance = CombinedLoss(losses=nested_losses,
                                    weights=nested_weights)

        elif target_class_path == "src.training.losses.BCELoss":
            # Special handling for BCE weight tensor conversion
            # Create a copy to avoid modifying the original config
            bce_cfg = cfg.copy()
            # Remove weight if present, handle conversion separately
            weight_list = bce_cfg.pop("weight", None)
            # Remove reduction if present, not needed for BCELoss init
            bce_cfg.pop("reduction", None)

            weight_tensor = torch.tensor(weight_list) \
                if weight_list is not None else None

            # Pass only relevant args to instantiate
            instance = hydra.utils.instantiate(
                bce_cfg, weight=weight_tensor, _convert_="partial"
            )

        else:
            # For other standard losses, instantiate directly
            instance = hydra.utils.instantiate(cfg, _convert_="partial")

        # Final type check
        if not isinstance(instance, SegmentationLoss):
            # Split the long error message line
            err_msg = (
                f"Instantiated object is not a SegmentationLoss: "
                f"{type(instance)} from config {cfg}"
            )
            raise TypeError(err_msg)
        return instance

    except Exception as e:
        # Catch instantiation errors or ValueErrors from recursive calls
        # Split the long error message line
        err_msg = f"Failed to instantiate loss from config {cfg}: {e}"
        raise ValueError(err_msg) from e


def get_metrics_from_cfg(cfg_dict: DictConfig) -> Dict[str, Metric]:
    """
    Instantiates a dictionary of metric objects based on a Hydra DictConfig.

    Args:
        cfg_dict: A Hydra DictConfig where keys are metric names and values
                  are DictConfigs defining each metric with a '_target_' field.

    Returns:
        A dictionary where keys are metric names and values are instantiated
        metric objects.

    Raises:
        ValueError: If any configuration is invalid or a target class
                    cannot be instantiated.
        TypeError: If the input is not a DictConfig, or if an
                 instantiated object is not a Metric.
    """
    metrics = {}
    if not isinstance(cfg_dict, DictConfig):
        raise TypeError(
            "Input must be a DictConfig of metric configurations."
        )

    for name, metric_cfg in cfg_dict.items():
        try:
            # Ensure it's a DictConfig and has _target_
            if not isinstance(metric_cfg, DictConfig) or \
               '_target_' not in metric_cfg:
                # Split the long error message line
                err_msg = (
                    f"Metric config for '{name}' missing '_target_' or "
                    f"is not a DictConfig: {metric_cfg}"
                )
                raise ValueError(err_msg)

            # Instantiate the metric using Hydra
            metric_instance = hydra.utils.instantiate(
                metric_cfg, _convert_="partial"
            )

            if not isinstance(metric_instance, Metric):
                # Split the long error message line
                err_msg = (
                    f"Instantiated object for '{name}' is not a Metric: "
                    f"{type(metric_instance)} from config {metric_cfg}"
                )
                raise TypeError(err_msg)

            metrics[name] = metric_instance
        except Exception as e:
            # Split the long error message line
            err_msg = (
                f"Failed to instantiate metric '{name}' from config "
                f"{metric_cfg}: {e}"
            )
            raise ValueError(err_msg) from e

    return metrics
