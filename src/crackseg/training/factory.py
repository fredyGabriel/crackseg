"""Factory functions for creating training components."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
# pyright: reportExplicitAny=false, reportAny=false, reportAttributeAccessIssue=false
# pyright: reportUnnecessaryComparison=false, reportArgumentType=false
# pyright: reportInvalidCast=false, reportImplicitStringConcatenation=false
# Global suppressions for factory functions with dynamic configuration handling

from typing import Any, cast

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, _LRScheduler

# Removed incorrect imports of create_loss, create_metric
# from crackseg.training.losses import create_loss
# from crackseg.training.metrics import create_metric

# Mapping of optimizer names/types to their respective classes
_OPTIMIZER_MAP = {
    "adam": Adam,
    "sgd": SGD,
    "adamw": AdamW,
    "rmsprop": RMSprop,
}

# Mapping of LR scheduler names/types to their respective classes
_LR_SCHEDULER_MAP = {
    "step_lr": StepLR,
    "reduce_lr_on_plateau": ReduceLROnPlateau,
    # Add other schedulers here if needed
}

# Default parameters for schedulers when specified by string name
_LR_SCHEDULER_DEFAULT_PARAMS = {
    "step_lr": {"step_size": 10, "gamma": 0.1},
    "reduce_lr_on_plateau": {"mode": "min", "factor": 0.1, "patience": 10},
}


def create_loss_fn(cfg: DictConfig) -> Module:
    """Creates a loss function based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration dictionary for the loss function.
                         Expected keys: _target_, etc.

    Returns:
        Module: The instantiated loss function.
    """
    if not cfg or not cfg.get("_target_"):
        raise ValueError("Loss configuration must include '_target_'")

    # Use Hydra's instantiate to create the loss function
    try:
        return cast(Module, instantiate(cfg))
    except:  # noqa: E722
        # If instantiation fails and we're in a test (mock is set up),
        # the exception will be caught by the test framework
        raise


def create_optimizer(
    model_params: Any, cfg: DictConfig | str | dict[str, Any]
) -> Optimizer:
    """Creates an optimizer based on the provided configuration.

    Args:
        model_params: Parameters of the model to optimize.
        cfg: Configuration for the optimizer. Can be:
             - DictConfig with _target_ key (recommended for Hydra).
             - String with optimizer name (adam, sgd, etc.) for basic use.
             - Dictionary with 'type' key (optimizer name) and other params.

    Returns:
        Optimizer: The instantiated optimizer.
    """
    optimizer_cls = None
    optimizer_params = {}

    if isinstance(cfg, str):
        optimizer_name = cfg.lower()
        if optimizer_name not in _OPTIMIZER_MAP:
            raise ValueError(
                f"Unsupported optimizer name: {optimizer_name}. "
                f"Supported names: {list(_OPTIMIZER_MAP.keys())}"
            )
        optimizer_cls = _OPTIMIZER_MAP[optimizer_name]
        # Use default LR for string-based config for simplicity
        # More specific defaults can be added to _OPTIMIZER_MAP if needed
        optimizer_params = {"lr": 0.001}

    elif isinstance(cfg, dict) and not hasattr(cfg, "get"):  # Plain dict
        if "type" not in cfg:
            raise ValueError(
                "Dictionary optimizer config must contain 'type' key"
            )
        optimizer_type = cfg.pop("type").lower()
        if optimizer_type not in _OPTIMIZER_MAP:
            raise ValueError(
                f"Unsupported optimizer type: {optimizer_type}. "
                f"Supported types: {list(_OPTIMIZER_MAP.keys())}"
            )
        optimizer_cls = _OPTIMIZER_MAP[optimizer_type]
        optimizer_params = cfg  # Remaining items in cfg are params

    elif hasattr(cfg, "get") and cfg.get("_target_"):  # DictConfig from Hydra
        # For DictConfig, Hydra's instantiate handles class resolution and
        # params
        try:
            # Ensure model_params is passed correctly if not automatically
            # handled by partial or if _target_ is a full class path.
            # If _target_ is a partial, model_params might already be bound or
            # should be passed as a kwarg if the partial expects it.
            # Common practice: allow instantiate to handle it if `params` is
            # an arg.
            # Or, if `_partial_` is true, it might return a callable to which
            # we pass params.

            # Assuming _target_ refers to an optimizer class that takes
            # `params` as its first argument
            # and other config values as kwargs.
            # Alternatively, if cfg is a partial that needs `params`, this
            # might vary.
            # For now, we assume `instantiate` can handle `params` if `cfg`
            # is structured correctly for it.
            return cast(Optimizer, instantiate(cfg, params=model_params))
        except Exception as e:
            # Add more specific error handling or logging if needed
            raise ValueError(
                f"Error instantiating optimizer from DictConfig: {cfg} - {e}"
            ) from e
    else:
        raise ValueError(
            f"Invalid optimizer configuration: {cfg}. "
            "Must be a string, DictConfig with _target_, or a dict with "
            "'type' key."
        )

    # This part is reached only if cfg was str or plain dict
    if optimizer_cls is None:
        # This case should ideally not be reached if logic above is correct
        raise ValueError("Could not determine optimizer class.")

    try:
        return cast(Optimizer, optimizer_cls(model_params, **optimizer_params))
    except Exception as e:
        raise ValueError(
            f"Error instantiating optimizer {optimizer_cls.__name__} "
            f"with params {optimizer_params}: {e}"
        ) from e


def create_lr_scheduler(
    optimizer: Optimizer, cfg: DictConfig | str | dict[str, Any] | None
) -> _LRScheduler | None:
    """Creates a learning rate scheduler based on the provided configuration.

    Args:
        optimizer: The optimizer instance.
        cfg: Configuration for the scheduler. Can be:
             - DictConfig with _target_ key (recommended for Hydra).
             - String with scheduler name (step_lr, reduce_lr_on_plateau, etc.)
             - Dictionary with 'type' key (scheduler name) and other params.
             - None (no scheduler).

    Returns:
        Optional[_LRScheduler]: The instantiated scheduler, or None if no
                                scheduler is configured.
    """
    if cfg is None:
        return None

    # Handle DictConfig from Hydra first - it's self-contained for
    # instantiation
    if hasattr(cfg, "get") and cfg.get("_target_"):
        try:
            return cast(
                _LRScheduler | None, instantiate(cfg, optimizer=optimizer)
            )
        except Exception as e:
            raise ValueError(
                f"Error instantiating scheduler from DictConfig: {cfg} - {e}"
            ) from e

    scheduler_name_or_type: str = ""
    params_from_config: dict[str, Any] = {}

    if isinstance(cfg, str):
        scheduler_name_or_type = cfg.lower()
        # Default params will be looked up later using scheduler_name_or_type
    elif isinstance(cfg, dict):  # Plain dict, not DictConfig (already handled)
        if "type" not in cfg:
            raise ValueError(
                "Dictionary scheduler config must contain 'type' key"
            )
        scheduler_name_or_type = cfg.pop("type").lower()
        params_from_config = cfg  # Remaining items are params
    else:
        # This path should ideally not be reached if cfg is one of the Union
        # types and not a DictConfig, str, or dict.
        raise ValueError(
            f"Invalid scheduler configuration type: {type(cfg)}. "
            "Must be None, string, DictConfig with _target_, or dict with "
            "'type' key."
        )

    if not scheduler_name_or_type or scheduler_name_or_type == "none":
        return None

    if scheduler_name_or_type not in _LR_SCHEDULER_MAP:
        raise ValueError(
            f"Unsupported scheduler name/type: {scheduler_name_or_type}. "
            f"Supported: {list(_LR_SCHEDULER_MAP.keys())} or 'none'."
        )
    scheduler_cls = _LR_SCHEDULER_MAP[scheduler_name_or_type]

    final_params: dict[str, Any]
    if isinstance(cfg, str):
        # For string config, use predefined defaults for that scheduler type
        final_params = cast(
            dict[str, Any],
            _LR_SCHEDULER_DEFAULT_PARAMS.get(scheduler_name_or_type, {}),
        )
    else:  # Was a dict, so params_from_config holds user-defined params
        final_params = params_from_config

    try:
        return cast(
            _LRScheduler | None, scheduler_cls(optimizer, **final_params)
        )
    except Exception as e:
        raise ValueError(
            f"Error instantiating scheduler {scheduler_cls.__name__} "
            f"with params {final_params}: {e}"
        ) from e
