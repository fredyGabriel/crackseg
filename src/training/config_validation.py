"""Helpers for validating Trainer configuration parameters."""

from typing import Any

from omegaconf import DictConfig


def validate_trainer_config(cfg: Any) -> None:
    """Validates required fields and types in Trainer config.
    Raises ValueError if invalid.
    Accepts dict, DictConfig, or object with attributes.
    """
    required_fields = [
        ("epochs", int),
        ("device", str),
        ("optimizer", (dict, DictConfig)),
        ("gradient_accumulation_steps", int),
    ]

    # Allow access by attribute or key
    def get_field(obj: Any, field: str) -> Any:
        if isinstance(obj, dict | DictConfig):
            if field in obj:
                return obj[field]
            else:
                raise ValueError(f"Missing required config field: '{field}'")
        elif hasattr(obj, field):
            return getattr(obj, field)
        else:
            raise ValueError(f"Missing required config field: '{field}'")

    for field, expected_type in required_fields:
        value = get_field(cfg, field)
        # For DictConfig, OmegaConf may return special types
        # but isinstance(value, int) works for both
        if field == "optimizer":
            if not (isinstance(value, dict) or isinstance(value, DictConfig)):
                raise TypeError(
                    f"Config field '{field}' must be dict or DictConfig, got "
                    f"{type(value)}"
                )
        else:
            # We ensure that expected_type is a class or tuple of classes
            if not isinstance(expected_type, tuple):
                types_tuple = (expected_type,)
            else:
                types_tuple = expected_type
            # We filter only real classes and cast directly in isinstance
            if not isinstance(value, types_tuple):
                raise TypeError(
                    f"Config field '{field}' must be {expected_type}, "
                    f"got {type(value)}"
                )
    # Example: epochs must be > 0
    epochs = get_field(cfg, "epochs")
    if epochs <= 0:
        raise ValueError("'epochs' must be > 0")
    grad_accum = get_field(cfg, "gradient_accumulation_steps")
    if grad_accum < 1:
        raise ValueError("'gradient_accumulation_steps' must be >= 1")
    # Optimizer config must have at least '_target_' and 'lr'
    opt_cfg = get_field(cfg, "optimizer")
    if (
        not isinstance(opt_cfg, dict | DictConfig)
        or "_target_" not in opt_cfg
        or "lr" not in opt_cfg
    ):
        raise ValueError("'optimizer' config must have '_target_' and 'lr'")
