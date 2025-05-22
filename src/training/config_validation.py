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

    # Permitir acceso por atributo o clave
    def get_field(obj, field):
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
        # Para DictConfig, OmegaConf puede devolver tipos especiales
        # pero isinstance(value, int) funciona para ambos
        if not isinstance(value, expected_type):
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
