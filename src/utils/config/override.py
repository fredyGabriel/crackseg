"""Configuration override utilities."""

from typing import Dict, Any, Optional, List
from copy import deepcopy

from omegaconf import DictConfig, OmegaConf

from src.utils.exceptions import ConfigError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def override_config(
    config: DictConfig,
    overrides: Dict[str, Any],
    strict: bool = True
) -> DictConfig:
    """Override configuration values.

    Args:
        config: Base configuration to override
        overrides: Dictionary of dot-notation paths and values to override
        strict: If True, raises error for non-existent paths

    Returns:
        Updated configuration

    Raises:
        ConfigError: If a path doesn't exist and strict=True
    """
    # Create a deep copy to avoid modifying the original
    config = deepcopy(config)

    for path, value in overrides.items():
        try:
            # Convert path to list of keys
            keys = path.split('.')

            # Navigate to the parent node
            node = config
            for key in keys[:-1]:
                if not OmegaConf.is_config(node) or key not in node:
                    if strict:
                        raise ConfigError(
                            f"Invalid config path: {path}",
                            details=f"Key '{key}' not found"
                        )
                    logger.warning(
                        f"Skipping override for non-existent path: {path}"
                    )
                    break
                node = node[key]
            else:
                # Set the value
                last_key = keys[-1]
                if not OmegaConf.is_config(node) or (
                    strict and last_key not in node
                ):
                    if strict:
                        raise ConfigError(
                            f"Invalid config path: {path}",
                            details=f"Key '{last_key}' not found"
                        )
                    logger.warning(
                        f"Skipping override for non-existent path: {path}"
                    )
                else:
                    node[last_key] = value
                    logger.debug(f"Overrode config value at {path}")

        except Exception as e:
            if strict:
                raise ConfigError(
                    f"Failed to override config at {path}",
                    details=str(e)
                )
            logger.warning(
                f"Failed to override config at {path}: {e}"
            )

    return config


def apply_overrides(
    cfg: DictConfig, overrides: Optional[List[str]] = None
) -> DictConfig:
    """Apply a list of override strings to a configuration using Hydra syntax.

    Args:
        cfg: Base configuration to override
        overrides: List of override strings in Hydra format (e.g., 'a.b=c')

    Returns:
        Updated configuration with overrides applied
    """
    if overrides is None:
        overrides = []
    # Convert override strings to a DictConfig and merge with base config
    override_conf = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, override_conf)


def save_config(cfg: DictConfig, path: str) -> None:
    """Save the configuration to a YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f.name)


def example_override_usage() -> None:
    """Example: how to use overrides with Hydra from the command line."""
    print("Command-line override example:")
    print(
        "python main.py "
        "training.epochs=100 "
        "model.model_name=deeplabv3+"
    )
    print(
        "This will override the values of epochs and "
        "model_name in the configuration."
    )

# Note: actual override integration is done when loading the config with Hydra,
# using the 'overrides' parameter in compose or from the CLI.
# These functions allow you to manipulate and save configurations
# programmatically if needed.
