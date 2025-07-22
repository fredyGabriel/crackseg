"""
Configuration template creation utilities. This module provides
functionality for creating new configuration files from templates with
customizable overrides.
"""

import logging
from pathlib import Path

import yaml

from .exceptions import ConfigError
from .io import load_config_file

logger = logging.getLogger(__name__)


def create_config_from_template(
    template_path: str,
    output_path: str,
    overrides: dict[str, object] | None = None,
) -> None:
    """
    Create a new configuration file from a template. Args: template_path:
    Path to the template configuration file. output_path: Path where the
    new configuration should be saved. overrides: Optional dictionary of
    values to override in the template. Raises: ConfigError: If the
    operation fails.
    """
    try:
        # Load template
        template_config = load_config_file(template_path)

        # Apply overrides if provided
        if overrides:
            _apply_overrides(template_config, overrides)

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save new configuration
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                template_config, f, default_flow_style=False, sort_keys=False
            )

        logger.info(f"Created configuration: {output_path}")

    except Exception as e:
        raise ConfigError(
            f"Failed to create configuration from template: {e}"
        ) from e


def _apply_overrides(
    config: dict[str, object], overrides: dict[str, object]
) -> None:
    """
    Recursively apply overrides to a configuration dictionary (in-place).
    Args: config: Configuration dictionary to modify. overrides:
    Dictionary of overrides to apply.
    """
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "model.encoder.type"
            parts = key.split(".")
            current: dict[str, object] = config

            for part in parts[:-1]:
                if not isinstance(current.get(part), dict):
                    current[part] = {}
                current = current[part]  # type: ignore

            current[parts[-1]] = value
        elif isinstance(config.get(key), dict) and isinstance(value, dict):
            # If both existing and override values are dicts, merge them
            _apply_overrides(config[key], value)  # type: ignore
        else:
            # Simple key or type mismatch replacement
            config[key] = value
