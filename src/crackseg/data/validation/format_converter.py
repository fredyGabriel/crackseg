"""Format conversion utilities for transform configuration.

This module provides utilities for converting between different transform
configuration formats for compatibility with various libraries.
"""

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def _convert_albumentations_format_to_standard(
    transform_config: (
        list[dict[str, Any] | DictConfig]
        | ListConfig
        | dict[str, Any]
        | DictConfig
    ),
) -> list[dict[str, Any] | DictConfig]:
    """Convert Albumentations format to standard format.

    Args:
        transform_config: Transform configuration in various formats.

    Returns:
        List of transform specifications in standard format.
    """
    if isinstance(transform_config, list | ListConfig):
        # Already in list format
        transforms_list = list(transform_config)
        result = []

        for transform_spec in transforms_list:
            if isinstance(transform_spec, dict):
                # Standard format: {"name": "Resize", "params": {...}}
                if "name" in transform_spec and "params" in transform_spec:
                    result.append(transform_spec)
                else:
                    # Albumentations format: {"Resize": {...}}
                    for name, params in transform_spec.items():
                        result.append({"name": name, "params": params})
            elif isinstance(transform_spec, DictConfig):
                # Convert DictConfig to dict
                spec_dict = OmegaConf.to_container(
                    transform_spec, resolve=True
                )
                if isinstance(spec_dict, dict):
                    if "name" in spec_dict and "params" in spec_dict:
                        result.append(transform_spec)
                    else:
                        # Albumentations format
                        for name, params in spec_dict.items():
                            result.append({"name": name, "params": params})

        return result

    elif isinstance(transform_config, dict | DictConfig):
        # Dictionary format: {"Resize": {...}, "Normalize": {...}}
        if isinstance(transform_config, DictConfig):
            config_dict = OmegaConf.to_container(
                transform_config, resolve=True
            )
            if not isinstance(config_dict, dict):
                raise ValueError("DictConfig must resolve to a dictionary")
        else:
            config_dict = transform_config

        result = []
        for name, params in config_dict.items():
            result.append({"name": name, "params": params})

        return result

    else:
        raise ValueError(
            f"Unsupported transform_config type: {type(transform_config)}"
        )


def convert_transform_format(
    transform_config: (
        list[dict[str, Any] | DictConfig]
        | ListConfig
        | dict[str, Any]
        | DictConfig
    ),
    target_format: str = "standard",
) -> list[dict[str, Any] | DictConfig]:
    """Convert transform configuration to target format.

    Args:
        transform_config: Transform configuration in any supported format.
        target_format: Target format ("standard" or "albumentations").

    Returns:
        Transform configuration in target format.
    """
    if target_format == "standard":
        return _convert_albumentations_format_to_standard(transform_config)
    elif target_format == "albumentations":
        # Convert standard format to Albumentations format
        standard_list = _convert_albumentations_format_to_standard(
            transform_config
        )
        result = {}
        for transform_spec in standard_list:
            if isinstance(transform_spec, dict):
                name = transform_spec.get("name")
                params = transform_spec.get("params", {})
                if name:
                    result[name] = params
        return result
    else:
        raise ValueError(f"Unsupported target_format: {target_format}")
