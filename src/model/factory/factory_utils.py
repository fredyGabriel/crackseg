"""
Factory utilities for component configuration and instantiation.

Provides helper functions for configuration validation, component handling,
configuration transformations, and logging to support the factory module.
"""

import logging
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

# Type variable for better type hinting
T = TypeVar("T")
ConfigDict = dict[str, Any]

# Create logger
log = logging.getLogger(__name__)

# Module-level variable to store the configured value for max items in config
# logs
# Default value, can be overridden by calling
# set_max_items_to_log_in_config_repr
_max_items_to_log_in_config_repr: int = 10


def get_max_items_to_log_in_config_repr() -> int:
    """Returns the configured maximum number of items to log from a config
    dict."""
    return _max_items_to_log_in_config_repr


def set_max_items_to_log_in_config_repr(value: int) -> None:
    """Sets the maximum number of items to log from a config dict.

    This function should be called by the main application script after
    loading the Hydra configuration.

    Args:
        value (int): The maximum number of items. Must be non-negative.
    """
    if value < 0:
        # Or log a warning and use a default, but raising an error is safer
        # for an invalid configuration.
        raise ValueError(
            "Maximum items to log must be a non-negative integer."
        )
    _max_items_to_log_in_config_repr = value


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


#
# Configuration Validation
#
def validate_config(
    config: dict[str, Any], required_keys: list[str], component_type: str
) -> None:
    """
    Validate that a configuration dictionary contains all required keys.

    Args:
        config (Dict[str, Any]): Configuration dictionary to validate.
        required_keys (List[str]): List of keys that must be present.
        component_type (str): Type of component for error messages.

    Raises:
        ConfigurationError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigurationError(
            f"Missing required configuration for {component_type}: "
            f"{', '.join(missing_keys)}"
        )


def validate_component_types(
    config: dict[str, Any], type_map: dict[str, list[str]]
) -> None:
    """
    Validate that component types in configuration are allowed values.

    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        type_map (Dict[str, List[str]]): Map of component keys to allowed
        values

    Raises:
        ConfigurationError: If a component type is not an allowed value
    """
    for key, allowed_types in type_map.items():
        if key in config:
            component_type = config[key]
            if component_type not in allowed_types:
                raise ConfigurationError(
                    f"Invalid {key} type: '{component_type}'. "
                    f"Allowed types: {', '.join(allowed_types)}"
                )


def check_parameter_types(
    params: dict[str, Any], type_specs: dict[str, type]
) -> None:
    """
    Validate that parameters have the correct types.

    Args:
        params (Dict[str, Any]): Parameter dictionary to validate
        type_specs (Dict[str, Type]): Map of parameter names to expected types

    Raises:
        ConfigurationError: If a parameter has the wrong type
    """
    for param_name, expected_type in type_specs.items():
        if param_name in params:
            value = params[param_name]
            if not isinstance(value, expected_type):
                raise ConfigurationError(
                    f"Parameter '{param_name}' has wrong type. "
                    f"Expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )


#
# Configuration Transformation
#
def hydra_to_dict(
    config: DictConfig | dict[str, Any], resolve: bool = True
) -> dict[str, Any]:
    """
    Safely convert a Hydra OmegaConf config to a standard dictionary.

    Args:
        config (Union[DictConfig, Dict[str, Any]]): Config to convert
        resolve (bool): Whether to resolve interpolations

    Returns:
        Dict[str, Any]: Standard Python dictionary
    """
    if isinstance(config, DictConfig):
        result = OmegaConf.to_container(config, resolve=resolve)
        if not isinstance(result, dict):
            raise TypeError("Config could not be converted to dict")
        return cast(dict[str, Any], result)
    # If already a dict, return a copy to avoid modifying the original
    return config.copy()


def extract_runtime_params(
    component: Any, param_mappings: dict[str, str]
) -> dict[str, Any]:
    """
    Extract runtime parameters from a component based on mappings.

    Args:
        component (Any): Component to extract parameters from
        param_mappings (Dict[str, str]): Map source attributes to target params

    Returns:
        Dict[str, Any]: Dictionary of extracted parameters
    """
    runtime_params = {}
    for src_attr, target_param in param_mappings.items():
        if hasattr(component, src_attr):
            runtime_params[target_param] = getattr(component, src_attr)
    return runtime_params


def merge_configs(
    base_config: dict[str, Any], override_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.

    Args:
        base_config (Dict[str, Any]): Base configuration
        override_config (Dict[str, Any]): Override configuration

    Returns:
        Dict[str, Any]: Merged configuration
    """
    result = base_config.copy()
    result.update(override_config)
    return result


def filter_config(
    config: dict[str, Any],
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
) -> dict[str, Any]:
    """
    Filter a configuration dictionary by including or excluding specific keys.

    Args:
        config (Dict[str, Any]): Configuration to filter
        include_keys (Optional[Set[str]]): Keys to include
            (if None, include all)
        exclude_keys (Optional[Set[str]]): Keys to exclude

    Returns:
        Dict[str, Any]: Filtered configuration
    """
    result = {}
    exclude_keys = exclude_keys or set()

    if include_keys is not None:
        # Include specified keys, except those in exclude_keys
        for key in include_keys:
            if key in config and key not in exclude_keys:
                result[key] = config[key]
    else:
        # Include all keys except those in exclude_keys
        for key, value in config.items():
            if key not in exclude_keys:
                result[key] = value

    return result


#
# Logging Helpers
#
def log_component_creation(
    component_type: str, component_name: str, level: int = logging.INFO
) -> None:
    """
    Log the creation of a component with consistent formatting.

    Args:
        component_type (str): Type of component (e.g., 'Encoder')
        component_name (str): Name of the component
        level (int): Logging level
    """
    log.log(level, f"Instantiated {component_type}: {component_name}")


def log_configuration_error(
    error_type: str,
    details: str,
    config: dict[str, Any] | None = None,
    level: int = logging.ERROR,
) -> None:
    """
    Log a configuration error with consistent formatting.

    Args:
        error_type (str): Type of error
        details (str): Error details
        config (Optional[Dict[str, Any]]): Configuration that caused the error
        level (int): Logging level
    """
    message = f"Configuration error ({error_type}): {details}"
    if config is not None:
        # Log first N key-value pairs to avoid overwhelming logs
        limit = get_max_items_to_log_in_config_repr()
        if isinstance(config, dict):
            config_items = list(config.items())
            # Log first 'limit' key-value pairs
            config_str = str(
                {k: v for i, (k, v) in enumerate(config_items) if i < limit}
            )
            if len(config_items) > limit:
                config_str = config_str[:-1] + ", ...}"  # Add ellipsis
        elif hasattr(config, "__dict__"):  # Handle dataclasses/objects
            config_items = list(vars(config).items())
            config_str = str(
                {k: v for i, (k, v) in enumerate(config_items) if i < limit}
            )
            if len(config_items) > limit:
                config_str = config_str[:-1] + ", ...}"
        else:
            config_str = str(config)  # Fallback for other types

        message += f"\nConfig (first {limit} items): {config_str}"
    log.log(level, message)
