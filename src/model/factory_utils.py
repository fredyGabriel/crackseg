"""
Factory utilities for component configuration and instantiation.

Provides helper functions for configuration validation, component handling,
configuration transformations, and logging to support the factory module.
"""

import logging
from typing import Dict, Any, List, Optional, Type, TypeVar, Union, Set
from omegaconf import DictConfig, OmegaConf

# Type variable for better type hinting
T = TypeVar('T')
ConfigDict = Dict[str, Any]

# Create logger
log = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""
    pass


#
# Configuration Validation
#
def validate_config(config: Dict[str, Any], required_keys: List[str],
                    component_type: str) -> None:
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


def validate_component_types(config: Dict[str, Any],
                             type_map: Dict[str, List[str]]) -> None:
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


def check_parameter_types(params: Dict[str, Any],
                          type_specs: Dict[str, Type]) -> None:
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
def hydra_to_dict(config: Union[DictConfig, Dict[str, Any]],
                  resolve: bool = True) -> Dict[str, Any]:
    """
    Safely convert a Hydra OmegaConf config to a standard dictionary.

    Args:
        config (Union[DictConfig, Dict[str, Any]]): Config to convert
        resolve (bool): Whether to resolve interpolations

    Returns:
        Dict[str, Any]: Standard Python dictionary
    """
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=resolve)
    # If already a dict, return as is
    return config


def extract_runtime_params(component: Any,
                           param_mappings: Dict[str, str]) -> Dict[str, Any]:
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


def merge_configs(base_config: Dict[str, Any],
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
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


def filter_config(config: Dict[str, Any],
                  include_keys: Optional[Set[str]] = None,
                  exclude_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
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
def log_component_creation(component_type: str,
                           component_name: str,
                           level: int = logging.INFO) -> None:
    """
    Log the creation of a component with consistent formatting.

    Args:
        component_type (str): Type of component (e.g., 'Encoder')
        component_name (str): Name of the component
        level (int): Logging level
    """
    log.log(level, f"Instantiated {component_type}: {component_name}")


def log_configuration_error(error_type: str, details: str,
                            config: Optional[Dict[str, Any]] = None,
                            level: int = logging.ERROR) -> None:
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
        # Log first 10 key-value pairs to avoid overwhelming logs
        config_str = str({k: v for i, (k, v) in
                         enumerate(config.items()) if i < 10})
        if len(config) > 10:
            config_str = config_str[:-1] + ", ...}"
        message += f"\nConfig: {config_str}"
    log.log(level, message)
