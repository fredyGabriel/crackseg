"""
Final Activation Handler.

This module handles the application of final activation layers to models
based on configuration. It supports both simple activation names and
complex activation configurations with parameters.
"""

import logging
from typing import Any

from torch import nn

from .components import InstantiationError

# Create logger
log = logging.getLogger(__name__)


def _create_activation_from_dict(
    activation_config: dict[str, Any],
) -> nn.Module | None:
    """Create activation layer from dictionary configuration.

    Args:
        activation_config: Dictionary with 'type' and optional parameters

    Returns:
        Instantiated activation layer or None if failed

    Raises:
        InstantiationError: If activation creation fails
    """
    if "type" not in activation_config:
        return None

    act_type = activation_config["type"]
    act_params = {k: v for k, v in activation_config.items() if k != "type"}

    try:
        # Try importing from torch.nn first
        if hasattr(nn, act_type):
            activation_class = getattr(nn, act_type)
        else:
            # Try dynamic import for custom activations
            module_name, class_name = act_type.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            activation_class = getattr(module, class_name)

        activation_instance = activation_class(**act_params)
        log.info(f"Created activation layer: {act_type}")
        return activation_instance

    except (AttributeError, ImportError, TypeError, ValueError) as e:
        raise InstantiationError(
            f"Failed to create activation '{act_type}': {e}"
        ) from e


def _create_activation_from_string(activation_name: str) -> nn.Module | None:
    """Create activation layer from string name.

    Args:
        activation_name: Name of the activation (e.g., "ReLU", "Sigmoid")

    Returns:
        Instantiated activation layer or None if failed

    Raises:
        InstantiationError: If activation creation fails
    """
    try:
        if not hasattr(nn, activation_name):
            raise InstantiationError(
                f"Unknown activation '{activation_name}' in torch.nn"
            )

        activation_class = getattr(nn, activation_name)
        activation_instance = activation_class()
        log.info(f"Created activation layer: {activation_name}")
        return activation_instance

    except (AttributeError, TypeError, ValueError) as e:
        raise InstantiationError(
            f"Failed to create activation '{activation_name}': {e}"
        ) from e


def apply_final_activation(
    model: nn.Module, config: dict[str, Any]
) -> nn.Module:
    """Apply a final activation layer to the model if configured.

    Args:
        model: The base model to apply activation to
        config: Configuration dictionary that may contain 'final_activation'

    Returns:
        Model with final activation applied, or original model if no activation

    Raises:
        InstantiationError: If activation creation fails
    """
    if "final_activation" not in config:
        return model

    activation_config = config["final_activation"]
    activation_instance = None

    try:
        if isinstance(activation_config, dict):
            activation_instance = _create_activation_from_dict(
                activation_config
            )
        elif isinstance(activation_config, str):
            activation_instance = _create_activation_from_string(
                activation_config
            )
        else:
            log.warning(
                f"Ignoring invalid 'final_activation' config: {activation_config}"
            )
            return model

        if activation_instance:
            return nn.Sequential(model, activation_instance)
        else:
            log.warning(
                f"Failed to create activation from config: {activation_config}"
            )
            return model

    except InstantiationError:
        # Re-raise InstantiationError directly
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise InstantiationError(
            f"Unexpected error applying final activation: {e}"
        ) from e
