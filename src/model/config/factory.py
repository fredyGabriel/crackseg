"""
Model Factory Configuration Processing.

Provides functionality to parse, validate, and normalize model configurations
before instantiation. Handles both standard and hybrid architectures with
proper error reporting and default values.
"""

import logging
from typing import Any

from src.model.factory.registry_setup import (
    architecture_registry,
    bottleneck_registry,
    component_registries,
    decoder_registry,
    encoder_registry,
)

from .instantiation import InstantiationError, instantiate_hybrid_model
from .validation import normalize_config, validate_architecture_config

# Create logger
log = logging.getLogger(__name__)


def parse_component_config(
    config: dict[str, Any], component_type: str
) -> dict[str, Any]:
    """
    Parse and process component configuration.

    Args:
        config: Component configuration
        component_type: Type of component ('encoder', 'bottleneck', 'decoder')

    Returns:
        Dict: Processed configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if "type" not in config:
        msg = f"{component_type} configuration must specify a 'type'"
        raise ValueError(msg)

    component_name = config["type"]
    registry: Any = None

    # Get the appropriate registry
    if component_type == "encoder":
        registry = encoder_registry
    elif component_type == "bottleneck":
        registry = bottleneck_registry
    elif component_type == "decoder":
        registry = decoder_registry
    else:
        # Try to find in component registries
        registry = component_registries.get(component_type)

    # Si el tipo de componente no es reconocido, lanzar ValueError
    if registry is None:
        raise ValueError(
            f"Unknown component type '{component_type}'. "
            f"No registry found for this type."
        )

    # Check if component exists in registry
    if component_name not in registry:
        available = ", ".join(registry.list_components())
        raise ValueError(
            f"Unknown {component_type} type '{component_name}'. "
            f"Available types: {available}"
        )

    # Process any component-specific configuration
    # (This will be expanded in the future based on component needs)

    return config


def _parse_main_components(config: dict[str, Any]) -> dict[str, Any]:
    """Parses the main encoder, bottleneck, and decoder components."""
    if "encoder" in config:
        config["encoder"] = parse_component_config(
            config["encoder"], "encoder"
        )
    if "bottleneck" in config:
        config["bottleneck"] = parse_component_config(
            config["bottleneck"], "bottleneck"
        )
    if "decoder" in config:
        config["decoder"] = parse_component_config(
            config["decoder"], "decoder"
        )
    return config


def _parse_hybrid_components(config: dict[str, Any]) -> dict[str, Any]:
    """Parses additional components for hybrid architectures."""
    if "components" in config:
        for name, comp_config in config["components"].items():
            if "type" not in comp_config:
                raise ValueError(f"Component '{name}' must specify a 'type'")

            component_type_str = None  # Renamed to avoid conflict
            comp_type_val = comp_config["type"]  # Renamed to avoid conflict

            if comp_type_val.endswith("Encoder"):
                component_type_str = "encoder"
            elif comp_type_val.endswith(
                "Bottleneck"
            ) or comp_type_val.endswith("Module"):
                component_type_str = "bottleneck"
            elif comp_type_val.endswith("Decoder"):
                component_type_str = "decoder"
            elif comp_type_val.endswith("Attention") or name == "attention":
                component_type_str = "attention"
            else:
                component_type_str = name  # Use name as a fallback

            if component_type_str is None:
                raise ValueError(
                    f"Cannot determine component type for '{name}'"
                )
            config["components"][name] = parse_component_config(
                comp_config, component_type_str
            )
    return config


def parse_architecture_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Parse and process architecture configuration.

    Args:
        config: Architecture configuration

    Returns:
        Dict: Processed configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if "type" not in config:
        raise ValueError("Architecture configuration must specify a 'type'")

    arch_type = config["type"]

    # Check if architecture exists
    if arch_type not in architecture_registry:
        available = ", ".join(architecture_registry.list_components())
        raise ValueError(
            f"Unknown architecture type '{arch_type}'. "
            f"Available types: {available}"
        )

    # Validate the configuration
    is_valid, errors = validate_architecture_config(config)
    if not is_valid:
        error_messages = []
        # Format error messages to be more readable
        for key, error in errors.items():
            if isinstance(error, dict):
                # Nested errors
                for sub_key, sub_error in error.items():
                    error_messages.append(f"  - {key}.{sub_key}: {sub_error}")
            else:
                error_messages.append(f"  - {key}: {error}")

        error_str = "\n".join(error_messages)
        # Split the message to avoid line too long
        err_prefix = f"Invalid configuration for architecture '{arch_type}':"
        err_msg = f"{err_prefix}\n{error_str}"
        raise ValueError(err_msg)

    # Normalize to fill in defaults
    normalized_config = normalize_config(config)

    # Parse main components
    normalized_config = _parse_main_components(normalized_config)

    # Parse additional components for hybrid architectures
    normalized_config = _parse_hybrid_components(normalized_config)

    return normalized_config


def create_model_from_config(config: dict[str, Any]) -> Any:
    """
    Create a model instance from a configuration dictionary.

    Args:
        config: Model configuration

    Returns:
        Any: Instantiated model

    Raises:
        ValueError: If configuration is invalid or component cannot be created
    """
    # Parse and validate the configuration
    try:
        processed_config = parse_architecture_config(config)

        # Use the specialized hybrid model instantiator
        return instantiate_hybrid_model(processed_config)

    except InstantiationError as e:
        # Convert instantiation errors to ValueError for backward compatibility
        raise ValueError(str(e)) from e


def create_hybrid_model_from_config(config: dict[str, Any]) -> Any:
    """
    Create a hybrid model instance with specialized handling.

    Args:
        config: Model configuration

    Returns:
        Any: Instantiated model

    Raises:
        ValueError: If configuration is invalid or component cannot be created
    """
    # This is a wrapper around create_model_from_config that adds additional
    # handling specific to hybrid architectures if needed
    return create_model_from_config(config)


def get_model_config_schema(model_type: str) -> dict[str, Any]:
    """
    Get a schema describing the configuration options for a model type.

    Args:
        model_type: Type of model

    Returns:
        Dict: Configuration schema with parameter descriptions and defaults

    Raises:
        ValueError: If model type is unknown
    """
    # This can be expanded in the future to provide more detailed schema
    # information for UI configuration builders or documentation

    if model_type not in architecture_registry:
        available = ", ".join(architecture_registry.list_components())
        raise ValueError(
            f"Unknown model type '{model_type}'. Available types: {available}"
        )

    # Basic schema for now, can be expanded with more metadata
    # from the actual component definitions
    schema = {
        "type": {
            "type": "string",
            "required": True,
            "description": "Type of architecture",
            "default": model_type,
        },
        "in_channels": {
            "type": "integer",
            "required": True,
            "description": "Number of input channels",
        },
        "out_channels": {
            "type": "integer",
            "required": True,
            "description": "Number of output channels",
        },
    }

    # Add component schemas based on the model type
    is_hybrid = model_type.startswith("Hybrid") or "Hybrid" in model_type

    # Basic components for all architecture types
    schema["encoder"] = {
        "type": "object",
        "required": True,
        "description": "Encoder configuration",
    }

    schema["bottleneck"] = {
        "type": "object",
        "required": True,
        "description": "Bottleneck configuration",
    }

    schema["decoder"] = {
        "type": "object",
        "required": True,
        "description": "Decoder configuration",
    }

    # For hybrid architectures, add components field
    if is_hybrid:
        schema["components"] = {
            "type": "object",
            "required": False,
            "description": "Additional components for hybrid architectures",
        }

    return schema
