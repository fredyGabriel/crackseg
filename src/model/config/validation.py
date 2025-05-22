"""
Configuration Validation System.

This module provides high-level functions for validating model configurations:
- validate_component_config: Validates a component configuration
- validate_architecture_config: Validates a complete architecture configuration
- normalize_config: Normalizes a configuration by filling in default values
"""

import logging
from typing import Any

from .schemas import (
    create_architecture_schema,
    create_bottleneck_schema,
    create_decoder_schema,
    create_encoder_schema,
    create_hybrid_schema,
    # Add other necessary schema imports
)

# Create logger
log = logging.getLogger(__name__)


def validate_component_config(
    config: dict[str, Any], component_type: str
) -> tuple[bool, dict[str, Any]]:
    """
    Validate a component configuration.

    Args:
        config: Configuration dictionary
        component_type: Type of component ('encoder', 'bottleneck', etc.)

    Returns:
        tuple: (is_valid, errors)
    """
    # print(
    #     f"[DEBUG] validate_component_config: type={type(config)}, "
    #     f"value={config}"
    # ) # Keep commented out unless debugging
    if not isinstance(config, dict):
        return False, {
            "_general": (
                f"Configuration must be a dictionary, got {type(config)}: "
                f"{config}"
            )
        }

    # Get component-specific validator
    comp_type_from_config = config.get(
        "type"
    )  # Type specified within the component's own config
    schema = None

    # Try to get schema based on the specific type declared in the component's
    # config
    if comp_type_from_config:
        if (
            comp_type_from_config == "encoder"
        ):  # Or more specific types like "ResNetEncoder" if factories exist
            schema = create_encoder_schema()
        elif comp_type_from_config == "bottleneck":
            schema = create_bottleneck_schema()
        elif comp_type_from_config == "decoder":
            schema = create_decoder_schema()
        # Add elif for other specific known types from config that map to a
        # schema factory
        else:
            log.debug(
                "No specific schema factory for comp_type "
                f"'{comp_type_from_config}' from config. "
                f"Will try category '{component_type}'."
            )
            # schema remains None, will fallback to component_type argument

    # If no schema was determined from comp_type_from_config, fallback to
    # component_type argument
    if schema is None:
        if component_type == "encoder":
            schema = create_encoder_schema()
        elif component_type == "bottleneck":
            schema = create_bottleneck_schema()
        elif component_type == "decoder":
            schema = create_decoder_schema()
        else:
            log.error(
                f"Unknown component type for validation: '{component_type}' "
                "and no specific schema found for config type "
                f"'{comp_type_from_config}'."
            )
            return False, {
                "_general": "Unknown component type or category for "
                f"validation: {component_type}"
            }

    # Attempt validation with the determined schema
    is_valid, errors = schema.validate(config)

    return is_valid, errors


def _validate_main_components(
    config: dict[str, Any], errors: dict[str, Any]
) -> bool:
    """Validates encoder, bottleneck, and decoder components in the config."""
    all_main_components_valid = True
    for comp_name in ["encoder", "bottleneck", "decoder"]:
        if comp_name in config:
            comp_cfg = config[comp_name]
            if not isinstance(comp_cfg, dict):
                # Skip validation if not a dict (avoids bug of passing str)
                # errors[comp_name] = f"Component '{comp_name}' config must be
                # a dictionary, got {type(comp_cfg)}"
                # all_main_components_valid = False # Or handle as error
                continue
            comp_valid, comp_err = validate_component_config(
                comp_cfg, comp_name
            )
            if not comp_valid:
                all_main_components_valid = False
                if comp_name in errors:
                    if isinstance(errors[comp_name], dict) and isinstance(
                        comp_err, dict
                    ):
                        errors[comp_name].update(comp_err)
                    else:
                        errors[comp_name] = comp_err
                else:
                    errors[comp_name] = comp_err
    return all_main_components_valid


def validate_architecture_config(
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Validate the entire architecture configuration."""
    arch_type = config.get("type")
    if not arch_type:
        return False, {"type": "Architecture type is missing"}

    # Determine which schema to use
    if "components" in config:
        # Use Hybrid schema if 'components' key is present
        schema = create_hybrid_schema()
        log.debug("Using HybridModelSchema for validation.")
    else:
        # Use standard architecture schema otherwise
        schema = create_architecture_schema()
        log.debug(f"Using standard ArchitectureSchema for {arch_type}.")

    is_valid, errors = schema.validate(config)
    # Ensure errors is a dict even if validation passes initially
    if errors is None:
        errors = {}

    # If basic validation passes, validate nested components
    if is_valid:
        # Validate main components using the helper function
        main_components_valid = _validate_main_components(config, errors)
        if not main_components_valid:
            is_valid = False

        # REMOVED: Detailed validation of additional components here.
        # The Hybrid schema allows unknown fields, and specific validation
        # will occur during instantiation where more context is available.

    return is_valid, errors


# Helper for normalizing individual components
def _normalize_individual_component(
    config_to_normalize: dict[str, Any], component_type_str: str
) -> dict[str, Any]:
    """Normalizes an individual component configuration using its specific
    schema."""
    schema = None
    if component_type_str == "encoder":
        schema = create_encoder_schema()
    elif component_type_str == "bottleneck":
        schema = create_bottleneck_schema()
    elif component_type_str == "decoder":
        schema = create_decoder_schema()
    # Add other specific component types if they have distinct schemas for
    # normalization
    # else:
    #     log.debug(f"No specific normalization schema for component type:
    # {component_type_str}")

    if schema:
        return schema.normalize(config_to_normalize)
    return (
        config_to_normalize  # Return as is if no specific schema for this type
    )


# Helper for normalizing the 'components' section in hybrid architectures
def _normalize_hybrid_additional_components(
    components_section: dict[str, Any],
) -> dict[str, Any]:
    """Normalizes the 'components' dictionary within a hybrid architecture
    config."""
    normalized_additional_components = {}
    if isinstance(components_section, dict):
        for comp_name, comp_config in components_section.items():
            if isinstance(comp_config, dict) and "type" in comp_config:
                # Extract base component type for recursive normalization call
                # This heuristic assumes types like "SpecificEncoder_v1" ->
                # "encoder"
                base_comp_type = comp_config["type"].split("_")[0].lower()
                normalized_additional_components[comp_name] = normalize_config(
                    comp_config, base_comp_type
                )
            else:
                # Pass through if not a typical component structure or type is
                # missing
                normalized_additional_components[comp_name] = comp_config
    else:
        # If 'components' is not a dict, return it as is or log a warning
        log.warning(
            "Expected 'components' section to be a dict, got "
            f"{type(components_section)}"
        )
        return components_section
    return normalized_additional_components


# Helper for normalizing a full architecture configuration
def _normalize_full_architecture(
    original_config: dict[str, Any],  # Used to check 'type' for hybrid
    config_to_normalize: dict[str, Any],
) -> dict[str, Any]:
    """Normalizes a complete architecture configuration, including nested
    components."""
    arch_type_str = original_config.get("type", "").lower()
    is_hybrid = arch_type_str == "hybrid"  # More direct check

    schema = (
        create_hybrid_schema() if is_hybrid else create_architecture_schema()
    )
    normalized_arch = schema.normalize(config_to_normalize)

    # Normalize main nested components (encoder, bottleneck, decoder)
    for comp_key in ["encoder", "bottleneck", "decoder"]:
        if comp_key in normalized_arch and isinstance(
            normalized_arch[comp_key], dict
        ):
            # Use the main normalize_config for these, as it routes to
            # _normalize_individual_component
            normalized_arch[comp_key] = normalize_config(
                normalized_arch[comp_key], comp_key
            )

    # Normalize additional components for hybrid architectures
    if is_hybrid and "components" in normalized_arch:
        normalized_arch["components"] = (
            _normalize_hybrid_additional_components(
                normalized_arch["components"]
            )
        )
    return normalized_arch


def normalize_config(
    config: dict[str, Any], component_type: str | None = None
) -> dict[str, Any]:
    """
    Normalize a configuration by filling in default values.

    Args:
        config: Configuration dictionary to normalize
        component_type: Type of component for specific normalization

    Returns:
        Dict: Normalized configuration
    """
    if not isinstance(config, dict):
        log.warning(
            f"Cannot normalize non-dictionary configuration: {type(config)}"
        )
        return config

    # Work on a copy to avoid modifying the original config dict in-place
    # This is important if the same config object is used elsewhere.
    current_config_state = dict(config)

    if component_type is None and "type" in current_config_state:
        # This path handles top-level architecture configurations.
        # 'config' is the original config to check its 'type' for hybrid status
        # 'current_config_state' is the one being progressively normalized.
        return _normalize_full_architecture(config, current_config_state)
    elif component_type is not None:
        # This path handles individual components (e.g., encoder, bottleneck).
        return _normalize_individual_component(
            current_config_state, component_type
        )
    else:
        # Fallback for ambiguous cases (e.g., component_type is None but no
        # 'type' key,
        # or a generic sub-dictionary not matching other criteria).
        # Returns the current state (a copy of the original config if no path
        # was taken).
        log.debug(
            "normalize_config: Ambiguous case or generic dict. "
            f"component_type='{component_type}', "
            f"'type' in config='{'type' in current_config_state}'. "
            "Returning as is."
        )
        return current_config_state
