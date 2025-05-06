"""
Configuration Validation System.

This module provides high-level functions for validating model configurations:
- validate_component_config: Validates a component configuration
- validate_architecture_config: Validates a complete architecture configuration
- normalize_config: Normalizes a configuration by filling in default values
"""

from typing import Dict, Any, Optional, Tuple
import logging

from .schemas import (
    create_encoder_schema,
    create_bottleneck_schema,
    create_decoder_schema,
    create_architecture_schema,
    create_hybrid_schema,
    # Add other necessary schema imports
)

# Create logger
log = logging.getLogger(__name__)


def validate_component_config(
    config: Dict[str, Any], component_type: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a component configuration.

    Args:
        config: Configuration dictionary
        component_type: Type of component ('encoder', 'bottleneck', etc.)

    Returns:
        tuple: (is_valid, errors)
    """
    print(
        f"[DEBUG] validate_component_config: type={type(config)}, "
        f"value={config}"
    )
    if not isinstance(config, dict):
        return False, {
            "_general": (
                f"Configuration must be a dictionary, got {type(config)}: "
                f"{config}"
            )
        }

    # Get component-specific validator
    comp_type = config.get('type')
    if comp_type:
        # Use specific schema factories if available
        if comp_type == "encoder":  # This condition might be too broad
            schema = create_encoder_schema()
        elif comp_type == "bottleneck":
            schema = create_bottleneck_schema()
        elif comp_type == "decoder":
            schema = create_decoder_schema()
        else:
            # Attempt to find a specific validator function if defined
            # This part depends on how specific validators are structured
            # Let's assume for now we fall back to generic
            log.debug(
                f"No specific schema factory for comp_type {comp_type},"
                f" using generic."
            )
            schema = None  # Signal to use generic below
    else:
        # Fall back to generic schema validation based on component_type
        log.debug(f"Using generic schema for {component_type}")
        schema = None

    # If no specific schema was found by comp_type, use component_type
    if schema is None:
        if component_type == 'encoder':
            schema = create_encoder_schema()
        elif component_type == 'bottleneck':
            schema = create_bottleneck_schema()
        elif component_type == 'decoder':
            schema = create_decoder_schema()
        else:
            # Handle unknown component types more robustly
            log.error(
                f"Unknown component type for validation: {component_type}"
            )
            return False, {
                "_general": f"Unknown component type: {component_type}"
            }

    # Special handling for hybrid schema - might need refinement
    # This logic might be complex if called for non-architecture components
    if "components" in config and schema.name == "architecture":
        # Check schema name
        schema = create_hybrid_schema()

    # Attempt validation
    is_valid, errors = schema.validate(config)

    return is_valid, errors


def validate_architecture_config(
    config: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
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
        component_errors = {}
        for comp_name in ["encoder", "bottleneck", "decoder"]:
            if comp_name in config:
                comp_cfg = config[comp_name]
                if not isinstance(comp_cfg, dict):
                    # Saltar validación si no es un dict (evita bug de pasar str)
                    continue
                comp_valid, comp_err = validate_component_config(
                    comp_cfg, comp_name
                )
                if not comp_valid:
                    component_errors[comp_name] = comp_err

        # Validate additional components if present (hybrid)
        # REMOVED: Detailed validation of additional components here.
        # The Hybrid schema allows unknown fields, and specific validation
        # will occur during instantiation where more context is available.
        # if "components" in config:
        #     additional_comp_errors = {}
        #     for name, comp_cfg in config["components"].items():
        #         # For now, use generic component validation
        #         comp_valid, comp_err = validate_component_config(
        #             comp_cfg,
        #             name  # Use component name as type hint for now
        #         )
        #         if not comp_valid:
        #             additional_comp_errors[name] = comp_err
        #     if additional_comp_errors:
        #          component_errors["components"] = additional_comp_errors

        if component_errors:
            is_valid = False
            # Merge component errors into main error dict
            for key, val in component_errors.items():
                if key in errors:
                    # Simple merge, might need refinement
                    if isinstance(errors[key], dict) and isinstance(val, dict):
                        errors[key].update(val)
                    else:
                        # Handle non-dict merge case if necessary
                        errors[key] = val
                else:
                    errors[key] = val

    return is_valid, errors


def normalize_config(
    config: Dict[str, Any], component_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalize a configuration by filling in default values.

    Args:
        config: Configuration dictionary to normalize
        component_type: Type of component for specific normalization

    Returns:
        Dict: Normalized configuration
    """
    if not isinstance(config, dict):
        log.warning("Cannot normalize non-dictionary configuration")
        return config

    normalized = dict(config)

    # For complete architecture configurations
    if component_type is None and 'type' in config:
        is_hybrid = config['type'].lower() == 'hybrid'
        schema = create_hybrid_schema() if is_hybrid else \
            create_architecture_schema()
        normalized = schema.normalize(normalized)

        # Normalize nested components
        if 'encoder' in normalized:
            normalized['encoder'] = normalize_config(
                normalized['encoder'], 'encoder'
            )

        if 'bottleneck' in normalized:
            normalized['bottleneck'] = normalize_config(
                normalized['bottleneck'], 'bottleneck'
            )

        if 'decoder' in normalized:
            normalized['decoder'] = normalize_config(
                normalized['decoder'], 'decoder'
            )

        # For hybrid architectures with additional components
        if is_hybrid and 'components' in normalized:
            components = normalized['components']
            if isinstance(components, dict):
                for comp_name, comp_config in components.items():
                    if isinstance(comp_config, dict) and 'type' in comp_config:
                        # Extract component type from the component type name
                        comp_type = comp_config['type'].split('_')[0].lower()
                        normalized['components'][comp_name] = normalize_config(
                            comp_config, comp_type
                        )

    # For individual components
    elif component_type is not None:
        schema = None

        if component_type == 'encoder':
            schema = create_encoder_schema()
        elif component_type == 'bottleneck':
            schema = create_bottleneck_schema()
        elif component_type == 'decoder':
            schema = create_decoder_schema()

        if schema:
            normalized = schema.normalize(normalized)

    return normalized
