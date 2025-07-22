"""
Configuration validation package. This package provides comprehensive
YAML configuration validation with detailed error reporting and
suggestions for fixes. It includes syntax validation, structure
validation, type checking, and value constraints validation.
"""

import os

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ..exceptions import ValidationError
from .yaml_engine import YAMLValidator

# Global validator instance for backward compatibility
_yaml_validator = YAMLValidator()


def validate_yaml_syntax(content: str) -> tuple[bool, str | None]:
    """
    Validate YAML syntax without Hydra composition. Args: content: YAML
    content as a string. Returns: Tuple of (is_valid, error_message). If
    valid, error_message is None.
    """
    try:
        yaml.safe_load(content)
        return True, None
    except yaml.YAMLError as e:
        # Extract line number if available
        error_msg = str(e)
        mark = getattr(e, "problem_mark", None)
        problem = getattr(e, "problem", None)
        if mark is not None and problem is not None:
            error_msg = (
                f"Line {getattr(mark, 'line', 0) + 1}, "
                f"Column {getattr(mark, 'column', 0) + 1}: {problem}"
            )
        return False, error_msg
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_yaml_advanced(content: str) -> tuple[bool, list[ValidationError]]:
    """
    Perform advanced YAML validation with detailed error reporting. Args:
    content: YAML content as string. Returns: Tuple of (is_valid,
    list_of_validation_errors).
    """
    return _yaml_validator.comprehensive_validate(content)


def validate_config_structure(
    config: dict[str, object],
) -> tuple[bool, list[ValidationError]]:
    """
    Validate configuration structure and schema. Args: config: Parsed
    configuration dictionary. Returns: Tuple of (is_valid,
    list_of_validation_errors).
    """
    return _yaml_validator.validate_structure(config)


def validate_config_types(
    config: dict[str, object],
) -> tuple[bool, list[ValidationError]]:
    """
    Validate configuration data types. Args: config: Parsed configuration
    dictionary. Returns: Tuple of (is_valid, list_of_validation_errors).
    """
    return _yaml_validator.validate_types(config)


def validate_config_values(
    config: dict[str, object],
) -> tuple[bool, list[ValidationError]]:
    """
    Validate configuration values and constraints. Args: config: Parsed
    configuration dictionary. Returns: Tuple of (is_valid,
    list_of_validation_errors).
    """
    return _yaml_validator.validate_values(config)


def validate_with_hydra(
    config_path: str, config_name: str
) -> tuple[bool, str | None]:
    """
    Validate configuration using Hydra composition. Args: config_path:
    Path to the configuration directory. config_name: Name of the
    configuration file (without .yaml extension). Returns: Tuple of
    (is_valid, error_message). If valid, error_message is None.
    """
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    try:
        # Initialize Hydra with the config path
        with initialize_config_dir(
            config_dir=os.path.abspath(config_path), version_base="1.3"
        ):
            # Try to compose the configuration
            compose(config_name=config_name)

            # If we get here, the configuration is valid
            return True, None

    except Exception as e:
        error_msg = str(e)

        # Try to extract more specific error information
        if "Could not find" in error_msg:
            return False, f"Configuration not found: {error_msg}"
        elif "Error merging" in error_msg:
            return False, f"Configuration merge error: {error_msg}"
        elif "Missing mandatory value" in error_msg:
            return False, f"Missing required value: {error_msg}"
        else:
            return False, f"Hydra validation error: {error_msg}"

    finally:
        # Clean up Hydra instance
        GlobalHydra.instance().clear()


# Export the ValidationError for convenience
__all__ = [
    "ValidationError",
    "validate_yaml_syntax",
    "validate_yaml_advanced",
    "validate_config_structure",
    "validate_config_types",
    "validate_config_values",
    "validate_with_hydra",
    "_yaml_validator",
]
