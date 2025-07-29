"""
Utility functions for CrackSeg schema validation.

This module provides helper functions and convenience methods for
schema validation operations.
"""

from typing import Any

from ..exceptions import ValidationError
from .core_validator import CrackSegSchemaValidator


def validate_crackseg_schema(
    config: dict[str, Any],
) -> tuple[bool, list[ValidationError], list[str]]:
    """
    Convenience function to validate CrackSeg configuration schema.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = CrackSegSchemaValidator()
    return validator.validate_complete_schema(config)


def validate_model_config(
    config: dict[str, Any],
) -> tuple[bool, list[ValidationError], list[str]]:
    """
    Validate only the model section of configuration.

    Args:
        config: Configuration dictionary containing model section

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = CrackSegSchemaValidator()
    return validator.validate_model_section(config)


def validate_training_config(
    config: dict[str, Any], full_config: dict[str, Any] | None = None
) -> tuple[bool, list[ValidationError], list[str]]:
    """
    Validate only the training section of configuration.

    Args:
        config: Configuration dictionary containing training section
        full_config: Complete configuration for cross-validation

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = CrackSegSchemaValidator()
    return validator.validate_training_section(config, full_config)


def validate_data_config(
    config: dict[str, Any],
) -> tuple[bool, list[ValidationError], list[str]]:
    """
    Validate only the data section of configuration.

    Args:
        config: Configuration dictionary containing data section

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = CrackSegSchemaValidator()
    return validator.validate_data_section(config)


def format_validation_errors(errors: list[ValidationError]) -> str:
    """
    Format validation errors into a readable string.

    Args:
        errors: List of validation errors

    Returns:
        Formatted error string
    """
    if not errors:
        return "No validation errors found."

    lines = ["Validation Errors:"]
    for i, error in enumerate(errors, 1):
        field = error.field or "unknown"
        value = error.value if error.value is not None else "None"
        lines.append(f"{i}. {field}: {error.message} (value: {value})")

    return "\n".join(lines)


def format_validation_warnings(warnings: list[str]) -> str:
    """
    Format validation warnings into a readable string.

    Args:
        warnings: List of warning messages

    Returns:
        Formatted warning string
    """
    if not warnings:
        return "No validation warnings found."

    lines = ["Validation Warnings:"]
    for i, warning in enumerate(warnings, 1):
        lines.append(f"{i}. {warning}")

    return "\n".join(lines)


def get_validation_summary(
    is_valid: bool, errors: list[ValidationError], warnings: list[str]
) -> dict[str, Any]:
    """
    Create a summary of validation results.

    Args:
        is_valid: Whether the configuration is valid
        errors: List of validation errors
        warnings: List of validation warnings

    Returns:
        Summary dictionary
    """
    return {
        "is_valid": is_valid,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": [
            {"field": e.field, "message": e.message, "value": e.value}
            for e in errors
        ],
        "warnings": warnings,
        "summary": f"Configuration is {'valid' if is_valid else 'invalid'} "
        f"({len(errors)} errors, {len(warnings)} warnings)",
    }


def validate_required_fields(
    config: dict[str, Any], required_fields: list[str]
) -> list[ValidationError]:
    """
    Validate that required fields are present in configuration.

    Args:
        config: Configuration dictionary
        required_fields: List of required field names

    Returns:
        List of validation errors for missing fields
    """
    errors: list[ValidationError] = []

    for field in required_fields:
        if field not in config or config[field] is None:
            errors.append(
                ValidationError(
                    f"Missing required field: {field}",
                    field=field,
                )
            )

    return errors


def validate_field_types(
    config: dict[str, Any], field_types: dict[str, type]
) -> list[ValidationError]:
    """
    Validate that fields have the correct types.

    Args:
        config: Configuration dictionary
        field_types: Dictionary mapping field names to expected types

    Returns:
        List of validation errors for type mismatches
    """
    errors: list[ValidationError] = []

    for field, expected_type in field_types.items():
        if field in config:
            value = config[field]
            if not isinstance(value, expected_type):
                errors.append(
                    ValidationError(
                        f"Field {field} must be of type "
                        f"{expected_type.__name__}, got "
                        f"{type(value).__name__}",
                        field=field,
                    )
                )

    return errors


def validate_nested_structure(
    config: dict[str, Any], structure: dict[str, Any]
) -> list[ValidationError]:
    """
    Validate nested configuration structure.

    Args:
        config: Configuration dictionary
        structure: Expected structure with field names and types

    Returns:
        List of validation errors
    """
    errors: list[ValidationError] = []

    for field, field_spec in structure.items():
        if field not in config:
            if field_spec.get("required", False):
                errors.append(
                    ValidationError(
                        f"Missing required field: {field}",
                        field=field,
                    )
                )
            continue

        value = config[field]
        expected_type = field_spec.get("type")

        if expected_type and not isinstance(value, expected_type):
            errors.append(
                ValidationError(
                    f"Field {field} must be of type "
                    f"{expected_type.__name__}, got "
                    f"{type(value).__name__}",
                    field=field,
                )
            )

        # Validate nested structure if specified
        if "nested" in field_spec and isinstance(value, dict):
            nested_errors = validate_nested_structure(
                value, field_spec["nested"]
            )
            for error in nested_errors:
                error.field = f"{field}.{error.field}"
            errors.extend(nested_errors)

    return errors
