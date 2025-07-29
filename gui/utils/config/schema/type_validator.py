"""
Type validation for CrackSeg configuration schemas.

This module provides type-specific validation logic for configuration
parameters, ensuring proper data types and value ranges.
"""

from typing import Any

from ..exceptions import ValidationError


class TypeValidator:
    """Type-specific validator for configuration parameters."""

    @staticmethod
    def validate_integer(
        value: Any,
        field: str,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> list[ValidationError]:
        """Validate integer values with optional range constraints."""
        errors: list[ValidationError] = []

        if not isinstance(value, int):
            errors.append(
                ValidationError(
                    f"{field} must be an integer",
                    field=field,
                )
            )
        else:
            if min_value is not None and value < min_value:
                errors.append(
                    ValidationError(
                        f"{field} must be >= {min_value}",
                        field=field,
                    )
                )
            if max_value is not None and value > max_value:
                errors.append(
                    ValidationError(
                        f"{field} must be <= {max_value}",
                        field=field,
                    )
                )

        return errors

    @staticmethod
    def validate_float(
        value: Any,
        field: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> list[ValidationError]:
        """Validate float values with optional range constraints."""
        errors: list[ValidationError] = []

        if not isinstance(value, int | float):
            errors.append(
                ValidationError(
                    f"{field} must be a number",
                    field=field,
                )
            )
        else:
            if min_value is not None and value < min_value:
                errors.append(
                    ValidationError(
                        f"{field} must be >= {min_value}",
                        field=field,
                    )
                )
            if max_value is not None and value > max_value:
                errors.append(
                    ValidationError(
                        f"{field} must be <= {max_value}",
                        field=field,
                    )
                )

        return errors

    @staticmethod
    def validate_string(
        value: Any,
        field: str,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> list[ValidationError]:
        """Validate string values with optional length constraints."""
        errors: list[ValidationError] = []

        if not isinstance(value, str):
            errors.append(
                ValidationError(
                    f"{field} must be a string",
                    field=field,
                )
            )
        else:
            if min_length is not None and len(value) < min_length:
                errors.append(
                    ValidationError(
                        f"{field} must be at least {min_length} characters",
                        field=field,
                    )
                )
            if max_length is not None and len(value) > max_length:
                errors.append(
                    ValidationError(
                        f"{field} must be at most {max_length} characters",
                        field=field,
                    )
                )

        return errors

    @staticmethod
    def validate_boolean(value: Any, field: str) -> list[ValidationError]:
        """Validate boolean values."""
        errors: list[ValidationError] = []

        if not isinstance(value, bool):
            errors.append(
                ValidationError(
                    f"{field} must be a boolean",
                    field=field,
                )
            )

        return errors

    @staticmethod
    def validate_list(
        value: Any,
        field: str,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> list[ValidationError]:
        """Validate list values with optional length constraints."""
        errors: list[ValidationError] = []

        if not isinstance(value, list):
            errors.append(
                ValidationError(
                    f"{field} must be a list",
                    field=field,
                )
            )
        else:
            if min_length is not None and len(value) < min_length:
                errors.append(
                    ValidationError(
                        f"{field} must have at least {min_length} items",
                        field=field,
                    )
                )
            if max_length is not None and len(value) > max_length:
                errors.append(
                    ValidationError(
                        f"{field} must have at most {max_length} items",
                        field=field,
                    )
                )

        return errors

    @staticmethod
    def validate_dict(
        value: Any, field: str, required_keys: set[str] | None = None
    ) -> list[ValidationError]:
        """Validate dictionary values with optional required keys."""
        errors: list[ValidationError] = []

        if not isinstance(value, dict):
            errors.append(
                ValidationError(
                    f"{field} must be a dictionary",
                    field=field,
                )
            )
        elif required_keys is not None:
            missing_keys = required_keys - set(value.keys())
            for key in missing_keys:
                errors.append(
                    ValidationError(
                        f"{field} missing required key: {key}",
                        field=f"{field}.{key}",
                    )
                )

        return errors

    @staticmethod
    def validate_enum(
        value: Any, field: str, valid_values: set[str]
    ) -> list[ValidationError]:
        """Validate enum values against a set of valid options."""
        errors: list[ValidationError] = []

        if not isinstance(value, str):
            errors.append(
                ValidationError(
                    f"{field} must be a string",
                    field=field,
                )
            )
        elif value not in valid_values:
            errors.append(
                ValidationError(
                    f"Invalid {field}: {value}. "
                    f"Valid options: {sorted(valid_values)}",
                    field=field,
                )
            )

        return errors
