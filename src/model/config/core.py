"""
Core classes for configuration validation.

Provides the base classes and enumerations for validating model configurations:
- ParamType: Supported parameter types
- ConfigParam: Parameter definition with validation rules
- ConfigSchema: Schema for validating component configurations
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Create logger
log = logging.getLogger(__name__)


class ParamType(Enum):
    """Supported parameter types for configuration validation."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    NESTED = "nested"  # For nested component configurations


@dataclass
class ConfigParam:
    """
    Definition of a configuration parameter with validation rules.

    Attributes:
        name: Parameter name
        param_type: Type of parameter
        required: Whether this parameter is required
        default: Default value if not provided
        choices: Allowed values for this parameter
        validator: Custom validation function
        description: Description of the parameter
        nested_schema: Schema for nested parameters (for NESTED type)
    """

    name: str
    param_type: ParamType
    required: bool = False
    default: Any = None
    choices: list[Any] | None = None
    validator: Callable[[Any], bool] | None = None
    description: str = ""
    nested_schema: "ConfigSchema" | None = None

    def _validate_value_type(self, value: Any) -> str | None:
        """Validates the type of the given value based on self.param_type."""
        error_message: str | None = None
        if self.param_type == ParamType.STRING and not isinstance(value, str):
            error_message = f"Parameter '{self.name}' must be a string"
        elif self.param_type == ParamType.INTEGER and not isinstance(
            value, int
        ):
            error_message = f"Parameter '{self.name}' must be an integer"
        elif self.param_type == ParamType.FLOAT and not isinstance(
            value, int | float
        ):
            error_message = f"Parameter '{self.name}' must be a number"
        elif self.param_type == ParamType.BOOLEAN and not isinstance(
            value, bool
        ):
            error_message = f"Parameter '{self.name}' must be a boolean"
        elif self.param_type == ParamType.LIST and not isinstance(value, list):
            error_message = f"Parameter '{self.name}' must be a list"
        elif self.param_type == ParamType.DICT and not isinstance(value, dict):
            error_message = f"Parameter '{self.name}' must be a dictionary"
        elif self.param_type == ParamType.NESTED:
            if not isinstance(value, dict):
                error_message = (
                    f"Nested parameter '{self.name}' must be a dictionary"
                )
            elif self.nested_schema:
                is_valid_nested, nested_errors = self.nested_schema.validate(
                    value
                )
                if not is_valid_nested:
                    error_message = (
                        f"Invalid nested config '{self.name}': {nested_errors}"
                    )
        return error_message

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """
        Validate a parameter value according to its rules.

        Args:
            value: Value to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        error_to_report: str | None = None

        # 1. Check if value is None
        if value is None:
            if self.required:
                error_to_report = f"Parameter '{self.name}' is required"
            else:
                # Not required and value is None, so it's valid.
                # No further checks are needed for a None value.
                return True, None

        # If value is not None, or if it was None but required
        # (error_to_report is set)
        # Proceed with other checks only if no error has been reported yet.
        if error_to_report is None:
            # 2. Type validation
            type_error = (
                self._validate_value_type(value)
                if self._validate_value_type is not None
                else (
                    None
                    if self._validate_value_type is not None
                    else (None, None)
                )
            )
            if type_error:
                error_to_report = type_error

        # 3. Choices validation (only if no error reported yet)
        if (
            error_to_report is None
            and self.choices
            and value not in self.choices
        ):
            choices_str = ", ".join(str(c) for c in self.choices)
            error_to_report = (
                f"Parameter '{self.name}' must be one of: {choices_str}"
            )

        # 4. Custom validator (only if no error reported yet)
        if (
            error_to_report is None
            and self.validator
            and not self.validator(value)
        ):
            error_to_report = (
                f"Parameter '{self.name}' failed custom validation"
            )

        # Final return based on whether an error was reported
        if error_to_report:
            return False, error_to_report

        return True, None


@dataclass
class ConfigSchema:
    """
    Schema for validating component configurations.

    Attributes:
        name: Schema name
        params: List of parameter definitions
        allow_unknown: Whether to allow parameters not in the schema
    """

    name: str
    params: list[ConfigParam] = field(default_factory=list)
    allow_unknown: bool = False

    def validate(
        self, config: dict[str, Any]
    ) -> tuple[bool, None | dict[str, str]]:
        """
        Validate a configuration against this schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            tuple: (is_valid, error_dict)
            Where error_dict maps parameter names to error messages
        """
        if not isinstance(config, dict):
            error_msg = "Configuration must be a dictionary, got "
            error_msg += f"{type(config)}"
            return False, {"_general": error_msg}

        errors = {}
        param_names = {p.name for p in self.params}

        # Check for unknown parameters
        if not self.allow_unknown:
            unknown_params = set(config.keys()) - param_names
            if unknown_params:
                unknown_str = ", ".join(unknown_params)
                errors["_unknown"] = f"Unknown parameters: {unknown_str}"

        # Validate each parameter
        for param in self.params:
            value = config.get(param.name)

            # Validate parameter
            is_valid, error = param.validate(value)
            if not is_valid and error is not None:
                errors[param.name] = error

        return len(errors) == 0, errors if errors else None

    def normalize(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize a configuration by filling in default values.

        Args:
            config: Configuration dictionary to normalize

        Returns:
            Dict: Normalized configuration
        """
        normalized = dict(config)

        # Fill in defaults for missing parameters
        for param in self.params:
            if param.name not in normalized and param.default is not None:
                normalized[param.name] = param.default

            # Normalize nested configurations
            if (
                param.param_type == ParamType.NESTED
                and param.name in normalized
                and param.nested_schema
            ):
                nested_config = normalized[param.name]
                normalized[param.name] = param.nested_schema.normalize(
                    nested_config
                )

        return normalized
