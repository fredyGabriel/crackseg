"""
Core classes for configuration validation.

Provides the base classes and enumerations for validating model configurations:
- ParamType: Supported parameter types
- ConfigParam: Parameter definition with validation rules
- ConfigSchema: Schema for validating component configurations
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum


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
    choices: Optional[List[Any]] = None
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    nested_schema: Optional["ConfigSchema"] = None

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a parameter value according to its rules.

        Args:
            value: Value to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check if value is None and parameter is required
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            # Use default if value is None and not required
            return True, None

        # Type validation
        if self.param_type == ParamType.STRING and not isinstance(value, str):
            return False, f"Parameter '{self.name}' must be a string"
        elif self.param_type == ParamType.INTEGER and not isinstance(value,
                                                                     int):
            return False, f"Parameter '{self.name}' must be an integer"
        elif self.param_type == ParamType.FLOAT and not isinstance(
            value, (int, float)
        ):
            return False, f"Parameter '{self.name}' must be a number"
        elif self.param_type == ParamType.BOOLEAN and not isinstance(value,
                                                                     bool):
            return False, f"Parameter '{self.name}' must be a boolean"
        elif self.param_type == ParamType.LIST and not isinstance(value, list):
            return False, f"Parameter '{self.name}' must be a list"
        elif self.param_type == ParamType.DICT and not isinstance(value, dict):
            return False, f"Parameter '{self.name}' must be a dictionary"
        elif self.param_type == ParamType.NESTED:
            if not isinstance(value, dict):
                msg = f"Nested parameter '{self.name}' must be a dictionary"
                return False, msg
            if self.nested_schema:
                # Validate nested configuration
                is_valid, errors = self.nested_schema.validate(value)
                if not is_valid:
                    msg = f"Invalid nested config '{self.name}': {errors}"
                    return False, msg

        # Choices validation
        if self.choices and value not in self.choices:
            choices_str = ", ".join(str(c) for c in self.choices)
            msg = f"Parameter '{self.name}' must be one of: {choices_str}"
            return False, msg

        # Custom validator
        if self.validator and not self.validator(value):
            return False, f"Parameter '{self.name}' failed custom validation"

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
    params: List[ConfigParam] = field(default_factory=list)
    allow_unknown: bool = False

    def validate(
        self, config: Dict[str, Any]
    ) -> tuple[bool, Union[None, Dict[str, str]]]:
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
        param_names = set(p.name for p in self.params)

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
            if not is_valid:
                errors[param.name] = error

        return len(errors) == 0, errors if errors else None

    def normalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
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
                param.param_type == ParamType.NESTED and
                param.name in normalized and
                param.nested_schema
            ):
                nested_config = normalized[param.name]
                normalized[param.name] = param.nested_schema.normalize(
                    nested_config
                )

        return normalized
