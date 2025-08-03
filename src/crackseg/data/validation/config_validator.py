"""General configuration validation utilities.

This module provides general configuration validation utilities for the crack
segmentation pipeline. It includes base classes and utilities for validating
various types of configurations.

Key Features:
    - Base configuration validator class
    - General validation utilities
    - Configuration format normalization
    - Error handling and reporting utilities

Core Components:
    - ConfigValidator: Base class for configuration validation
    - Validation utilities for common configuration patterns
    - Error reporting and formatting utilities

Common Usage:
    # Create custom validator
    class MyConfigValidator(ConfigValidator):
        def validate(self, config):
            # Custom validation logic
            pass

    # Use general validation utilities
    validate_required_keys(config, ["key1", "key2"])
    validate_numeric_range(value, 0.0, 1.0, "parameter_name")

Integration:
    - Used by specific validators (data, transform)
    - Provides base functionality for custom validators
    - Compatible with Hydra configuration system
    - Supports various configuration formats

Error Handling:
    - Standardized error message formatting
    - Context-aware error reporting
    - Clear guidance for configuration correction
    - Type-specific validation with appropriate error types

References:
    - Data validation: src.data.validation.data_validator
    - Transform validation: src.data.validation.transform_validator
    - Configuration: configs/
"""

from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig


class ConfigValidator:
    """Base class for configuration validation.

    Provides common validation functionality and error handling for
    configuration validation across the crack segmentation pipeline.

    Attributes:
        name: Name of the validator for error reporting.
        required_keys: List of required configuration keys.
        optional_keys: List of optional configuration keys.

    Methods:
        validate: Main validation method to be implemented by subclasses.
        validate_required_keys: Validate presence of required keys.
        validate_optional_keys: Validate optional keys if present.
        format_error: Format error messages with context.
    """

    def __init__(self, name: str = "ConfigValidator") -> None:
        """Initialize the configuration validator.

        Args:
            name: Name of the validator for error reporting.
        """
        self.name = name
        self.required_keys: list[str] = []
        self.optional_keys: list[str] = []

    def validate(self, config: dict[str, Any] | DictConfig) -> None:
        """Validate configuration.

        Args:
            config: Configuration to validate.

        Raises:
            ValueError: If configuration is invalid.
        """
        raise NotImplementedError(
            f"{self.name}.validate() must be implemented by subclasses"
        )

    def validate_required_keys(
        self, config: dict[str, Any] | DictConfig, keys: Sequence[str]
    ) -> None:
        """Validate that required keys are present in configuration.

        Args:
            config: Configuration to validate.
            keys: Required keys to check for.

        Raises:
            ValueError: If any required key is missing.
        """
        missing_keys = []
        for key in keys:
            if key not in config:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(
                f"{self.name}: Missing required keys: {', '.join(missing_keys)}"
            )

    def validate_optional_keys(
        self, config: dict[str, Any] | DictConfig, keys: Sequence[str]
    ) -> None:
        """Validate optional keys if present in configuration.

        Args:
            config: Configuration to validate.
            keys: Optional keys to validate if present.
        """
        for key in keys:
            if key in config:
                # Subclasses can override this method to add specific validation
                self._validate_optional_key(config, key)

    def _validate_optional_key(
        self, config: dict[str, Any] | DictConfig, key: str
    ) -> None:
        """Validate a specific optional key.

        Args:
            config: Configuration containing the key.
            key: Key to validate.

        Raises:
            ValueError: If the key value is invalid.
        """
        # Default implementation does nothing
        # Subclasses can override for specific validation
        pass

    def format_error(
        self, message: str, context: dict[str, Any] | None = None
    ) -> str:
        """Format error message with context.

        Args:
            message: Base error message.
            context: Additional context information.

        Returns:
            Formatted error message.
        """
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            return f"{self.name}: {message} (Context: {context_str})"
        return f"{self.name}: {message}"


def validate_required_keys(
    config: dict[str, Any] | DictConfig, keys: Sequence[str]
) -> None:
    """Validate that required keys are present in configuration.

    Args:
        config: Configuration to validate.
        keys: Required keys to check for.

    Raises:
        ValueError: If any required key is missing.
    """
    missing_keys = []
    for key in keys:
        if key not in config:
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")


def validate_numeric_range(
    value: float, min_val: float, max_val: float, param_name: str
) -> None:
    """Validate that a numeric value is within a specified range.

    Args:
        value: Value to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        param_name: Name of the parameter for error reporting.

    Raises:
        ValueError: If value is outside the specified range.
    """
    if not isinstance(value, int | float):
        raise ValueError(
            f"{param_name} must be a number, got {type(value).__name__}"
        )

    if not min_val <= value <= max_val:
        raise ValueError(
            f"{param_name} must be between {min_val} and {max_val}, got {value}"
        )


def validate_list_length(
    value: list[Any] | tuple[Any, ...], expected_length: int, param_name: str
) -> None:
    """Validate that a list has the expected length.

    Args:
        value: List to validate.
        expected_length: Expected length of the list.
        param_name: Name of the parameter for error reporting.

    Raises:
        ValueError: If list length doesn't match expected length.
    """
    if not isinstance(value, list | tuple):
        raise ValueError(
            f"{param_name} must be a list or tuple, got {type(value).__name__}"
        )

    if len(value) != expected_length:
        raise ValueError(
            f"{param_name} must have length {expected_length}, got {len(value)}"
        )


def validate_positive_number(value: float, param_name: str) -> None:
    """Validate that a number is positive.

    Args:
        value: Value to validate.
        param_name: Name of the parameter for error reporting.

    Raises:
        ValueError: If value is not positive.
    """
    if not isinstance(value, int | float):
        raise ValueError(
            f"{param_name} must be a number, got {type(value).__name__}"
        )

    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")
