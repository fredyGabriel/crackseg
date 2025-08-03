"""Validation utilities for TensorBoard component.

This module provides validation functions for TensorBoard component inputs,
configurations, and state management.
"""

from pathlib import Path
from typing import Any


class ValidationError(Exception):
    """Raised when validation fails in TensorBoard component."""

    pass


def validate_log_directory(log_dir: Path | None) -> tuple[bool, str | None]:
    """Validate log directory for TensorBoard usage.

    Args:
        log_dir: Path to the log directory to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.

    Example:
        >>> valid, error = validate_log_directory(Path("logs/tensorboard"))
        >>> if not valid:
        ...     print(f"Validation failed: {error}")
    """
    if log_dir is None:
        return False, "Log directory cannot be None"

    # Convert to Path if needed (for runtime validation of Any type inputs)
    try:
        if not hasattr(log_dir, "exists"):  # Not a Path-like object
            log_dir = Path(log_dir)
    except (TypeError, ValueError) as e:
        return False, f"Invalid path format: {e}"

    # Check if path exists
    if not log_dir.exists():
        return False, f"Log directory does not exist: {log_dir}"

    # Check if it's a directory
    if not log_dir.is_dir():
        return False, f"Path is not a directory: {log_dir}"

    # Check read permissions
    try:
        list(log_dir.iterdir())
    except PermissionError:
        return False, f"No read permission for directory: {log_dir}"

    return True, None


def validate_component_config(
    config: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate TensorBoard component configuration.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    # Runtime validation - config could be Any at runtime despite annotation
    if not hasattr(config, "get") or not hasattr(config, "keys"):
        return False, "Configuration must be a dictionary"

    # Validate required fields
    required_fields = ["default_height", "auto_startup", "show_controls"]
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"

    # Validate height
    height = config.get("default_height")
    if not isinstance(height, int) or height <= 0:
        return False, "default_height must be a positive integer"

    # Validate boolean fields
    bool_fields = ["auto_startup", "show_controls", "show_status"]
    for field in bool_fields:
        if field in config and not isinstance(config[field], bool):
            return False, f"{field} must be a boolean value"

    # Validate timeout values
    timeout_fields = ["startup_timeout", "health_check_interval"]
    for field in timeout_fields:
        if field in config:
            value = config[field]
            if not isinstance(value, int | float) or value <= 0:
                return False, f"{field} must be a positive number"

    return True, None


def validate_iframe_dimensions(
    height: int | None, width: int | None
) -> tuple[bool, str | None]:
    """Validate iframe dimensions for embedding.

    Args:
        height: Iframe height in pixels.
        width: Iframe width in pixels.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if height is not None:
        # Runtime validation for type safety
        if not hasattr(height, "__index__") or height <= 0:
            return False, "Height must be a positive integer"
        if height < 100:
            return False, "Height must be at least 100 pixels"
        if height > 2000:
            return False, "Height must not exceed 2000 pixels"

    if width is not None:
        # Runtime validation for type safety
        if not hasattr(width, "__index__") or width <= 0:
            return False, "Width must be a positive integer"
        if width < 200:
            return False, "Width must be at least 200 pixels"
        if width > 3000:
            return False, "Width must not exceed 3000 pixels"

    return True, None


def validate_session_state_keys(
    state: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate session state structure for TensorBoard component.

    Args:
        state: Session state dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    # Runtime validation - state could be Any at runtime despite annotation
    if not hasattr(state, "get") or not hasattr(state, "keys"):
        return False, "Session state must be a dictionary"

    # Expected fields with their types
    expected_fields = {
        "last_log_dir": (type(None), Path),
        "startup_attempted": bool,
        "last_status_check": (int, float),
        "error_message": (type(None), str),
        "startup_attempts": int,
        "startup_progress": (int, float),
    }

    for field, expected_types in expected_fields.items():
        if field in state:
            if not isinstance(state[field], expected_types):
                return False, f"Field {field} has invalid type"

    # Validate ranges
    if "startup_progress" in state:
        progress = state["startup_progress"]
        if not (0.0 <= progress <= 1.0):
            return False, "startup_progress must be between 0.0 and 1.0"

    if "startup_attempts" in state:
        attempts = state["startup_attempts"]
        if attempts < 0:
            return False, "startup_attempts must be non-negative"

    return True, None


def validate_error_info(error_info: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate error information structure.

    Args:
        error_info: Error information dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    # Runtime validation - error_info could be Any at runtime
    if not hasattr(error_info, "get") or not hasattr(error_info, "keys"):
        return False, "Error info must be a dictionary"

    # Optional but typed fields
    typed_fields = {
        "error_message": (type(None), str),
        "error_type": (type(None), str),
        "startup_attempts": int,
        "recovery_attempted": bool,
    }

    for field, expected_types in typed_fields.items():
        if field in error_info:
            if not isinstance(error_info[field], expected_types):
                return False, f"Field {field} has invalid type"

    return True, None
