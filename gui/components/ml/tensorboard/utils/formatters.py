"""
Formatting utilities for TensorBoard component. This module provides
functions for formatting various data types used in the TensorBoard
component, such as uptime, error messages, and sizes.
"""

from typing import Any


def format_uptime(uptime_seconds: float) -> str:
    """Format uptime in human-readable form.

    Args:
        uptime_seconds: Uptime duration in seconds.

    Returns:
        Formatted uptime string (e.g., "45s", "2.5m", "1.2h").

    Example:
        >>> format_uptime(45.0)
        "45s"
        >>> format_uptime(150.0)
        "2.5m"
    """
    if uptime_seconds < 60:
        return f"{uptime_seconds:.0f}s"
    elif uptime_seconds < 3600:
        return f"{uptime_seconds / 60:.1f}m"
    else:
        return f"{uptime_seconds / 3600:.1f}h"


def format_error_message(
    error_message: str | None,
    error_type: str | None = None,
    max_length: int = 200,
) -> str:
    """Format error message for display with optional truncation.

    Args:
        error_message: Raw error message.
        error_type: Optional error type for categorization.
        max_length: Maximum message length before truncation.

    Returns:
        Formatted error message with type prefix if available.

    Example:
        >>> format_error_message("Port already in use", "PortConflictError")
        "PortConflictError: Port already in use"
    """
    if not error_message:
        return "Unknown error occurred"

    # Add error type prefix if available
    if error_type:
        formatted = f"{error_type}: {error_message}"
    else:
        formatted = error_message

    # Truncate if too long
    if len(formatted) > max_length:
        formatted = formatted[: max_length - 3] + "..."

    return formatted


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted size string (e.g., "1.5 MB", "234 KB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_progress_percentage(progress: float) -> str:
    """Format progress as percentage string.

    Args:
        progress: Progress value between 0.0 and 1.0.

    Returns:
        Formatted percentage string.

    Example:
        >>> format_progress_percentage(0.75)
        "75%"
    """
    return f"{progress * 100:.0f}%"


def format_metric_value(value: Any, metric_type: str = "default") -> str:
    """
    Format metric values for consistent display. Args: value: The metric
    value to format. metric_type: Type of metric for specialized
    formatting. Returns: Formatted metric string.
    """
    if value is None:
        return "N/A"

    if metric_type == "percentage":
        return f"{float(value):.1f}%"
    elif metric_type == "time":
        return format_uptime(float(value))
    elif metric_type == "count":
        return f"{int(value):,}"
    elif isinstance(value, float):
        return f"{value:.2f}"
    else:
        return str(value)
