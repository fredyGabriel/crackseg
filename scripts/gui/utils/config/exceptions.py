"""Configuration-related exceptions.

This module contains custom exception classes for configuration-related errors,
providing detailed error information for better debugging and user feedback.
"""


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""

    pass


class ValidationError(ConfigError):
    """Custom exception for configuration validation errors."""

    def __init__(
        self,
        message: str,
        line: int | None = None,
        column: int | None = None,
        field: str | None = None,
        suggestions: list[str] | None = None,
        is_critical: bool = True,
    ) -> None:
        """Initialize validation error with detailed information.

        Args:
            message: Error description.
            line: Line number where error occurred.
            column: Column number where error occurred.
            field: Configuration field name that caused the error.
            suggestions: List of suggested fixes.
            is_critical: Whether this error should prevent configuration usage.
        """
        self.message = message
        self.line = line
        self.column = column
        self.field = field
        self.suggestions = suggestions or []
        self.is_critical = is_critical

        # Build detailed error message
        details = []
        if line is not None:
            details.append(f"Line {line}")
        if column is not None:
            details.append(f"Column {column}")
        if field:
            details.append(f"Field '{field}'")

        location = ", ".join(details)
        full_message = f"{message}"
        if location:
            full_message += f" (at {location})"

        if self.suggestions:
            full_message += f"\nSuggestions: {', '.join(self.suggestions)}"

        super().__init__(full_message)
