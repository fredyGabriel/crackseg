"""Hydra override parsing and validation.

This module provides functionality for parsing and validating Hydra
configuration overrides from text input, ensuring security and correctness.
"""

from typing import Any

from ..parsing import AdvancedOverrideParser, OverrideParsingError


class HydraOverrideManager:
    """Manages Hydra override parsing and validation.

    Provides a high-level interface for parsing override text from UI input
    and validating individual override strings for security and correctness.

    Features:
    - Advanced text parsing with validation
    - Security-focused validation
    - Type checking capabilities
    - Error message collection

    Example:
        >>> manager = HydraOverrideManager()
        >>> valid, errors = manager.parse_overrides_text("trainer.epochs=100")
        >>> is_valid, error = manager.validate_single_override("model.lr=0.01")
    """

    def __init__(self) -> None:
        """Initialize the override manager."""
        self._parser = AdvancedOverrideParser()

    def parse_overrides_text(
        self, overrides_text: str, validate_types: bool = True
    ) -> tuple[list[str], list[str]]:
        """Parse override text and return valid overrides and errors.

        Args:
            overrides_text: Raw text containing Hydra overrides
            validate_types: Whether to perform type validation

        Returns:
            Tuple of (valid_overrides, error_messages)
        """
        try:
            self._parser.parse_overrides(overrides_text, validate_types)
            valid_overrides = self._parser.get_valid_overrides()
            errors = self._parser.get_parsing_errors()
            return valid_overrides, errors
        except OverrideParsingError as e:
            return [], [str(e)]

    def validate_single_override(
        self, override: str
    ) -> tuple[bool, str | None]:
        """Validate a single override string.

        Args:
            override: Single override string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self._parser.validate_override_string(override)

    def get_parsing_statistics(self) -> dict[str, Any]:
        """Get statistics from the last parsing operation.

        Returns:
            Dictionary with parsing statistics
        """
        return {
            "valid_count": len(self._parser.get_valid_overrides()),
            "error_count": len(self._parser.get_parsing_errors()),
            "total_processed": (
                len(self._parser.get_valid_overrides())
                + len(self._parser.get_parsing_errors())
            ),
        }
