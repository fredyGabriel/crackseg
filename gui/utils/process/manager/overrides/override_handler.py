"""Override parsing and validation for training configurations.

This module handles parsing, validation, and command building for Hydra
configuration overrides.
"""

from pathlib import Path

from ..parsing import AdvancedOverrideParser, OverrideParsingError


class OverrideHandler:
    """Handle parsing and validation of configuration overrides."""

    def __init__(self) -> None:
        """Initialize the override handler."""
        self._override_parser = AdvancedOverrideParser()

    def validate_overrides(self, overrides: list[str]) -> list[str]:
        """Validate a list of overrides.

        Args:
            overrides: List of override strings to validate

        Returns:
            List of validated overrides

        Raises:
            OverrideParsingError: If any override is invalid
        """
        validated_overrides = []
        for override in overrides:
            is_valid, error = self.validate_single_override(override)
            if not is_valid:
                raise OverrideParsingError(
                    f"Invalid override '{override}': {error}"
                )
            validated_overrides.append(override)
        return validated_overrides

    def validate_single_override(
        self, override: str
    ) -> tuple[bool, str | None]:
        """Validate a single override string.

        Args:
            override: Override string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self._override_parser.parse_override(override)
            return True, None
        except Exception as e:
            return False, str(e)

    def build_command(
        self,
        config_path: Path,
        config_name: str,
        overrides: list[str] | None = None,
    ) -> list[str]:
        """Build the command for training execution.

        Args:
            config_path: Path to Hydra configuration directory
            config_name: Name of the configuration to use
            overrides: List of configuration overrides

        Returns:
            List of command arguments
        """
        command = ["python", "src/main.py"]

        # Add config path and name
        command.extend(["--config-dir", str(config_path)])
        command.extend(["--config-name", config_name])

        # Add overrides
        if overrides:
            for override in overrides:
                command.extend(["--override", override])

        return command

    def parse_overrides_text(
        self, overrides_text: str, validate_types: bool = True
    ) -> tuple[list[str], list[str]]:
        """Parse overrides from text input.

        Args:
            overrides_text: Text containing overrides
            validate_types: Whether to validate types

        Returns:
            Tuple of (valid_overrides, invalid_overrides)
        """
        return self._override_parser.parse_overrides_text(
            overrides_text, validate_types
        )
