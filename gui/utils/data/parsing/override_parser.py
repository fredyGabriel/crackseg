"""Advanced Hydra override parsing with security validation.

This module provides comprehensive parsing and validation of Hydra
configuration overrides with security features, type checking,
and support for complex argument structures.
"""

import re
import shlex
from dataclasses import dataclass
from typing import Any

from .exceptions import OverrideParsingError


@dataclass
class ParsedOverride:
    """Represents a parsed and validated Hydra override.

    Attributes:
        key: Configuration key (e.g., 'trainer.max_epochs')
        value: Processed value after normalization
        raw_value: Original value string as provided
        override_type: Type of override
            ('config', 'package', 'delete', 'force')
        is_valid: Whether the override passed validation
        error_message: Validation error if any
    """

    key: str
    value: Any
    raw_value: str
    override_type: str  # 'config', 'package', 'delete', 'force'
    is_valid: bool = True
    error_message: str | None = None


class AdvancedOverrideParser:
    """Advanced parser for Hydra configuration overrides.

    Supports complex argument structures, nested configurations,
    and comprehensive validation using shlex for safe parsing.

    Features:
    - Security validation against dangerous patterns
    - Type checking for common configuration values
    - Support for all Hydra override types
    - Safe command-line style parsing with shlex

    Example:
        >>> parser = AdvancedOverrideParser()
        >>> overrides = parser.parse_overrides(
        ...     "trainer.max_epochs=100 model.encoder=resnet50"
        ... )
        >>> valid_list = parser.get_valid_overrides()
    """

    # Regex patterns for Hydra override validation
    HYDRA_KEY_PATTERN = re.compile(
        r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
    )
    HYDRA_PACKAGE_PATTERN = re.compile(
        r"^\+[a-zA-Z_][a-zA-Z0-9_/]*(/[a-zA-Z_][a-zA-Z0-9_]*)*=.*$"
    )
    HYDRA_DELETE_PATTERN = re.compile(
        r"^~[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
    )
    HYDRA_FORCE_PATTERN = re.compile(
        r"^\+\+[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*=.*$"
    )

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        re.compile(r"__import__"),
        re.compile(r"eval\s*\("),
        re.compile(r"exec\s*\("),
        re.compile(r"subprocess"),
        re.compile(r"os\.system"),
        re.compile(r"shell=True"),
        re.compile(r"[;&|`$]"),  # Shell metacharacters
        re.compile(r"\.\.\/"),  # Path traversal
        re.compile(r"~\/\."),  # Hidden files
    ]

    def __init__(self) -> None:
        """Initialize the advanced override parser."""
        self._parsed_overrides: list[ParsedOverride] = []

    def parse_overrides(
        self, overrides_text: str, validate_types: bool = True
    ) -> list[ParsedOverride]:
        """Parse complex override string into validated override objects.

        Args:
            overrides_text: Raw text containing Hydra overrides
            validate_types: Whether to perform type validation

        Returns:
            List of parsed and validated override objects

        Raises:
            OverrideParsingError: If parsing fails critically
        """
        self._parsed_overrides = []

        if not overrides_text.strip():
            return self._parsed_overrides

        try:
            # Use shlex for safe parsing of command-like strings
            raw_overrides = self._safe_split_overrides(overrides_text)

            for raw_override in raw_overrides:
                parsed = self._parse_single_override(
                    raw_override, validate_types
                )
                self._parsed_overrides.append(parsed)

        except Exception as e:
            raise OverrideParsingError(
                f"Failed to parse overrides: {e}"
            ) from e

        return self._parsed_overrides

    def get_valid_overrides(self) -> list[str]:
        """Get list of valid override strings for subprocess execution.

        Returns:
            List of validated override strings ready for command line
        """
        return [
            override.raw_value
            for override in self._parsed_overrides
            if override.is_valid
        ]

    def get_parsing_errors(self) -> list[str]:
        """Get list of parsing errors for user feedback.

        Returns:
            List of error messages for invalid overrides
        """
        return [
            f"{override.raw_value}: {override.error_message}"
            for override in self._parsed_overrides
            if not override.is_valid and override.error_message
        ]

    def validate_override_string(
        self, override: str
    ) -> tuple[bool, str | None]:
        """Validate a single override string quickly.

        Args:
            override: Single override string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = self._parse_single_override(override, validate_types=True)
            return parsed.is_valid, parsed.error_message
        except Exception as e:
            return False, str(e)

    def _safe_split_overrides(self, overrides_text: str) -> list[str]:
        """Safely split override text using shlex.

        Args:
            overrides_text: Raw override text

        Returns:
            List of individual override strings
        """
        try:
            # First, check for dangerous patterns
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern.search(overrides_text):
                    raise OverrideParsingError(
                        f"Dangerous pattern detected: {pattern.pattern}"
                    )

            # Use shlex for safe splitting (handles quotes, escapes)
            overrides = shlex.split(overrides_text)

            # Filter out empty strings
            return [
                override.strip() for override in overrides if override.strip()
            ]

        except ValueError as e:
            # shlex.split can raise ValueError for unclosed quotes
            raise OverrideParsingError(f"Invalid quote structure: {e}") from e

    def _parse_single_override(
        self, override: str, validate_types: bool
    ) -> ParsedOverride:
        """Parse and validate a single override string.

        Args:
            override: Single override string
            validate_types: Whether to validate value types

        Returns:
            ParsedOverride object with validation results
        """
        override = override.strip()

        # Determine override type and validate format
        override_type, is_format_valid, format_error = self._classify_override(
            override
        )

        if not is_format_valid:
            return ParsedOverride(
                key="",
                value="",
                raw_value=override,
                override_type=override_type,
                is_valid=False,
                error_message=format_error,
            )

        # Extract key and value
        key, value, raw_value = self._extract_key_value(
            override, override_type
        )

        # Validate key format
        if not self._is_valid_key(key, override_type):
            return ParsedOverride(
                key=key,
                value=value,
                raw_value=override,
                override_type=override_type,
                is_valid=False,
                error_message=f"Invalid key format: {key}",
            )

        # Type validation if requested
        type_error = None
        if validate_types and override_type == "config":
            type_error = self._validate_value_type(key, raw_value)

        return ParsedOverride(
            key=key,
            value=value,
            raw_value=override,
            override_type=override_type,
            is_valid=type_error is None,
            error_message=type_error,
        )

    def _classify_override(
        self, override: str
    ) -> tuple[str, bool, str | None]:
        """Classify the type of Hydra override and validate format.

        Args:
            override: Override string to classify

        Returns:
            Tuple of (type, is_valid, error_message)
        """
        if not override:
            return "unknown", False, "Empty override"

        # Package override: +key=value
        if self.HYDRA_PACKAGE_PATTERN.match(override):
            return "package", True, None

        # Force override: ++key=value
        if self.HYDRA_FORCE_PATTERN.match(override):
            return "force", True, None

        # Delete override: ~key
        if self.HYDRA_DELETE_PATTERN.match(override):
            return "delete", True, None

        # Regular config override: key=value
        if "=" in override and not override.startswith(("+", "~")):
            key, _, value = override.partition("=")
            if self.HYDRA_KEY_PATTERN.match(key.strip()):
                return "config", True, None
            else:
                return "config", False, f"Invalid key format: {key.strip()}"

        return "unknown", False, f"Unrecognized override format: {override}"

    def _extract_key_value(
        self, override: str, override_type: str
    ) -> tuple[str, str, str]:
        """Extract key and value from override string.

        Args:
            override: Override string
            override_type: Type of override

        Returns:
            Tuple of (key, processed_value, raw_value)
        """
        if override_type == "delete":
            # Delete overrides: ~key
            key = override[1:].strip()
            return key, "", ""

        elif override_type in ("package", "force"):
            # Package/Force overrides: +key=value or ++key=value
            prefix_len = 2 if override_type == "force" else 1
            key_value = override[prefix_len:]
            key, _, raw_value = key_value.partition("=")
            return key.strip(), self._process_value(raw_value), raw_value

        else:
            # Regular config overrides: key=value
            key, _, raw_value = override.partition("=")
            return key.strip(), self._process_value(raw_value), raw_value

    def _process_value(self, raw_value: str) -> str:
        """Process and normalize a configuration value.

        Args:
            raw_value: Raw value string

        Returns:
            Processed value string
        """
        value = raw_value.strip()

        # Handle quoted strings
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]

        return value

    def _is_valid_key(self, key: str, override_type: str) -> bool:
        """Validate if a key has proper format for its override type.

        Args:
            key: Configuration key
            override_type: Type of override

        Returns:
            True if key format is valid
        """
        if not key:
            return False

        # Package overrides can contain slashes
        if override_type == "package":
            # Allow pattern like "model/encoder" or "model/encoder.attention"
            package_key_pattern = re.compile(
                r"^[a-zA-Z_][a-zA-Z0-9_/]*(/[a-zA-Z_][a-zA-Z0-9_]*)*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
            )
            return bool(package_key_pattern.match(key))

        # All other override types use the standard key pattern
        return bool(self.HYDRA_KEY_PATTERN.match(key))

    def _validate_value_type(self, key: str, value: str) -> str | None:
        """Validate the type and format of a configuration value.

        Args:
            key: Configuration key
            value: Value to validate

        Returns:
            Error message if invalid, None if valid
        """
        if not value:
            return None  # Empty values are allowed

        # Check for potentially dangerous values
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(value):
                return f"Potentially dangerous value: {value}"

        # Validate common configuration patterns
        try:
            # Check if it's a list: [item1,item2,item3]
            if value.startswith("[") and value.endswith("]"):
                self._validate_list_value(value)

            # Check if it's a dict-like: {key1:val1,key2:val2}
            elif value.startswith("{") and value.endswith("}"):
                self._validate_dict_value(value)

            # Check numeric values
            elif self._is_numeric(value):
                self._validate_numeric_value(value)

            # Check boolean values
            elif value.lower() in ("true", "false", "null", "none"):
                pass  # Valid boolean/null values

            # String values - basic validation
            else:
                self._validate_string_value(value)

        except ValueError as e:
            return f"Invalid value format: {e}"

        return None

    def _validate_list_value(self, value: str) -> None:
        """Validate list-format configuration values.

        Args:
            value: List value to validate

        Raises:
            ValueError: If list format is invalid
        """
        # Remove brackets and split by comma
        inner = value[1:-1].strip()
        if not inner:
            return  # Empty list is valid

        items = [item.strip() for item in inner.split(",")]
        for item in items:
            if not item:
                raise ValueError("Empty list item")
            # Recursively validate each item
            if self._is_numeric(item):
                self._validate_numeric_value(item)

    def _validate_dict_value(self, value: str) -> None:
        """Validate dict-format configuration values.

        Args:
            value: Dict value to validate

        Raises:
            ValueError: If dict format is invalid
        """
        # Basic validation for dict-like syntax
        inner = value[1:-1].strip()
        if not inner:
            return  # Empty dict is valid

        # Simple validation - just check for balanced colons
        if ":" not in inner:
            raise ValueError("Dict format requires key:value pairs")

    def _validate_numeric_value(self, value: str) -> None:
        """Validate numeric configuration values.

        Args:
            value: Numeric value to validate

        Raises:
            ValueError: If numeric format is invalid
        """
        # Try to parse as int or float
        try:
            if "." in value or "e" in value.lower():
                float(value)
            else:
                int(value)
        except ValueError as e:
            raise ValueError(f"Invalid numeric value: {value}") from e

    def _validate_string_value(self, value: str) -> None:
        """Validate string configuration values.

        Args:
            value: String value to validate

        Raises:
            ValueError: If string contains dangerous patterns
        """
        # Check length
        if len(value) > 1000:
            raise ValueError("String value too long (max 1000 characters)")

        # Check for null bytes
        if "\x00" in value:
            raise ValueError("String contains null bytes")

    def _is_numeric(self, value: str) -> bool:
        """Check if a value appears to be numeric.

        Args:
            value: Value to check

        Returns:
            True if value appears numeric
        """
        # Simple numeric pattern check
        numeric_pattern = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
        return bool(numeric_pattern.match(value.strip()))
