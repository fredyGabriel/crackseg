"""
Custom exceptions for override parsing in CrackSeg. This module
defines specific exception types for Hydra override parsing errors,
validation failures, and security violations.
"""


class OverrideParsingError(Exception):
    """Custom exception for override parsing errors.

    Raised when Hydra override parsing fails due to:
    - Invalid syntax in override strings
    - Security policy violations
    - Type validation failures
    - Malformed configuration values

    Examples:
        >>> raise OverrideParsingError("Invalid override format: key==value")
        >>> raise OverrideParsingError("Security violation detected")
    """

    pass
