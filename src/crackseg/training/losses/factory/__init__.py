"""
Clean recursive loss factory implementation.
This module provides the main factory for creating loss hierarchies
from configuration without circular dependencies.
"""

from .config_parser import ConfigParser, ConfigParsingError, ParsedNode
from .config_validator import ConfigValidationError, ConfigValidator
from .recursive_factory import RecursiveLossFactory

# Global factory instance
factory = RecursiveLossFactory()

__all__ = [
    "factory",
    "RecursiveLossFactory",
    "ConfigValidator",
    "ConfigValidationError",
    "ConfigParser",
    "ConfigParsingError",
    "ParsedNode",
]
