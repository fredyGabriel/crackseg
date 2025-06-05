"""Configuration I/O utilities for the CrackSeg GUI application.

This package provides functionality for loading, validating, and managing
YAML configuration files used with Hydra for model training.

The package maintains backward compatibility with the original config_io module
while providing better modularity and maintainability.
"""

# Public API - maintains backward compatibility
from .cache import ConfigCache
from .exceptions import ConfigError, ValidationError
from .formatters import format_validation_report, get_validation_suggestions
from .io import (
    get_config_metadata,
    load_and_validate_config,
    load_config_file,
    scan_config_directories,
)
from .templates import create_config_from_template
from .validation import (
    validate_config_structure,
    validate_config_types,
    validate_config_values,
    validate_with_hydra,
    validate_yaml_advanced,
    validate_yaml_syntax,
)

__all__ = [
    # Exceptions
    "ConfigError",
    "ValidationError",
    # Core classes
    "ConfigCache",
    # I/O operations
    "load_config_file",
    "scan_config_directories",
    "get_config_metadata",
    "load_and_validate_config",
    # Validation
    "validate_yaml_syntax",
    "validate_yaml_advanced",
    "validate_config_structure",
    "validate_config_types",
    "validate_config_values",
    "validate_with_hydra",
    # Templates and formatting
    "create_config_from_template",
    "format_validation_report",
    "get_validation_suggestions",
    # Global instances for backward compatibility
    "_config_cache",
    "_yaml_validator",
]

# Global instances for backward compatibility
from .cache import _config_cache  # pyright: ignore[reportPrivateUsage]
from .validation import _yaml_validator  # pyright: ignore[reportPrivateUsage]

__version__ = "1.0.0"
