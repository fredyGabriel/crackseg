"""Configuration I/O utilities for the CrackSeg GUI application.

This module provides backward compatibility with the original config_io module.
All functionality has been moved to the scripts.gui.utils.config package for
better modularity and maintainability.

This file serves as a compatibility layer and will be deprecated in future
versions.
Please update your imports to use:
    from scripts.gui.utils.config import ...
"""

# Import everything from the new config package for backward compatibility
from scripts.gui.utils.config import *  # noqa: F403, F401

# Explicit imports for better IDE support and documentation
from scripts.gui.utils.config import (
    ConfigCache,
    ConfigError,
    ValidationError,
    _config_cache,
    _yaml_validator,
    create_config_from_template,
    format_validation_report,
    get_config_metadata,
    get_validation_suggestions,
    load_and_validate_config,
    load_config_file,
    scan_config_directories,
    validate_config_structure,
    validate_config_types,
    validate_config_values,
    validate_with_hydra,
    validate_yaml_advanced,
    validate_yaml_syntax,
)

# Export YAMLValidator for backward compatibility
from scripts.gui.utils.config.validation.yaml_engine import YAMLValidator

__all__ = [
    # Exceptions
    "ConfigError",
    "ValidationError",
    # Core classes
    "ConfigCache",
    "YAMLValidator",
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
    # Legacy support
    "_config_cache",
    "_yaml_validator",
]
