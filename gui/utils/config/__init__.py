"""
Configuration I/O utilities for the CrackSeg GUI application. This
package provides functionality for loading, validating, and managing
YAML configuration files used with Hydra for model training. The
package maintains backward compatibility with the original config_io
module while providing better modularity and maintainability. New
Advanced Features: - Unified configuration loading with comprehensive
validation - Advanced schema validation for crack segmentation models
- Sophisticated YAML parsing with nested structure support -
Comprehensive error reporting with detailed suggestions
"""

# Public API - maintains backward compatibility
from .cache import ConfigCache

# New advanced modules
from .config_loader import (
    ConfigLoadResult,
    UnifiedConfigLoader,
    load_config_with_validation,
)
from .error_reporter import (
    ConfigErrorReporter,
    ErrorCategory,
    ErrorReport,
    ErrorSeverity,
    generate_error_report,
)
from .exceptions import ConfigError, ValidationError
from .formatters import format_validation_report, get_validation_suggestions

# Enhanced I/O operations
from .io import (
    create_upload_progress_placeholder,
    get_config_metadata,
    get_upload_file_info,
    load_and_validate_config,
    load_config_file,
    scan_config_directories,
    update_upload_progress,
    upload_config_file,
    validate_uploaded_content,
)

# Advanced parsing capabilities
from .parsing_engine import (
    AdvancedYAMLParser,
    ConfigPath,
    extract_config_value,
    parse_nested_config,
)

# Enhanced schema validation
from .schema_validator import CrackSegSchemaValidator, validate_crackseg_schema
from .templates import create_config_from_template

# Comprehensive validation system
from .validation import (
    validate_config_structure,
    validate_config_types,
    validate_config_values,
    validate_with_hydra,
    validate_yaml_advanced,
    validate_yaml_syntax,
)

__all__ = [
    # Core exceptions
    "ConfigError",
    "ValidationError",
    # Core classes
    "ConfigCache",
    # New advanced classes
    "UnifiedConfigLoader",
    "ConfigLoadResult",
    "AdvancedYAMLParser",
    "ConfigPath",
    "CrackSegSchemaValidator",
    "ConfigErrorReporter",
    "ErrorReport",
    "ErrorSeverity",
    "ErrorCategory",
    # Enhanced I/O operations
    "load_config_file",
    "scan_config_directories",
    "get_config_metadata",
    "load_and_validate_config",
    "load_config_with_validation",  # New unified loader
    # File upload operations
    "upload_config_file",
    "get_upload_file_info",
    "validate_uploaded_content",
    "create_upload_progress_placeholder",
    "update_upload_progress",
    # Enhanced validation
    "validate_yaml_syntax",
    "validate_yaml_advanced",
    "validate_config_structure",
    "validate_config_types",
    "validate_config_values",
    "validate_with_hydra",
    "validate_crackseg_schema",  # New schema validator
    # Advanced parsing
    "parse_nested_config",  # New parser
    "extract_config_value",  # New extractor
    # Error reporting
    "generate_error_report",  # New error reporter
    # Templates and formatting
    "create_config_from_template",
    "format_validation_report",
    "get_validation_suggestions",
    # Global instances for backward compatibility
    "_config_cache",
    "_yaml_validator",
]

# Import global instances for backward compatibility
from .cache import _config_cache
from .validation import _yaml_validator

__version__ = "1.0.0"
