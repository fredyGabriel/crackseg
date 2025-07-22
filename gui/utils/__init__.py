"""
Utilities package for the CrackSeg GUI application. This package
contains configuration, session state management, and other utility
modules.
"""

from gui.utils.config_io import (
    ConfigCache,
    ConfigError,
    ValidationError,
    YAMLValidator,
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
from gui.utils.gui_config import PAGE_CONFIG
from gui.utils.session_state import SessionState, SessionStateManager

__all__ = [
    "PAGE_CONFIG",
    "SessionState",
    "SessionStateManager",
    # config_io exports
    "ConfigCache",
    "ConfigError",
    "ValidationError",
    "YAMLValidator",
    "load_config_file",
    "load_and_validate_config",
    "scan_config_directories",
    "validate_yaml_syntax",
    "validate_yaml_advanced",
    "validate_config_structure",
    "validate_config_types",
    "validate_config_values",
    "validate_with_hydra",
    "get_config_metadata",
    "get_validation_suggestions",
    "format_validation_report",
    "create_config_from_template",
]
