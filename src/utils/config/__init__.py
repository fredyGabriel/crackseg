"""Configuration utilities for the crack segmentation project."""

# Import necessary functions and classes from submodules
from .env import get_env_var, load_env
from .override import apply_overrides, override_config, save_config
from .schema import ConfigSchema
from .validation import validate_config

# Define what gets imported with 'from src.utils.config import *'
__all__ = [
    # Schema
    "ConfigSchema",
    # Validation
    "validate_config",
    # Overrides
    "override_config",
    "apply_overrides",
    "save_config",
    # Environment
    "load_env",
    "get_env_var",
]
