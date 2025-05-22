"""Utility functions for the Crack Segmentation project.

This package provides various helper modules and functions for tasks such as
configuration management, logging, device handling, seeding, and error
handling.
"""

# --- Core Utilities ---
from .checkpointing import load_checkpoint, save_checkpoint

# --- Configuration Subpackage ---
from .config import (
    ConfigSchema,
    apply_overrides,
    get_env_var,  # Moved from top-level
    load_env,  # Moved from top-level
    override_config,
    save_config,
    validate_config,
)
from .device import get_device
from .early_stopping import EarlyStopping

# --- Exception Handling ---
from .exceptions import (
    ConfigError,
    CrackSegError,
    DataError,
    ModelError,
    ResourceError,
    TrainingError,
    ValidationError,
)
from .factory import (
    get_loss_fn,
    get_metrics_from_cfg,
    get_optimizer,
    import_class,
)

# --- Logging Subpackage ---
from .logging import (
    BaseLogger,
    ExperimentLogger,
    get_logger,
)
from .paths import ensure_dir, get_abs_path  # Added paths imports
from .seeds import set_random_seeds

# Define what gets imported with 'from src.utils import *'
__all__ = [
    # Core
    "get_device",
    "set_random_seeds",
    "save_checkpoint",
    "load_checkpoint",
    "EarlyStopping",
    "import_class",
    "get_optimizer",
    "get_loss_fn",
    "get_metrics_from_cfg",
    "get_abs_path",  # Added path function
    "ensure_dir",  # Added path function
    # Config
    "ConfigSchema",
    "validate_config",
    "override_config",
    "apply_overrides",
    "save_config",
    "load_env",
    "get_env_var",
    # Logging
    "BaseLogger",
    "ExperimentLogger",
    "get_logger",
    # Exceptions
    "CrackSegError",
    "ConfigError",
    "DataError",
    "ModelError",
    "TrainingError",
    "ResourceError",
    "ValidationError",
]
