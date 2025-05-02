"""Utility functions for the Crack Segmentation project.

This package provides various helper modules and functions for tasks such as
configuration management, logging, device handling, seeding, and error
handling.
"""

# --- Core Utilities ---
from .device import get_device
from .seeds import set_random_seeds
from .checkpointing import save_checkpoint, load_checkpoint
from .early_stopping import EarlyStopping
from .factory import (
    import_class,
    get_optimizer,
    get_loss_fn,
    get_metrics_from_cfg
)
from .paths import get_abs_path, ensure_dir  # Added paths imports

# --- Configuration Subpackage ---
from .config import (
    ConfigSchema,
    validate_config,
    override_config,
    apply_overrides,
    save_config,
    load_env,         # Moved from top-level
    get_env_var,      # Moved from top-level
)

# --- Logging Subpackage ---
from .logging import (
    BaseLogger,
    ExperimentLogger,
    get_logger,
)

# --- Exception Handling ---
from .exceptions import (
    CrackSegError,
    ConfigError,
    DataError,
    ModelError,
    TrainingError,
    ResourceError,
    ValidationError,
)

# Define what gets imported with 'from src.utils import *'
__all__ = [
    # Core
    'get_device',
    'set_random_seeds',
    'save_checkpoint',
    'load_checkpoint',
    'EarlyStopping',
    'import_class',
    'get_optimizer',
    'get_loss_fn',
    'get_metrics_from_cfg',
    'get_abs_path',   # Added path function
    'ensure_dir',     # Added path function
    # Config
    'ConfigSchema',
    'validate_config',
    'override_config',
    'apply_overrides',
    'save_config',
    'load_env',
    'get_env_var',
    # Logging
    'BaseLogger',
    'ExperimentLogger',
    'get_logger',
    # Exceptions
    'CrackSegError',
    'ConfigError',
    'DataError',
    'ModelError',
    'TrainingError',
    'ResourceError',
    'ValidationError',
]
