"""Utility functions for the Crack Segmentation project.

This package provides various helper modules and functions for tasks such as
configuration management, logging, device handling, seeding, and error
handling.
"""

# --- Core Utilities ---
# --- Artifact Management ---
from .artifact_manager import (
    ArtifactManager,
    ArtifactMetadata,
)

# --- Checkpointing ---
from .checkpointing import (
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    CheckpointSpec,
    generate_checkpoint_metadata,
    load_checkpoint,
    load_checkpoint_dict,
    save_checkpoint,
    verify_checkpoint_integrity,
)

# --- Component Cache ---
from .component_cache import (
    cache_component,
    clear_component_cache,
    generate_cache_key,
    get_cache_info,
    get_cached_component,
)

# --- Configuration Subpackage ---
from .config import (
    ConfigSchema,
    apply_overrides,
    get_env_var,
    load_env,
    override_config,
    save_config,
    validate_config,
)
from .core import (
    ConfigError,
    CrackSegError,
    DataError,
    EvaluationError,
    ModelError,
    ResourceError,
    TrainingError,
    ValidationError,
    ensure_dir,
    get_abs_path,
    get_device,
    set_random_seeds,
)

# --- Experiment Management ---
from .experiment import (
    ExperimentManager,
    initialize_experiment,
)

# --- Factory and Component Creation ---
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
    MetricsManager,
    create_unified_metrics_logger,
    # Training logging functions
    format_metrics,
    get_logger,
    log_training_results,
    log_validation_results,
    # Logger setup functions
    setup_internal_logger,
    setup_project_logger,
)

# --- Monitoring ---
from .monitoring import (
    AlertingSystem,
    BaseCallback,
    CoverageMonitor,
    MonitoringManager,
    ResourceMonitor,
)

# --- Training Utilities ---
from .training import (
    EarlyStopping,
    GradScaler,
    amp_autocast,
    optimizer_step_with_accumulation,
    setup_early_stopping,
    step_scheduler_helper,
)

# --- Visualization ---
from .visualization import (
    visualize_predictions,
)

# Define what gets imported with 'from crackseg.utils import *'
__all__ = [
    # Core utilities
    "get_device",
    "set_random_seeds",
    "ensure_dir",
    "get_abs_path",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_dict",
    "verify_checkpoint_integrity",
    "CheckpointSaveConfig",
    "CheckpointLoadConfig",
    "CheckpointSpec",
    "generate_checkpoint_metadata",
    # Training utilities
    "EarlyStopping",
    "setup_early_stopping",
    "step_scheduler_helper",
    "GradScaler",
    "amp_autocast",
    "optimizer_step_with_accumulation",
    # Factory and components
    "get_loss_fn",
    "get_metrics_from_cfg",
    "get_optimizer",
    "import_class",
    "cache_component",
    "clear_component_cache",
    "generate_cache_key",
    "get_cached_component",
    "get_cache_info",
    # Experiment management
    "initialize_experiment",
    "ExperimentManager",
    # Artifact management
    "ArtifactManager",
    "ArtifactMetadata",
    # Configuration
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
    "MetricsManager",
    "format_metrics",
    "log_validation_results",
    "log_training_results",
    "create_unified_metrics_logger",
    "setup_internal_logger",
    "setup_project_logger",
    # Visualization
    "visualize_predictions",
    # Exceptions
    "CrackSegError",
    "ConfigError",
    "DataError",
    "EvaluationError",
    "ModelError",
    "TrainingError",
    "ResourceError",
    "ValidationError",
    # Monitoring
    "ResourceMonitor",
    "AlertingSystem",
    "BaseCallback",
    "CoverageMonitor",
    "MonitoringManager",
]
