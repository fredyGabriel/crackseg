"""Logging utilities for the crack segmentation project.

This package provides comprehensive logging functionality including:
- Base logging classes and utilities
- Experiment-specific logging
- Training metrics management
- Logger setup and configuration utilities

Main components:
- base: Core logging classes and get_logger function
- experiment: ExperimentLogger for experiment tracking
- metrics_manager: MetricsManager for training metrics
- training: Training-specific logging helpers
- setup: Logger setup and configuration utilities
"""

# Base logging functionality
from .base import (
    BaseLogger,
    NoOpLogger,
    flatten_dict,
    get_logger,
    log_metrics_dict,
)

# Experiment logging
from .experiment import ExperimentLogger

# Metrics management
from .metrics_manager import MetricsManager

# Logger setup utilities
from .setup import (
    configure_root_logger,
    setup_internal_logger,
    setup_project_logger,
)

# Training-specific logging
from .training import (
    create_unified_metrics_logger,
    format_metrics,
    log_epoch_metrics,
    log_step_metrics,
    log_training_results,
    log_validation_results,
    safe_log,
)

# Define what gets imported with 'from crackseg.utils.logging import *'
__all__ = [
    # === BASE LOGGING ===
    "BaseLogger",
    "NoOpLogger",
    "get_logger",
    "flatten_dict",
    "log_metrics_dict",
    # === EXPERIMENT LOGGING ===
    "ExperimentLogger",
    # === METRICS MANAGEMENT ===
    "MetricsManager",
    # === TRAINING LOGGING ===
    "format_metrics",
    "safe_log",
    "log_validation_results",
    "log_training_results",
    "create_unified_metrics_logger",
    "log_epoch_metrics",
    "log_step_metrics",
    # === LOGGER SETUP ===
    "setup_internal_logger",
    "setup_project_logger",
    "configure_root_logger",
]
