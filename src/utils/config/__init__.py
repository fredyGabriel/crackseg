"""Configuration utilities for the crack segmentation project.

This module provides a comprehensive configuration management system with:
- Schema definitions for model, training, and data configurations
- Standardized storage with validation and metadata
- Environment variable management
- Configuration override utilities
- Path and value validation

Main components:
- schema: Configuration schemas and dataclasses
- standardized_storage: Advanced storage with validation and metadata
- validation: Basic configuration validation
- override: Configuration override utilities
- env: Environment variable management
- init: Hydra initialization utilities
"""

# Core schema definitions
# Environment and initialization
from .env import get_env_var, load_env
from .init import get_config, initialize_hydra, load_config, print_config

# Configuration overrides and saving
from .override import apply_overrides, override_config, save_config
from .schema import (
    # Loss configurations
    BaseLossConfig,
    # Metric configurations
    BaseMetricConfig,
    BCEDiceLossConfig,
    BCELossConfig,
    CombinedLossConfig,
    ConfigSchema,
    DataConfig,
    DiceLossConfig,
    F1ScoreConfig,
    FocalLossConfig,
    IoUScoreConfig,
    LoggingConfig,
    ModelConfig,
    PrecisionScoreConfig,
    RecallScoreConfig,
    TrainingConfig,
)

# Standardized storage and validation
from .standardized_storage import (
    ConfigurationSchema,
    StandardizedConfigStorage,
    compare_configurations,
    create_configuration_backup,
    enrich_configuration_with_environment,
    generate_environment_metadata,
    migrate_legacy_configuration,
    validate_configuration_completeness,
)

# Basic validation
from .validation import (
    validate_config,
    validate_config_with_standardized_checks,
    validate_model_config,
    validate_paths,
    validate_training_config,
)

# Define what gets imported with 'from src.utils.config import *'
__all__ = [
    # === CORE SCHEMAS ===
    "ConfigSchema",
    "DataConfig",
    "LoggingConfig",
    "ModelConfig",
    "TrainingConfig",
    # === LOSS CONFIGURATIONS ===
    "BaseLossConfig",
    "BCEDiceLossConfig",
    "BCELossConfig",
    "CombinedLossConfig",
    "DiceLossConfig",
    "FocalLossConfig",
    # === METRIC CONFIGURATIONS ===
    "BaseMetricConfig",
    "F1ScoreConfig",
    "IoUScoreConfig",
    "PrecisionScoreConfig",
    "RecallScoreConfig",
    # === STANDARDIZED STORAGE ===
    "ConfigurationSchema",
    "StandardizedConfigStorage",
    "generate_environment_metadata",
    "validate_configuration_completeness",
    "enrich_configuration_with_environment",
    "compare_configurations",
    "create_configuration_backup",
    "migrate_legacy_configuration",
    # === VALIDATION ===
    "validate_config",
    "validate_config_with_standardized_checks",
    "validate_model_config",
    "validate_paths",
    "validate_training_config",
    # === OVERRIDES & SAVING ===
    "override_config",
    "apply_overrides",
    "save_config",
    # === ENVIRONMENT & INITIALIZATION ===
    "load_env",
    "get_env_var",
    "initialize_hydra",
    "get_config",
    "load_config",
    "print_config",
]
