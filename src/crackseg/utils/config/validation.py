"""Configuration validation utilities.

This module provides basic validation for Hydra configurations.
For advanced validation with schema checking, see standardized_storage module.
"""

from pathlib import Path

from omegaconf import DictConfig

from crackseg.utils.core.exceptions import ConfigError, ValidationError


def validate_paths(config: DictConfig) -> None:
    """Validate that required paths exist.

    Args:
        config: Configuration to validate

    Raises:
        ValidationError: If a required path does not exist
    """
    # Data directories validation
    # Check if using unified configuration
    if hasattr(config.data, "data_root") and config.data.data_root:
        # Unified mode: validate that data_root exists
        data_root_path = Path(config.data.data_root)
        if not data_root_path.exists():
            raise ValidationError(
                f"Data root directory does not exist: {config.data.data_root}",
                field="data_root",
            )

        # Check for unified structure
        images_dir = data_root_path / "images"
        masks_dir = data_root_path / "masks"
        if not images_dir.exists():
            raise ValidationError(
                f"Images directory does not exist: {images_dir}",
                field="images_dir",
            )
        if not masks_dir.exists():
            raise ValidationError(
                f"Masks directory does not exist: {masks_dir}",
                field="masks_dir",
            )
    else:
        # Legacy mode: validate individual directories (for backward compatibility)
        data_paths = {}
        if config.data.train_dir:
            data_paths["train_dir"] = config.data.train_dir
        if config.data.val_dir:
            data_paths["val_dir"] = config.data.val_dir
        if config.data.test_dir:
            data_paths["test_dir"] = config.data.test_dir

        for name, path in data_paths.items():
            if not Path(path).exists():
                raise ValidationError(
                    f"Directory does not exist: {path}", field=name
                )

    # Log directory
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


def validate_training_config(config: DictConfig) -> None:
    """Validate training configuration values.

    Args:
        config: Configuration to validate

    Raises:
        ValidationError: If any training parameters are invalid
    """
    training = config.training

    if training.epochs <= 0:
        raise ValidationError("Must be positive", field="training.epochs")

    if training.optimizer.lr <= 0:
        raise ValidationError(
            "Must be positive", field="training.optimizer.lr"
        )

    if training.optimizer.weight_decay < 0:
        raise ValidationError(
            "Must be non-negative", field="training.optimizer.weight_decay"
        )

    if training.early_stopping_patience <= 0:
        raise ValidationError(
            "Must be positive", field="training.early_stopping_patience"
        )


def validate_model_config(config: DictConfig) -> None:
    """Validate model configuration values.

    Args:
        config: Configuration to validate

    Raises:
        ValidationError: If any model parameters are invalid
    """
    model = config.model

    if model.in_channels <= 0:
        raise ValidationError("Must be positive", field="model.in_channels")

    if model.out_channels <= 0:
        raise ValidationError("Must be positive", field="model.out_channels")

    if not model.features or any(f <= 0 for f in model.features):
        raise ValidationError(
            "Must be a non-empty list of positive integers",
            field="model.features",
        )


def validate_config(config: DictConfig) -> None:
    """Validate the complete configuration.

    This function provides basic validation for Hydra configurations.
    For more advanced validation with schema checking and completeness
    analysis, use validate_configuration_completeness from the
    standardized_storage module.

    Args:
        config: Configuration to validate

    Raises:
        ConfigError: If the configuration is invalid
    """
    try:
        validate_paths(config)
        validate_training_config(config)
        validate_model_config(config)
    except ValidationError as e:
        raise ConfigError(e.message, field=e.field, details=e.details) from e


def validate_config_with_standardized_checks(
    config: DictConfig,
) -> dict[str, bool]:
    """Perform both basic and standardized validation checks.

    This is a convenience function that combines basic validation with
    the standardized storage validation approach.

    Args:
        config: Configuration to validate

    Returns:
        Dictionary with validation results from both approaches

    Raises:
        ConfigError: If basic validation fails
    """
    # Import here to avoid circular imports
    try:
        from .standardized_storage import validate_configuration_completeness
    except ImportError:
        # Fallback if standardized_storage is not available
        validate_configuration_completeness = None

    # Basic validation (raises on failure)
    validate_config(config)

    results = {"basic_validation": True}

    # Standardized validation (returns detailed results)
    if validate_configuration_completeness:
        standardized_results = validate_configuration_completeness(config)
        results.update(
            {
                "standardized_validation": standardized_results["is_valid"],
                "completeness_score": standardized_results.get(
                    "completeness_score", 0.0
                ),
                "missing_required": standardized_results.get(
                    "missing_required", []
                ),
                "missing_recommended": standardized_results.get(
                    "missing_recommended", []
                ),
            }
        )

    return results
