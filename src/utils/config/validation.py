"""Configuration validation utilities."""

from pathlib import Path

from omegaconf import DictConfig

from src.utils.exceptions import ConfigError, ValidationError


def validate_paths(config: DictConfig) -> None:
    """Validate that required paths exist.

    Args:
        config: Configuration to validate

    Raises:
        ValidationError: If a required path does not exist
    """
    # Data directories
    data_paths = {
        "train_dir": config.data.train_dir,
        "val_dir": config.data.val_dir,
    }
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
