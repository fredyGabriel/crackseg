from src.utils.config_schema import (
    DataConfig, ModelConfig, TrainingConfig, EvaluationConfig, RootConfig
)


def validate_data_config(cfg: DataConfig) -> None:
    """Validate DataConfig parameters."""
    if not cfg.data_root:
        raise ValueError("data_root must not be empty.")
    if not (0 < cfg.train_split < 1):
        raise ValueError("train_split must be in (0, 1).")
    if not (0 < cfg.val_split < 1):
        raise ValueError("val_split must be in (0, 1).")
    if not (0 < cfg.test_split < 1):
        raise ValueError("test_split must be in (0, 1).")
    if abs(cfg.train_split + cfg.val_split + cfg.test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0.")
    if any(s <= 0 for s in cfg.image_size):
        raise ValueError("All image_size values must be positive.")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate ModelConfig parameters."""
    if not cfg.model_name:
        raise ValueError("model_name must not be empty.")
    if cfg.input_channels <= 0:
        raise ValueError("input_channels must be positive.")
    if cfg.output_channels <= 0:
        raise ValueError("output_channels must be positive.")
    if not cfg.encoder_name:
        raise ValueError("encoder_name must not be empty.")


def validate_training_config(cfg: TrainingConfig) -> None:
    """Validate TrainingConfig parameters."""
    if cfg.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if cfg.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if cfg.step_size <= 0:
        raise ValueError("step_size must be positive.")
    if not (0 < cfg.gamma <= 1):
        raise ValueError("gamma must be in (0, 1].")
    if not cfg.optimizer:
        raise ValueError("optimizer must not be empty.")
    if not cfg.scheduler:
        raise ValueError("scheduler must not be empty.")


def validate_evaluation_config(cfg: EvaluationConfig) -> None:
    """Validate EvaluationConfig parameters."""
    if not cfg.metrics or not isinstance(cfg.metrics, list):
        raise ValueError("metrics must be a non-empty list.")
    if not all(isinstance(m, str) and m for m in cfg.metrics):
        raise ValueError("All metrics must be non-empty strings.")
    if not cfg.save_dir:
        raise ValueError("save_dir must not be empty.")


def validate_config(cfg: RootConfig) -> None:
    """Validate the entire configuration object."""
    validate_data_config(cfg.data)
    validate_model_config(cfg.model)
    validate_training_config(cfg.training)
    validate_evaluation_config(cfg.evaluation)
