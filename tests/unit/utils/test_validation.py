from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.config.validation import (
    validate_config,
    validate_model_config,
    validate_paths,
    validate_training_config,
)
from src.utils.exceptions import ConfigError, ValidationError


def make_valid_config(tmp_path):
    """Create a valid DictConfig for testing validation."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    log_dir = tmp_path / "logs"
    config = OmegaConf.create(
        {
            "data": {
                "train_dir": str(data_dir),
                "val_dir": str(data_dir),
                "test_dir": str(data_dir),
            },
            "logging": {"log_dir": str(log_dir)},
            "training": {
                "epochs": 10,
                "optimizer": {"lr": 0.01, "weight_decay": 0.0},
                "early_stopping_patience": 5,
            },
            "model": {
                "in_channels": 3,
                "out_channels": 1,
                "features": [16, 32, 64],
            },
        }
    )
    return config


def test_validate_paths_valid(tmp_path):
    """Test validate_paths with valid directories."""
    cfg = make_valid_config(tmp_path)
    # Should not raise
    validate_paths(cfg)
    # Log dir should be created if not exists
    log_dir = Path(cfg.logging.log_dir)
    assert log_dir.exists()


def test_validate_paths_invalid(tmp_path):
    """Test validate_paths raises error for missing data dir."""
    cfg = make_valid_config(tmp_path)
    cfg.data.train_dir = str(tmp_path / "nonexistent")
    with pytest.raises(ValidationError):
        validate_paths(cfg)


def test_validate_training_config_valid(tmp_path):
    """Test validate_training_config with valid parameters."""
    cfg = make_valid_config(tmp_path)
    validate_training_config(cfg)


def test_validate_training_config_invalid(tmp_path):
    """Test validate_training_config raises error for invalid params."""
    cfg = make_valid_config(tmp_path)
    cfg.training.epochs = 0
    with pytest.raises(ValidationError):
        validate_training_config(cfg)
    cfg.training.epochs = 10
    cfg.training.optimizer.lr = -1
    with pytest.raises(ValidationError):
        validate_training_config(cfg)
    cfg.training.optimizer.lr = 0.01
    cfg.training.optimizer.weight_decay = -0.1
    with pytest.raises(ValidationError):
        validate_training_config(cfg)
    cfg.training.optimizer.weight_decay = 0.0
    cfg.training.early_stopping_patience = 0
    with pytest.raises(ValidationError):
        validate_training_config(cfg)


def test_validate_model_config_valid(tmp_path):
    """Test validate_model_config with valid parameters."""
    cfg = make_valid_config(tmp_path)
    validate_model_config(cfg)


def test_validate_model_config_invalid(tmp_path):
    """Test validate_model_config raises error for invalid params."""
    cfg = make_valid_config(tmp_path)
    cfg.model.in_channels = 0
    with pytest.raises(ValidationError):
        validate_model_config(cfg)
    cfg.model.in_channels = 3
    cfg.model.out_channels = 0
    with pytest.raises(ValidationError):
        validate_model_config(cfg)
    cfg.model.out_channels = 1
    cfg.model.features = []
    with pytest.raises(ValidationError):
        validate_model_config(cfg)
    cfg.model.features = [-1, 32]
    with pytest.raises(ValidationError):
        validate_model_config(cfg)


def test_validate_config_valid(tmp_path):
    """Test validate_config with a fully valid config."""
    cfg = make_valid_config(tmp_path)
    validate_config(cfg)


def test_validate_config_invalid(tmp_path):
    """Test validate_config raises ConfigError for invalid config."""
    cfg = make_valid_config(tmp_path)
    cfg.model.in_channels = 0
    with pytest.raises(ConfigError):
        validate_config(cfg)
