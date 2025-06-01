"""Unit tests for the logging utilities."""

import json
import logging
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

# Updated imports for the new logging structure
from src.utils.logging import ExperimentLogger, get_logger

# --- Fixtures ---


@pytest.fixture
def mock_logger_instance() -> MagicMock:
    """Fixture to create a mock underlying logger instance."""
    logger_instance = MagicMock(spec=logging.Logger)
    logger_instance.level = logging.DEBUG
    return logger_instance


@pytest.fixture
@patch("src.utils.logging.experiment.get_logger")
def experiment_logger(
    mock_get_logger: MagicMock, tmp_path: Path, mock_logger_instance: MagicMock
) -> ExperimentLogger:
    """Fixture for ExperimentLogger, mocking the internal logger.

    Patches get_logger to return a mock instance *before* init.
    Disables actual system info logging via patch.
    """
    mock_get_logger.return_value = mock_logger_instance

    config = OmegaConf.create({"test_param": "value"})
    logger = ExperimentLogger(
        log_dir=str(tmp_path),
        experiment_name="test_exp",
        config=config,
        log_system_stats=False,
        log_level="DEBUG",
    )
    return logger


# --- Tests for BaseLogger methods (tested via ExperimentLogger) ---


# def test_base_logger_log_methods(experiment_logger):
#     """Test BaseLogger logging methods call underlying logger via
#       ExperimentLogger."""
#     # Removed this test as BaseLogger is abstract and ExperimentLogger
#     # doesn't directly expose log_info etc. wrapper methods.
#     # Calls would be tested on the internal logger mock directly if needed,
#     # but ExperimentLogger tests cover the main functionalities.
#     pass


# --- Tests for ExperimentLogger ---


def test_experiment_logger_init(
    experiment_logger: ExperimentLogger, tmp_path: Path
) -> None:
    """Test ExperimentLogger initialization creates files/logs config."""
    mock_internal_logger = cast(MagicMock, experiment_logger.logger)

    assert isinstance(experiment_logger, ExperimentLogger)
    assert experiment_logger.log_dir == tmp_path
    # Buscar el subdirectorio de experimento generado dinámicamente
    exp_dirs = list((tmp_path / "experiments").glob("*-test_exp"))
    assert exp_dirs, "No experiment directory found"
    exp_dir = exp_dirs[0]
    config_path = exp_dir / "config.json"
    assert config_path.exists(), f"Config file not found at {config_path}"
    # Check calls during init
    init_log_call = f"Initialized experiment 'test_exp' in {tmp_path}"
    mock_internal_logger.info.assert_any_call(init_log_call)
    mock_internal_logger.info.assert_any_call("Experiment configuration:")
    mock_internal_logger.info.assert_any_call("  test_param: value")


def test_experiment_logger_log_scalar(
    experiment_logger: ExperimentLogger, tmp_path: Path
) -> None:
    """Test log_scalar writes to file and logs to console."""
    mock_internal_logger = cast(MagicMock, experiment_logger.logger)

    experiment_logger.log_scalar(tag="train/loss", value=0.5, step=10)

    # Buscar el subdirectorio de experimento generado dinámicamente
    exp_dirs = list((tmp_path / "experiments").glob("*-test_exp"))
    assert exp_dirs, "No experiment directory found"
    exp_dir = exp_dirs[0]
    metrics_path = exp_dir / "metrics" / "metrics.jsonl"
    assert metrics_path.exists(), f"Metrics file not found at {metrics_path}"
    with open(metrics_path) as f:
        line = f.readline()
        data = json.loads(line)
        assert data["name"] == "train/loss"
        assert data["value"] == 0.5  # noqa: PLR2004
        assert data["step"] == 10  # noqa: PLR2004

    # Check console log call (info level)
    expected_info_log = "[10] train/loss: 0.5"
    mock_internal_logger.info.assert_any_call(expected_info_log)


# --- Test logger setup/retrieval ---


def test_get_logger_setup(tmp_path: Path):
    """Test logger setup via get_logger and manual handler addition."""
    log_file_path = tmp_path / "get_logger_test.log"
    logger_name = "get_logger_module"

    logger = get_logger(logger_name, level="DEBUG")
    assert logger.level == logging.DEBUG
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    original_handlers = list(logger.handlers)

    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    test_message = "Testing get_logger file output"
    logger.info(test_message)
    assert log_file_path.exists()
    log_content = log_file_path.read_text()
    assert test_message in log_content
    assert f"[INFO] {logger_name}:" in log_content

    file_handler.close()
    logger.removeHandler(file_handler)
    assert logger.handlers == original_handlers


def test_get_logger_instance():
    """Test retrieving a logger instance returns the same object."""
    logger1 = get_logger("my_module_instance")
    logger2 = get_logger("my_module_instance")
    assert logger1 is logger2
    assert logger1.name == "my_module_instance"
