"""Tests for logging utility functions."""

import logging  # Needed for caplog

from src.utils.loggers import log_metrics_dict, BaseLogger  # get_logger unused


class MockLogger(BaseLogger):
    """Mock logger for testing."""
    def __init__(self):
        self.logs = {}

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.logs[(tag, step)] = value

    def log_config(self, config) -> None:
        pass

    def close(self) -> None:
        pass


def test_log_metrics_dict_basic():
    """Test logging a simple dictionary of metrics."""
    mock_logger = MockLogger()
    metrics = {"loss": 0.5, "accuracy": 0.9}
    step = 10

    log_metrics_dict(mock_logger, metrics, step)

    assert mock_logger.logs == {
        ("loss", 10): 0.5,
        ("accuracy", 10): 0.9
    }


def test_log_metrics_dict_with_prefix():
    """Test logging metrics with a prefix."""
    mock_logger = MockLogger()
    metrics = {"iou": 0.75, "f1": 0.85}
    step = 5
    prefix = "val/"

    log_metrics_dict(mock_logger, metrics, step, prefix)

    assert mock_logger.logs == {
        ("val/iou", 5): 0.75,
        ("val/f1", 5): 0.85
    }


def test_log_metrics_dict_empty():
    """Test logging an empty dictionary."""
    mock_logger = MockLogger()
    metrics = {}
    step = 1

    log_metrics_dict(mock_logger, metrics, step)
    assert not mock_logger.logs  # Should be empty


def test_log_metrics_dict_with_non_float():
    """Test logging metrics containing non-float values (should handle)."""
    mock_logger = MockLogger()
    metrics = {"loss": 0.1, "epoch": 2, "is_best": False}
    step = 2

    log_metrics_dict(mock_logger, metrics, step, prefix="train/")

    # Expect only float/int metrics to be logged
    assert mock_logger.logs == {
        ("train/loss", 2): 0.1,
        ("train/epoch", 2): 2.0  # Implicitly converted by log_scalar
    }


def test_log_metrics_dict_logger_none():
    """Test that nothing happens if logger is None."""
    metrics = {"loss": 0.5}
    step = 1
    # Should not raise an error
    log_metrics_dict(None, metrics, step)


def test_log_metric_failure_warning(caplog):
    """Test that a warning is logged if log_scalar fails."""
    caplog.set_level(logging.WARNING)  # Ensure warnings are captured

    class FailingMockLogger(MockLogger):
        def log_scalar(self, tag: str, value: float, step: int) -> None:
            if tag == "fail_me":
                raise ValueError("Mock logging error")
            super().log_scalar(tag, value, step)

    failing_logger = FailingMockLogger()
    metrics = {"loss": 0.5, "fail_me": 0.1}
    step = 3

    log_metrics_dict(failing_logger, metrics, step)

    # Check logs recorded by standard Python logger
    assert "Failed to log metric 'fail_me' at step 3" in caplog.text
    assert "Mock logging error" in caplog.text
    # Ensure other metrics were still logged
    assert failing_logger.logs == {("loss", 3): 0.5}
