"""
Unit tests for MonitoringManager.

Tests the core functionality of the monitoring manager including
metrics logging, context switching, and history management.
"""

from unittest.mock import patch

from src.utils.monitoring.manager import MonitoringManager


class TestMonitoringManager:
    """Test suite for MonitoringManager functionality."""

    def test_initialization(self) -> None:
        """Test MonitoringManager initialization."""
        manager = MonitoringManager()

        assert manager.current_step == 0
        assert manager.current_epoch == 0
        assert manager.context == "train"
        assert len(manager.get_history()) == 0

    def test_log_metrics(self) -> None:
        """Test logging metrics with default step."""
        manager = MonitoringManager()

        metrics = {"loss": 0.5, "accuracy": 0.85}
        manager.log(metrics)

        history = manager.get_history()
        assert "train/loss_values" in history
        assert "train/accuracy_values" in history
        assert history["train/loss_values"] == [0.5]
        assert history["train/accuracy_values"] == [0.85]

    def test_log_metrics_with_custom_step(self) -> None:
        """Test logging metrics with custom step."""
        manager = MonitoringManager()

        metrics = {"loss": 0.3}
        manager.log(metrics, step=100)

        history = manager.get_history()
        assert history["train/loss_steps"] == [100]

    def test_context_switching(self) -> None:
        """Test context switching affects metric keys."""
        manager = MonitoringManager()

        # Log in train context
        manager.set_context("train")
        manager.log({"loss": 0.5})

        # Switch to validation context
        manager.set_context("val")
        manager.log({"loss": 0.3})

        history = manager.get_history()
        assert "train/loss_values" in history
        assert "val/loss_values" in history
        assert history["train/loss_values"] == [0.5]
        assert history["val/loss_values"] == [0.3]

    def test_get_last_metric(self) -> None:
        """Test retrieving last metric value."""
        manager = MonitoringManager()

        manager.log({"loss": 0.5})
        manager.log({"loss": 0.3})

        last_loss = manager.get_last_metric("train/loss")
        assert last_loss == 0.3

    def test_get_last_metric_nonexistent(self) -> None:
        """Test retrieving non-existent metric returns None."""
        manager = MonitoringManager()

        last_metric = manager.get_last_metric("nonexistent/metric")
        assert last_metric is None

    def test_reset(self) -> None:
        """Test resetting manager state."""
        manager = MonitoringManager()

        # Add some data
        manager.current_step = 10
        manager.current_epoch = 5
        manager.set_context("val")
        manager.log({"loss": 0.5})

        # Reset
        manager.reset()

        assert manager.current_step == 0
        assert manager.current_epoch == 0
        assert manager.context == "train"
        assert len(manager.get_history()) == 0

    def test_timestamping(self) -> None:
        """Test that timestamps are recorded for metrics."""
        manager = MonitoringManager()

        with patch("time.time", return_value=1234567890.0):
            manager.log({"loss": 0.5})

        history = manager.get_history()
        assert "train/loss_timestamps" in history
        assert history["train/loss_timestamps"] == [1234567890.0]

    def test_step_tracking(self) -> None:
        """Test that steps are tracked correctly."""
        manager = MonitoringManager()

        manager.log({"loss": 0.5}, step=10)
        manager.log({"loss": 0.3}, step=20)

        history = manager.get_history()
        assert history["train/loss_steps"] == [10, 20]

    def test_multiple_metrics_single_log(self) -> None:
        """Test logging multiple metrics in a single call."""
        manager = MonitoringManager()

        metrics = {"loss": 0.5, "accuracy": 0.85, "f1_score": 0.78}
        manager.log(metrics)

        history = manager.get_history()
        assert len(history) == 9  # 3 metrics * 3 tracking arrays each
        assert history["train/loss_values"] == [0.5]
        assert history["train/accuracy_values"] == [0.85]
        assert history["train/f1_score_values"] == [0.78]

    def test_mixed_metric_types(self) -> None:
        """Test logging different types of numeric metrics."""
        manager = MonitoringManager()

        metrics = {
            "int_metric": 42,
            "float_metric": 3.14,
            "zero_metric": 0,
            "negative_metric": -1.5,
        }
        manager.log(metrics)

        history = manager.get_history()
        assert history["train/int_metric_values"] == [42]
        assert history["train/float_metric_values"] == [3.14]
        assert history["train/zero_metric_values"] == [0]
        assert history["train/negative_metric_values"] == [-1.5]

    def test_context_persistence(self) -> None:
        """Test that context persists across multiple log calls."""
        manager = MonitoringManager()

        manager.set_context("validation")
        manager.log({"metric1": 1.0})
        manager.log({"metric2": 2.0})

        history = manager.get_history()
        assert "validation/metric1_values" in history
        assert "validation/metric2_values" in history
