"""
Unit tests for monitoring callbacks.

Tests the base callback functionality, callback handler, and specific
callback implementations like TimerCallback.
"""

from typing import Any
from unittest.mock import patch

import pytest

from crackseg.utils.monitoring.callbacks import (
    BaseCallback,
    CallbackHandler,
    TimerCallback,
)
from crackseg.utils.monitoring.manager import MonitoringManager


class TestBaseCallback:
    """Test suite for BaseCallback functionality."""

    def test_initialization(self) -> None:
        """Test BaseCallback initialization."""
        callback = BaseCallback()
        assert callback.metrics_manager is None

    def test_set_manager(self) -> None:
        """Test setting the monitoring manager."""
        callback = BaseCallback()
        manager = MonitoringManager()

        callback.set_manager(manager)
        assert callback.metrics_manager is manager

    def test_default_methods_do_nothing(self) -> None:
        """Test that default callback methods don't raise exceptions."""
        callback = BaseCallback()

        # These should not raise exceptions
        callback.on_train_begin()
        callback.on_train_end()
        callback.on_epoch_begin(0)
        callback.on_epoch_end(0)
        callback.on_batch_begin(0)
        callback.on_batch_end(0)


class TestCallbackHandler:
    """Test suite for CallbackHandler functionality."""

    def test_initialization(self) -> None:
        """Test CallbackHandler initialization."""
        manager = MonitoringManager()
        callback1 = BaseCallback()
        callback2 = BaseCallback()
        callbacks = [callback1, callback2]

        handler = CallbackHandler(callbacks, manager)

        assert handler.callbacks == callbacks
        assert handler.metrics_manager is manager
        assert callback1.metrics_manager is manager
        assert callback2.metrics_manager is manager

    def test_dynamic_method_creation(self) -> None:
        """Test dynamic method creation for callback execution."""
        manager = MonitoringManager()

        class TestCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.calls = []

            def on_train_begin(
                self, logs: dict[str, Any] | None = None
            ) -> None:
                self.calls.append("on_train_begin")

            def on_epoch_begin(
                self, epoch: int, logs: dict[str, Any] | None = None
            ) -> None:
                self.calls.append(f"on_epoch_begin_{epoch}")

        callback1 = TestCallback()
        callback2 = TestCallback()
        handler = CallbackHandler([callback1, callback2], manager)

        # Call methods dynamically
        handler.on_train_begin()
        handler.on_epoch_begin(5)

        # Verify both callbacks were called
        assert "on_train_begin" in callback1.calls
        assert "on_train_begin" in callback2.calls
        assert "on_epoch_begin_5" in callback1.calls
        assert "on_epoch_begin_5" in callback2.calls

    def test_empty_callbacks_list(self) -> None:
        """Test handler with empty callbacks list."""
        manager = MonitoringManager()
        handler = CallbackHandler([], manager)

        # Should not raise exceptions
        handler.on_train_begin()
        handler.on_epoch_end(0)

    def test_callback_error_handling(self) -> None:
        """Test that callback errors don't crash the handler."""
        manager = MonitoringManager()

        class FaultyCallback(BaseCallback):
            def on_train_begin(
                self, logs: dict[str, Any] | None = None
            ) -> None:
                raise ValueError("Test error")

        class GoodCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.called = False

            def on_train_begin(
                self, logs: dict[str, Any] | None = None
            ) -> None:
                self.called = True

        faulty_callback = FaultyCallback()
        good_callback = GoodCallback()
        handler = CallbackHandler([faulty_callback, good_callback], manager)

        # This should not raise an exception
        try:
            handler.on_train_begin()
        except ValueError:
            pytest.fail("Handler should handle callback errors gracefully")

        # Good callback should still be called
        assert good_callback.called


class TestTimerCallback:
    """Test suite for TimerCallback functionality."""

    def test_initialization(self) -> None:
        """Test TimerCallback initialization."""
        callback = TimerCallback()
        assert callback._start_time == 0
        assert callback._epoch_start_time == 0
        assert callback._batch_start_time == 0

    def test_train_timing(self) -> None:
        """Test training duration measurement."""
        callback = TimerCallback()
        manager = MonitoringManager()
        callback.set_manager(manager)

        # Use a single mock for all time.time() calls
        with patch("time.time", side_effect=[100.0, 150.0, 150.0]):
            callback.on_train_begin()
            callback.on_train_end()

        history = manager.get_history()
        assert "train/train_duration_sec_values" in history
        assert history["train/train_duration_sec_values"] == [50.0]

    def test_epoch_timing(self) -> None:
        """Test epoch duration measurement."""
        callback = TimerCallback()
        manager = MonitoringManager()
        callback.set_manager(manager)

        # Use a single mock for all time.time() calls
        with patch("time.time", side_effect=[100.0, 110.0, 110.0]):
            callback.on_epoch_begin(0)
            callback.on_epoch_end(0)

        history = manager.get_history()
        assert "train/epoch_duration_sec_values" in history
        assert history["train/epoch_duration_sec_values"] == [10.0]

    def test_batch_timing(self) -> None:
        """Test batch duration measurement."""
        callback = TimerCallback()
        manager = MonitoringManager()
        callback.set_manager(manager)

        # Use a single mock for all time.time() calls
        with patch("time.time", side_effect=[100.0, 100.5, 100.5]):
            callback.on_batch_begin(0)
            callback.on_batch_end(0)

        history = manager.get_history()
        assert "train/batch_duration_ms_values" in history
        assert history["train/batch_duration_ms_values"] == [
            500.0
        ]  # 0.5 * 1000

    def test_epoch_tracking(self) -> None:
        """Test that epoch number is tracked correctly."""
        callback = TimerCallback()
        manager = MonitoringManager()
        callback.set_manager(manager)

        callback.on_epoch_begin(5)
        assert manager.current_epoch == 5

    def test_step_tracking(self) -> None:
        """Test that step counter is incremented."""
        callback = TimerCallback()
        manager = MonitoringManager()
        callback.set_manager(manager)

        initial_step = manager.current_step
        callback.on_batch_begin(0)
        assert manager.current_step == initial_step + 1

    def test_multiple_epochs(self) -> None:
        """Test timing across multiple epochs."""
        callback = TimerCallback()
        manager = MonitoringManager()
        callback.set_manager(manager)

        # Use a single mock for all time.time() calls
        with patch(
            "time.time", side_effect=[100.0, 110.0, 110.0, 120.0, 125.0, 125.0]
        ):
            callback.on_epoch_begin(0)
            callback.on_epoch_end(0)
            callback.on_epoch_begin(1)
            callback.on_epoch_end(1)

        history = manager.get_history()
        assert history["train/epoch_duration_sec_values"] == [10.0, 5.0]

    def test_multiple_batches(self) -> None:
        """Test timing across multiple batches."""
        callback = TimerCallback()
        manager = MonitoringManager()
        callback.set_manager(manager)

        # Use a single mock for all time.time() calls
        with patch(
            "time.time", side_effect=[100.0, 100.1, 100.1, 100.2, 100.4, 100.4]
        ):
            callback.on_batch_begin(0)
            callback.on_batch_end(0)
            callback.on_batch_begin(1)
            callback.on_batch_end(1)

        history = manager.get_history()
        # Use approximate comparison for floating point values
        batch_durations = history["train/batch_duration_ms_values"]
        assert len(batch_durations) == 2
        assert abs(batch_durations[0] - 100.0) < 1e-6
        assert abs(batch_durations[1] - 200.0) < 1e-6

    def test_without_manager(self) -> None:
        """Test callback behavior when no manager is set."""
        callback = TimerCallback()

        # Should not raise exceptions
        callback.on_train_begin()
        callback.on_train_end()
        callback.on_epoch_begin(0)
        callback.on_epoch_end(0)
        callback.on_batch_begin(0)
        callback.on_batch_end(0)

    def test_timing_precision(self) -> None:
        """Test that timing measurements are reasonably precise."""
        callback = TimerCallback()
        manager = MonitoringManager()
        callback.set_manager(manager)

        # Use realistic time values
        start_time = 1234567890.123456
        end_time = 1234567890.987654
        expected_duration = end_time - start_time

        # Use a single mock for all time.time() calls
        with patch("time.time", side_effect=[start_time, end_time, end_time]):
            callback.on_train_begin()
            callback.on_train_end()

        history = manager.get_history()
        actual_duration = history["train/train_duration_sec_values"][0]
        assert abs(actual_duration - expected_duration) < 1e-6
