"""
Integration tests for the monitoring framework.

Simple integration tests that verify the monitoring framework works
with the Trainer without requiring complex setup.
"""

from typing import Any
from unittest.mock import patch

import pytest
import torch

from crackseg.utils.monitoring import (
    BaseCallback,
    CallbackHandler,
    GPUStatsCallback,
    MonitoringManager,
    SystemStatsCallback,
    TimerCallback,
)


class MonitoringTestCallback(BaseCallback):
    """Test callback that tracks method calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.calls.append("on_train_begin")

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.calls.append("on_train_end")

    def on_epoch_begin(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.calls.append(f"on_epoch_begin_{epoch}")

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.calls.append(f"on_epoch_end_{epoch}")

    def on_batch_begin(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.calls.append(f"on_batch_begin_{batch}")

    def on_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.calls.append(f"on_batch_end_{batch}")


class TestMonitoringFrameworkIntegration:
    """Integration tests for the monitoring framework."""

    def test_monitoring_manager_basic_functionality(self) -> None:
        """Test basic monitoring manager functionality."""
        manager = MonitoringManager()

        # Test logging metrics
        manager.log({"loss": 0.5, "accuracy": 0.8})

        # Test history retrieval
        history = manager.get_history()
        assert "train/loss_values" in history
        assert "train/accuracy_values" in history
        assert history["train/loss_values"] == [0.5]
        assert history["train/accuracy_values"] == [0.8]

    def test_callback_handler_functionality(self) -> None:
        """Test callback handler with multiple callbacks."""
        manager = MonitoringManager()
        test_callback = MonitoringTestCallback()
        timer_callback = TimerCallback()

        handler = CallbackHandler([test_callback, timer_callback], manager)

        # Test callback execution
        handler.on_train_begin()
        handler.on_epoch_begin(0)
        handler.on_batch_begin(0)
        handler.on_batch_end(0)
        handler.on_epoch_end(0)
        handler.on_train_end()

        # Verify test callback was called
        assert "on_train_begin" in test_callback.calls
        assert "on_epoch_begin_0" in test_callback.calls
        assert "on_batch_begin_0" in test_callback.calls
        assert "on_batch_end_0" in test_callback.calls
        assert "on_epoch_end_0" in test_callback.calls
        assert "on_train_end" in test_callback.calls

    def test_system_stats_callback_integration(self) -> None:
        """Test SystemStatsCallback integration."""
        manager = MonitoringManager()
        system_callback = SystemStatsCallback()
        system_callback.set_manager(manager)

        # Test system stats collection
        system_callback.on_epoch_end(0)

        history = manager.get_history()
        assert "train/cpu_util_percent_values" in history
        assert "train/ram_used_gb_values" in history
        assert "train/ram_util_percent_values" in history

        # Values should be reasonable
        cpu_percent = history["train/cpu_util_percent_values"][0]
        memory_used = history["train/ram_used_gb_values"][0]
        memory_percent = history["train/ram_util_percent_values"][0]

        assert 0 <= cpu_percent <= 100
        assert memory_used > 0
        assert 0 <= memory_percent <= 100

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_gpu_stats_callback_integration(self) -> None:
        """Test GPUStatsCallback integration when CUDA is available."""
        manager = MonitoringManager()
        gpu_callback = GPUStatsCallback()
        gpu_callback.set_manager(manager)

        # Test GPU stats collection
        gpu_callback.on_train_begin()  # Initialize pynvml
        gpu_callback.on_epoch_end(0)

        history = manager.get_history()
        assert "train/gpu_vram_used_gb_values" in history
        assert "train/gpu_vram_util_percent_values" in history
        assert "train/gpu_util_percent_values" in history

        # Values should be reasonable
        gpu_memory_used = history["train/gpu_vram_used_gb_values"][0]
        gpu_memory_percent = history["train/gpu_vram_util_percent_values"][0]
        gpu_utilization = history["train/gpu_util_percent_values"][0]

        assert gpu_memory_used >= 0
        assert 0 <= gpu_memory_percent <= 100
        assert 0 <= gpu_utilization <= 100

        # Clean up
        gpu_callback.on_train_end()

    def test_callback_error_handling(self) -> None:
        """Test that callback errors don't crash the system."""

        class FaultyCallback(BaseCallback):
            def on_train_begin(
                self, logs: dict[str, Any] | None = None
            ) -> None:
                raise ValueError("Test error")

        manager = MonitoringManager()
        faulty_callback = FaultyCallback()
        good_callback = MonitoringTestCallback()

        handler = CallbackHandler([faulty_callback, good_callback], manager)

        # This should not raise an exception
        handler.on_train_begin()

        # Good callback should still be called
        assert "on_train_begin" in good_callback.calls

    def test_metrics_collection_with_timing(self) -> None:
        """Test metrics collection with timing information."""
        manager = MonitoringManager()
        timer_callback = TimerCallback()
        timer_callback.set_manager(manager)

        # Simulate training timing
        with patch("time.time", side_effect=[100.0, 150.0, 150.0]):
            timer_callback.on_train_begin()
            timer_callback.on_train_end()

        history = manager.get_history()
        assert "train/train_duration_sec_values" in history
        assert history["train/train_duration_sec_values"] == [50.0]

    def test_context_switching(self) -> None:
        """Test that monitoring manager correctly switches contexts."""
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

    def test_multiple_callbacks_integration(self) -> None:
        """Test integration with multiple callbacks."""
        manager = MonitoringManager()
        test_callback = MonitoringTestCallback()
        timer_callback = TimerCallback()
        system_callback = SystemStatsCallback()

        handler = CallbackHandler(
            [test_callback, timer_callback, system_callback], manager
        )

        # Execute callbacks
        handler.on_train_begin()
        handler.on_epoch_begin(0)
        handler.on_batch_begin(0)
        handler.on_batch_end(0)
        handler.on_epoch_end(0)
        handler.on_train_end()

        # Verify all callbacks were executed
        assert len(test_callback.calls) == 6

        # Verify metrics were collected
        history = manager.get_history()
        assert len(history) > 0

        # Should have system stats (from epoch_end)
        assert "train/cpu_util_percent_values" in history
        assert "train/ram_used_gb_values" in history
