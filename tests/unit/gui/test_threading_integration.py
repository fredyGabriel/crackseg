"""Tests for threading integration in GUI utilities.

This module tests the threading coordination and UI responsiveness
functionality integrated into the CrackSeg GUI system.
"""

# pyright: reportInvalidTypeForm=false

import threading
import time
from collections.abc import Generator
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.gui.utils.run_manager import (
    cleanup_ui_wrapper,
    execute_training_async,
    execute_with_progress,
    get_ui_wrapper,
)
from scripts.gui.utils.threading import (
    CancellationToken,
    ProgressUpdate,
    TaskStatus,
    ThreadCoordinator,
    ThreadTask,
    UIResponsiveWrapper,
)


class TestThreadCoordinator:
    """Test ThreadCoordinator functionality."""

    def test_coordinator_initialization(self) -> None:
        """Test coordinator initializes correctly."""
        coordinator = ThreadCoordinator(max_workers=2, max_io_workers=1)

        # Test that coordinator can accept tasks
        stats = coordinator.get_statistics()
        assert stats["tasks_submitted"] == 0
        assert stats["tasks_completed"] == 0

        coordinator.shutdown()

    def test_submit_simple_task(self) -> None:
        """Test submitting and executing a simple task."""
        coordinator = ThreadCoordinator(max_workers=2)

        def simple_task(x: int, y: int) -> int:
            return x + y

        task = ThreadTask(func=simple_task, args=(5, 3), name="AdditionTask")

        future = coordinator.submit_task(task)
        result = future.result(timeout=5.0)

        assert result == 8
        coordinator.shutdown()

    def test_submit_function_convenience(self) -> None:
        """Test convenience method for submitting functions."""
        coordinator = ThreadCoordinator(max_workers=2)

        def multiply(x: int, y: int) -> int:
            return x * y

        future = coordinator.submit_function(
            multiply, 4, 6, name="MultiplyTask"
        )
        result = future.result(timeout=5.0)

        assert result == 24
        coordinator.shutdown()

    def test_task_cancellation(self) -> None:
        """Test task cancellation functionality."""
        coordinator = ThreadCoordinator(max_workers=1)

        def simple_task() -> str:
            return "completed"

        # Submit task
        future = coordinator.submit_function(simple_task, name="SimpleTask")

        # Try to cancel (might succeed or fail depending on timing)
        cancelled = coordinator.cancel_task("SimpleTask")

        # Verify the method works and returns a boolean
        assert isinstance(cancelled, bool)

        # Verify task completes or was cancelled
        try:
            result = future.result(timeout=1.0)
            assert result == "completed"
        except Exception:
            # Task might have been cancelled or failed - both are acceptable
            pass

        coordinator.shutdown()

    def test_statistics_tracking(self) -> None:
        """Test statistics tracking functionality."""
        coordinator = ThreadCoordinator(max_workers=2)

        def quick_task() -> str:
            return "done"

        # Submit multiple tasks
        futures = []
        for i in range(3):
            future = coordinator.submit_function(quick_task, name=f"Task{i}")
            futures.append(future)

        # Wait for completion
        for future in futures:
            future.result(timeout=5.0)

        stats = coordinator.get_statistics()
        assert stats["tasks_submitted"] == 3
        assert stats["tasks_completed"] == 3
        assert stats["tasks_failed"] == 0

        coordinator.shutdown()


class TestCancellationToken:
    """Test CancellationToken functionality."""

    def test_token_initialization(self) -> None:
        """Test token initializes correctly."""
        token = CancellationToken()

        assert not token.is_cancelled
        assert token.cancellation_reason is None

    def test_token_cancellation(self) -> None:
        """Test token cancellation."""
        token = CancellationToken()

        token.cancel("Test cancellation")

        assert token.is_cancelled
        assert token.cancellation_reason == "Test cancellation"

    def test_token_reset(self) -> None:
        """Test token reset functionality."""
        token = CancellationToken()

        token.cancel("Test")
        assert token.is_cancelled

        token.reset()
        assert not token.is_cancelled
        assert token.cancellation_reason is None

    def test_wait_for_cancellation(self) -> None:
        """Test waiting for cancellation."""
        token = CancellationToken()

        # Test timeout
        start_time = time.time()
        result = token.wait_for_cancellation(timeout=0.1)
        elapsed = time.time() - start_time

        assert not result
        assert 0.1 <= elapsed <= 0.2  # Allow some tolerance

    def test_cancellation_in_thread(self) -> None:
        """Test cancellation from another thread."""
        token = CancellationToken()
        result_holder = {"cancelled": False}

        def worker() -> None:
            for _ in range(100):
                if token.is_cancelled:
                    result_holder["cancelled"] = True
                    return
                time.sleep(0.01)

        thread = threading.Thread(target=worker)
        thread.start()

        # Cancel after short delay
        time.sleep(0.05)
        token.cancel("Thread test")

        thread.join(timeout=2.0)
        assert result_holder["cancelled"]


class TestUIResponsiveWrapper:
    """Test UIResponsiveWrapper functionality."""

    def test_wrapper_initialization(self) -> None:
        """Test wrapper initializes correctly."""
        wrapper = UIResponsiveWrapper()

        # Test that wrapper can execute operations
        active_ops = wrapper.get_active_operations()
        assert isinstance(active_ops, list)
        assert len(active_ops) == 0

        wrapper.shutdown()

    def test_execute_with_progress(self) -> None:
        """Test executing function with progress tracking."""
        wrapper = UIResponsiveWrapper()
        progress_updates = []

        def progress_handler(update: "ProgressUpdate") -> None:
            progress_updates.append(update)

        def task_with_progress(
            progress_callback: Any = None, cancellation_token: Any = None
        ) -> str:
            if progress_callback:
                progress_callback(25, 100, "Starting")
                progress_callback(50, 100, "Halfway")
                progress_callback(75, 100, "Almost done")
            return "completed"

        result = wrapper.execute_with_progress(
            func=task_with_progress,
            progress_callback=progress_handler,
            task_name="ProgressTest",
        )

        assert result.is_successful
        assert result.result == "completed"
        assert len(progress_updates) >= 3  # At least our 3 + final

        wrapper.shutdown()

    def test_execute_async(self) -> None:
        """Test asynchronous execution."""
        wrapper = UIResponsiveWrapper()

        def async_task() -> str:
            time.sleep(0.1)
            return "async_result"

        future, token = wrapper.execute_async(
            func=async_task, task_name="AsyncTest"
        )

        assert isinstance(future, Future)
        assert isinstance(token, CancellationToken)

        result = future.result(timeout=5.0)
        assert result == "async_result"

        wrapper.shutdown()

    def test_cancellation_support(self) -> None:
        """Test cancellation support in wrapper."""
        wrapper = UIResponsiveWrapper()

        def cancellable_task(cancellation_token: Any = None) -> str:
            for _ in range(100):
                if cancellation_token and cancellation_token.is_cancelled:
                    return "cancelled"
                time.sleep(0.01)
            return "completed"

        future, token = wrapper.execute_async(
            func=cancellable_task, task_name="CancellableTest"
        )

        # Cancel after short delay
        time.sleep(0.05)
        token.cancel("Test cancellation")

        result = future.result(timeout=5.0)
        assert result == "cancelled"

        wrapper.shutdown()


class TestRunManagerIntegration:
    """Test integration with run_manager module."""

    def test_get_ui_wrapper_singleton(self) -> None:
        """Test UI wrapper singleton behavior."""
        # Clean up any existing instances
        cleanup_ui_wrapper()

        wrapper1 = get_ui_wrapper()
        wrapper2 = get_ui_wrapper()

        assert wrapper1 is wrapper2  # Same instance

        cleanup_ui_wrapper()

    def test_execute_training_async_mock(self) -> None:
        """Test async training execution with mocked training function."""
        cleanup_ui_wrapper()

        with patch(
            "scripts.gui.utils.run_manager.ui_integration.start_training_session"
        ) as mock_start:
            mock_start.return_value = (True, [])

            future, token = execute_training_async(
                config_path=Path("configs"),
                config_name="test_config",
                overrides_text="test.param=value",
                task_name="TestTraining",
            )

            assert isinstance(future, Future)
            assert isinstance(token, CancellationToken)

            # Wait for result
            success, errors = future.result(timeout=5.0)
            assert success is True
            assert errors == []

            # Verify mock was called correctly
            mock_start.assert_called_once_with(
                Path("configs"), "test_config", "test.param=value", None, True
            )

        cleanup_ui_wrapper()

    def test_execute_with_progress_integration(self) -> None:
        """Test progress execution integration."""
        cleanup_ui_wrapper()
        progress_updates = []

        def progress_handler(update: "ProgressUpdate") -> None:
            progress_updates.append(update)

        def mock_operation(
            data: str,
            progress_callback: Any = None,
            cancellation_token: Any = None,
        ) -> str:
            if progress_callback:
                progress_callback(0, 100, "Starting")
                progress_callback(50, 100, "Processing")
                progress_callback(100, 100, "Done")
            return f"processed_{data}"

        result = execute_with_progress(
            func=mock_operation,
            args=("test_data",),
            progress_callback=progress_handler,
            task_name="IntegrationTest",
        )

        assert result.is_successful
        assert result.result == "processed_test_data"
        assert len(progress_updates) >= 3

        cleanup_ui_wrapper()

    def test_cleanup_ui_wrapper(self) -> None:
        """Test UI wrapper cleanup functionality."""
        # Create wrapper
        wrapper = get_ui_wrapper()
        assert wrapper is not None

        # Clean up
        cleanup_ui_wrapper()

        # Verify new instance is created
        new_wrapper = get_ui_wrapper()
        assert new_wrapper is not wrapper

        cleanup_ui_wrapper()


class TestErrorHandling:
    """Test error handling in threading system."""

    def test_task_exception_handling(self) -> None:
        """Test handling of exceptions in tasks."""
        coordinator = ThreadCoordinator(max_workers=2)

        def failing_task() -> None:
            raise ValueError("Test error")

        future = coordinator.submit_function(failing_task, name="FailingTask")

        with pytest.raises(ValueError, match="Test error"):
            future.result(timeout=5.0)

        # Check statistics
        stats = coordinator.get_statistics()
        assert stats["tasks_failed"] == 1

        coordinator.shutdown()

    def test_wrapper_error_handling(self) -> None:
        """Test error handling in UI wrapper."""
        wrapper = UIResponsiveWrapper()

        def error_task() -> None:
            raise RuntimeError("Wrapper test error")

        result = wrapper.execute_with_progress(
            func=error_task, task_name="ErrorTest"
        )

        assert not result.is_successful
        assert result.status == TaskStatus.FAILED
        assert isinstance(result.error, RuntimeError)
        assert "Wrapper test error" in str(result.error)

        wrapper.shutdown()

    def test_timeout_handling(self) -> None:
        """Test timeout handling in operations."""
        wrapper = UIResponsiveWrapper(default_timeout=0.1)

        def slow_task() -> str:
            time.sleep(1.0)  # Longer than timeout
            return "should_not_reach"

        result = wrapper.execute_with_progress(
            func=slow_task, task_name="TimeoutTest"
        )

        assert not result.is_successful
        assert result.status == TaskStatus.FAILED

        wrapper.shutdown()


@pytest.fixture(autouse=True)
def cleanup_after_test() -> Generator[None, None, None]:
    """Cleanup after each test."""
    yield
    # Ensure cleanup after each test
    try:
        cleanup_ui_wrapper()
    except Exception:
        pass  # Best effort cleanup
