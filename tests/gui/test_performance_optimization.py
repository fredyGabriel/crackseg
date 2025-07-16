"""
Tests for performance optimization utilities and optimized GUI components.

This module tests CSS caching, update debouncing, memory management,
performance monitoring, and the optimized loading/progress components.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from gui.components.loading_spinner_optimized import (
    OptimizedLoadingSpinner,
    optimized_loading_spinner,
)
from gui.components.progress_bar_optimized import (
    OptimizedProgressBar,
    OptimizedStepBasedProgress,
    create_optimized_progress_bar,
    create_optimized_step_progress,
)
from gui.utils.performance_optimizer import (
    AsyncOperationManager,
    MemoryManager,
    OptimizedHTMLBuilder,
    PerformanceOptimizer,
    get_optimizer,
    inject_css_once,
    should_update,
    track_performance,
)


class TestPerformanceOptimizer:
    """Test performance optimization utilities."""

    def test_singleton_pattern(self):
        """Test that PerformanceOptimizer uses singleton pattern."""
        optimizer1 = PerformanceOptimizer()
        optimizer2 = PerformanceOptimizer()
        optimizer3 = get_optimizer()

        assert optimizer1 is optimizer2
        assert optimizer2 is optimizer3

    @patch("streamlit.markdown")
    def test_css_injection_caching(self, mock_markdown: MagicMock) -> None:
        """Test that CSS is injected only once per key."""
        optimizer = get_optimizer()
        css_content = "<style>test css</style>"

        # First injection should call st.markdown
        optimizer.inject_css_once("test_key", css_content)
        assert mock_markdown.call_count == 1

        # Second injection with same key should not call st.markdown
        optimizer.inject_css_once("test_key", css_content)
        assert mock_markdown.call_count == 1

        # Different key should call st.markdown
        optimizer.inject_css_once("different_key", css_content)
        assert mock_markdown.call_count == 2

    def test_update_debouncing(self):
        """Test update debouncing functionality."""
        optimizer = get_optimizer()
        component_id = "test_component"
        min_interval = 0.1

        # First update should be allowed
        assert optimizer.should_update(component_id, min_interval) is True

        # Immediate second update should be blocked
        assert optimizer.should_update(component_id, min_interval) is False

        # Update after interval should be allowed
        time.sleep(min_interval + 0.01)
        assert optimizer.should_update(component_id, min_interval) is True

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        optimizer = get_optimizer()
        optimizer.reset_performance_metrics()

        component_id = "test_component"
        operation = "test_operation"
        start_time = time.time()

        # Track some performance
        optimizer.track_performance(component_id, operation, start_time)

        # Get performance report
        report = optimizer.get_performance_report()

        assert component_id in report
        assert report[component_id]["count"] == 1
        assert len(report[component_id]["operations"]) == 1
        assert report[component_id]["operations"][0]["operation"] == operation

    def test_placeholder_management(self):
        """Test placeholder registration and cleanup."""
        optimizer = get_optimizer()

        # Mock placeholder
        mock_placeholder = Mock()

        # Register placeholder
        optimizer.register_placeholder(mock_placeholder)

        # Clean up placeholders
        optimizer.cleanup_placeholders()

        # Verify cleanup was called
        mock_placeholder.empty.assert_called_once()


class TestOptimizedHTMLBuilder:
    """Test optimized HTML building with template caching."""

    def test_progress_html_building(self):
        """Test optimized progress HTML building."""
        html = OptimizedHTMLBuilder.build_progress_html(
            title="Test Progress",
            progress=0.5,
            step_info="Step 5/10",
            elapsed_str="1m 30s",
            remaining_str="1m 30s",
            description="Testing progress",
            state="normal",
        )

        assert "Test Progress" in html
        assert (
            "50%" in html
        )  # Updated to handle both "50%" and "50.0%" formats
        assert "Step 5/10" in html
        assert "1m 30s" in html
        assert "Testing progress" in html
        assert "crackseg-progress-container" in html

    def test_spinner_html_building(self):
        """Test optimized spinner HTML building."""
        html = OptimizedHTMLBuilder.build_spinner_html(
            message="Loading...",
            subtext="Please wait",
            spinner_type="default",
        )

        assert "Loading..." in html
        assert "Please wait" in html
        assert "crackseg-spinner-container" in html

    def test_template_caching(self):
        """Test that HTML templates are cached for performance."""
        # Build HTML twice with same parameters
        html1 = OptimizedHTMLBuilder.build_progress_html(
            title="Test",
            progress=0.5,
            step_info="Step 1",
            elapsed_str="1s",
            remaining_str="1s",
            state="normal",
        )

        html2 = OptimizedHTMLBuilder.build_progress_html(
            title="Test2",
            progress=0.7,
            step_info="Step 2",
            elapsed_str="2s",
            remaining_str="0s",
            state="normal",
        )

        # Both should use the same template structure
        assert "crackseg-progress-container" in html1
        assert "crackseg-progress-container" in html2
        assert "Test" in html1
        assert "Test2" in html2


class TestAsyncOperationManager:
    """Test async operation management."""

    def setUp(self):
        """Reset operation manager state."""
        AsyncOperationManager._active_operations.clear()

    def test_operation_lifecycle(self):
        """Test complete operation lifecycle."""
        operation_id = "test_operation"
        title = "Test Operation"

        # Start operation
        AsyncOperationManager.start_operation(operation_id, title)
        status = AsyncOperationManager.get_operation_status(operation_id)

        assert status is not None
        assert status["title"] == title
        assert status["status"] == "running"

        # Update operation - check status is not None before accessing
        AsyncOperationManager.update_operation(operation_id, 0.5, "running")
        status = AsyncOperationManager.get_operation_status(operation_id)
        assert status is not None
        assert status["progress"] == 0.5

        # Finish operation - check status is not None before accessing
        AsyncOperationManager.finish_operation(operation_id, True)
        status = AsyncOperationManager.get_operation_status(operation_id)
        assert status is not None
        assert status["status"] == "completed"

    def test_cleanup_completed_operations(self):
        """Test cleanup of old completed operations."""
        operation_id = "old_operation"

        # Start and finish operation
        AsyncOperationManager.start_operation(operation_id, "Old Operation")
        AsyncOperationManager.finish_operation(operation_id, True)

        # Manually set old end time
        AsyncOperationManager._active_operations[operation_id]["end_time"] = (
            time.time() - 400
        )

        # Clean up
        AsyncOperationManager.cleanup_completed_operations()

        # Operation should be removed
        assert AsyncOperationManager.get_operation_status(operation_id) is None


class TestMemoryManager:
    """Test memory management utilities."""

    def setUp(self):
        """Reset memory manager state."""
        MemoryManager._memory_usage.clear()
        MemoryManager._cleanup_callbacks.clear()

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        component_id = "test_component"
        operation = "test_operation"
        memory_delta = 10.5

        MemoryManager.track_memory_usage(component_id, operation, memory_delta)

        report = MemoryManager.get_memory_report()
        assert component_id in report
        assert report[component_id]["total_allocated"] == memory_delta
        assert len(report[component_id]["operations"]) == 1

    def test_cleanup_callbacks(self):
        """Test cleanup callback registration and execution."""
        component_id = "test_component"
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        # Register callback
        MemoryManager.register_cleanup_callback(component_id, test_callback)

        # Clean up component
        MemoryManager.cleanup_component(component_id)

        # Verify callback was called
        assert callback_called


class TestOptimizedLoadingSpinner:
    """Test optimized loading spinner component."""

    @patch("streamlit.markdown")
    @patch("streamlit.spinner")
    def test_spinner_context_manager(
        self, mock_spinner: MagicMock, mock_markdown: MagicMock
    ) -> None:
        """Test optimized spinner context manager."""
        with OptimizedLoadingSpinner.spinner("Test loading..."):
            pass

        # Verify spinner was called
        mock_spinner.assert_called_once_with("Test loading...")

    def test_contextual_messages(self):
        """Test contextual message generation."""
        message, subtext, spinner_type = (
            OptimizedLoadingSpinner.get_contextual_message("config")
        )

        assert "Loading configuration" in message
        assert subtext is not None
        assert spinner_type == "default"

    def test_error_classification(self):
        """Test optimized error classification."""
        from gui.utils.error_state import ErrorType

        # Test CUDA error
        cuda_error = Exception("CUDA out of memory")
        error_type = OptimizedLoadingSpinner._classify_error(cuda_error)
        assert error_type == ErrorType.VRAM_EXHAUSTED

        # Test file not found
        file_error = FileNotFoundError("config.yaml not found")
        error_type = OptimizedLoadingSpinner._classify_error(file_error)
        assert error_type == ErrorType.CONFIG_NOT_FOUND

    @patch("streamlit.markdown")
    def test_css_injection_optimization(
        self, mock_markdown: MagicMock
    ) -> None:
        """Test that CSS is injected efficiently."""
        # Create multiple spinners
        OptimizedLoadingSpinner._ensure_css_injected()
        OptimizedLoadingSpinner._ensure_css_injected()
        OptimizedLoadingSpinner._ensure_css_injected()

        # CSS should only be injected once due to caching
        assert mock_markdown.call_count <= 1

    @patch("streamlit.columns")
    @patch("streamlit.progress")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_progress_with_spinner(
        self,
        mock_caption: MagicMock,
        mock_markdown: MagicMock,
        mock_progress: MagicMock,
        mock_columns: MagicMock,
    ) -> None:
        """Test progress with spinner display."""
        # Mock columns that support context manager protocol
        col1 = MagicMock()
        col2 = MagicMock()
        col1.__enter__ = Mock(return_value=col1)
        col1.__exit__ = Mock(return_value=None)
        col2.__enter__ = Mock(return_value=col2)
        col2.__exit__ = Mock(return_value=None)
        mock_columns.return_value = [col1, col2]

        OptimizedLoadingSpinner.show_progress_with_spinner(
            message="Test progress",
            progress=0.5,
            subtext="Testing...",
        )

        # Verify components were called
        mock_columns.assert_called_once()
        mock_progress.assert_called_once_with(0.5)


class TestOptimizedProgressBar:
    """Test optimized progress bar component."""

    def test_progress_bar_lifecycle(self):
        """Test complete progress bar lifecycle."""
        progress_bar = OptimizedProgressBar("test_progress")

        # Start progress
        progress_bar.start("Test Operation", total_steps=10)
        assert progress_bar._is_active is True
        assert progress_bar.title == "Test Operation"
        assert progress_bar.total_steps == 10

        # Update progress
        progress_bar.update(
            0.5, current_step=5, step_description="Halfway done"
        )
        assert progress_bar.current_step == 5

        # Finish progress
        progress_bar.finish("Completed successfully")
        assert progress_bar._is_active is False

    def test_time_formatting(self):
        """Test efficient time formatting."""
        # Test seconds
        assert OptimizedProgressBar._format_time(30) == "30s"

        # Test minutes
        assert OptimizedProgressBar._format_time(90) == "1m 30s"

        # Test hours
        assert OptimizedProgressBar._format_time(3661) == "1h 1m"

    @patch("streamlit.empty")
    def test_placeholder_management(self, mock_empty: MagicMock) -> None:
        """Test placeholder creation and cleanup."""
        progress_bar = OptimizedProgressBar("test_progress")

        # Start should create placeholder
        progress_bar.start("Test")
        assert progress_bar._placeholder is not None

        # Cleanup should clear placeholder
        progress_bar._cleanup_resources()
        assert progress_bar._is_active is False


class TestOptimizedStepBasedProgress:
    """Test optimized step-based progress tracker."""

    def test_step_progress_lifecycle(self):
        """Test step-based progress lifecycle."""
        steps = ["Step 1", "Step 2", "Step 3"]

        with OptimizedStepBasedProgress("Test Operation", steps) as progress:
            assert progress.current_step_index == 0

            # Advance through steps
            progress.next_step()
            assert progress.current_step_index == 1

            progress.next_step("Custom step description")
            assert progress.current_step_index == 2

    def test_step_progress_setting(self):
        """Test setting progress for specific steps."""
        steps = ["Step 1", "Step 2", "Step 3"]

        with OptimizedStepBasedProgress("Test Operation", steps) as progress:
            # Set progress for specific step
            progress.set_step_progress(0, 0.5)
            progress.set_step_progress(1, 1.0)

    def test_error_handling_in_step_progress(self):
        """Test error handling in step-based progress."""
        steps = ["Step 1", "Step 2"]

        with pytest.raises(ValueError):
            with OptimizedStepBasedProgress(
                "Test Operation", steps
            ) as progress:
                progress.next_step()
                raise ValueError("Test error")


class TestConvenienceFunctions:
    """Test convenience functions for easy component creation."""

    def test_optimized_loading_spinner_function(self):
        """Test optimized loading spinner convenience function."""
        spinner_context = optimized_loading_spinner(
            message="Test loading",
            timeout_seconds=10,
        )

        assert spinner_context is not None

    def test_create_optimized_progress_bar(self):
        """Test optimized progress bar creation."""
        progress_bar = create_optimized_progress_bar("test_id")

        assert isinstance(progress_bar, OptimizedProgressBar)
        assert progress_bar.operation_id == "test_id"

    def test_create_optimized_step_progress(self):
        """Test optimized step progress creation."""
        steps = ["Step 1", "Step 2", "Step 3"]
        step_progress = create_optimized_step_progress("Test", steps)

        assert isinstance(step_progress, OptimizedStepBasedProgress)
        assert step_progress.steps == steps
        assert step_progress.title == "Test"


class TestPerformanceImprovements:
    """Test performance improvements and benchmarks."""

    def test_css_caching_performance(self):
        """Test that CSS caching provides performance benefits."""
        # Measure time for multiple CSS injections with caching
        start_time = time.time()

        for _ in range(100):
            inject_css_once("test_css", "<style>test</style>")

        cached_time = time.time() - start_time

        # Performance should be much better with caching
        # (Hard to test precisely, but should be very fast)
        assert cached_time < 1.0  # Should complete in less than 1 second

    def test_update_debouncing_performance(self):
        """Test that update debouncing prevents excessive calls."""
        component_id = "performance_test"
        allowed_updates = 0

        # Try to update 100 times rapidly
        for _ in range(100):
            if should_update(component_id, 0.1):
                allowed_updates += 1

        # Should be debounced to much fewer updates
        assert allowed_updates < 10

    def test_performance_tracking_overhead(self):
        """Test that performance tracking has minimal overhead."""
        component_id = "overhead_test"

        # Measure time for performance tracking itself
        start_time = time.time()

        for i in range(1000):
            track_performance(component_id, f"operation_{i}", time.time())

        tracking_time = time.time() - start_time

        # Performance tracking should be very fast
        assert tracking_time < 1.0  # Should complete in less than 1 second


if __name__ == "__main__":
    pytest.main([__file__])
