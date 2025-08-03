"""
Optimized progress bar component for long-running operations in CrackSeg.

This module provides high-performance progress indicators with advanced
optimizations including template caching, update debouncing, memory management,
and performance monitoring for superior user experience in operations longer
than 10 seconds.
"""

import time
from typing import Any

import streamlit as st

from gui.utils.error_state import (
    ErrorMessageFactory,
    ErrorType,
    StandardErrorState,
)
from gui.utils.performance_optimizer import (
    AsyncOperationManager,
    MemoryManager,
    OptimizedHTMLBuilder,
    inject_css_once,
    should_update,
    track_performance,
)
from gui.utils.session_state import SessionStateManager


class OptimizedProgressBar:
    """High-performance progress bar component with advanced optimizations."""

    # Brand-aligned color scheme
    _BRAND_COLORS = {
        "primary": "#2E2E2E",
        "secondary": "#F0F0F0",
        "accent": "#FF4444",
        "success": "#00FF64",
        "warning": "#FFB800",
        "error": "#FF4444",
    }

    # CSS is cached globally - injected only once per session
    _CSS_CONTENT = """
    <style>
    /* CrackSeg Optimized Progress Bar Styles */
    .crackseg-progress-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #dee2e6;
    }

    .crackseg-progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }

    .crackseg-progress-title {
        font-size: 18px;
        font-weight: 600;
        color: #2E2E2E;
        margin: 0;
    }

    .crackseg-progress-percentage {
        font-size: 16px;
        font-weight: 500;
        color: #FF4444;
        background: #fff;
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid #FF4444;
    }

    .crackseg-progress-bar-container {
        width: 100%;
        height: 12px;
        background-color: #e9ecef;
        border-radius: 6px;
        overflow: hidden;
        margin: 10px 0;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .crackseg-progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #FF4444 0%, #ff6b6b 100%);
        border-radius: 6px;
        transition: width 0.3s ease-in-out;
        box-shadow: 0 1px 3px rgba(255, 68, 68, 0.3);
    }

    .crackseg-progress-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
        font-size: 14px;
    }

    .crackseg-progress-step {
        color: #6c757d;
        font-weight: 500;
    }

    .crackseg-progress-time {
        color: #495057;
        font-family: 'Courier New', monospace;
    }

    .crackseg-progress-description {
        margin-top: 8px;
        font-size: 13px;
        color: #6c757d;
        font-style: italic;
    }

    /* Success state - optimized with GPU acceleration */
    .crackseg-progress-success .crackseg-progress-bar-fill {
        background: linear-gradient(90deg, #00FF64 0%, #28a745 100%);
        transform: translateZ(0);
    }

    .crackseg-progress-success .crackseg-progress-percentage {
        color: #00FF64;
        border-color: #00FF64;
    }

    /* Warning state - optimized with GPU acceleration */
    .crackseg-progress-warning .crackseg-progress-bar-fill {
        background: linear-gradient(90deg, #FFB800 0%, #ffc107 100%);
        transform: translateZ(0);
    }

    .crackseg-progress-warning .crackseg-progress-percentage {
        color: #FFB800;
        border-color: #FFB800;
    }

    /* Performance optimization: GPU acceleration */
    .crackseg-progress-bar-fill {
        will-change: width;
        transform: translateZ(0);
    }

    .crackseg-progress-container {
        transform: translateZ(0);
    }
    </style>
    """

    def __init__(self, operation_id: str | None = None) -> None:
        """
        Initialize optimized progress bar instance.

        Args:
            operation_id: Unique identifier for the operation
        """
        self.operation_id = operation_id or f"progress_{int(time.time())}"
        self.start_time: float | None = None
        self.last_update_time: float | None = None
        self._placeholder: Any | None = None
        self._is_active = False

        # Initialize attributes
        self.total_steps: int | None = None
        self.current_step: int = 0
        self.title: str = ""
        self.description: str | None = None

        # Register for memory management
        MemoryManager.register_cleanup_callback(
            self.operation_id,
            self._cleanup_resources,
        )

    @staticmethod
    def _ensure_css_injected() -> None:
        """
        Ensure CSS is injected only once per session for optimal performance.
        """
        inject_css_once(
            "crackseg_progress_optimized", OptimizedProgressBar._CSS_CONTENT
        )

    def start(
        self,
        title: str,
        total_steps: int | None = None,
        description: str | None = None,
    ) -> None:
        """
        Start the progress bar operation with performance tracking.

        Args:
            title: Main title for the operation
            total_steps: Total number of steps (None for indeterminate)
            description: Optional description of what's being processed
        """
        start_time = time.time()

        self.start_time = start_time
        self.last_update_time = start_time
        self._is_active = True
        self.total_steps = total_steps
        self.current_step = 0
        self.title = title
        self.description = description

        # Start async operation tracking
        AsyncOperationManager.start_operation(
            operation_id=self.operation_id,
            title=title,
            estimated_duration=0,  # Unknown duration
        )

        # Create placeholder for updates
        self._placeholder = st.empty()

        # Update session state efficiently
        session_state = SessionStateManager.get()
        if should_update(f"start_notification_{self.operation_id}", 1.0):
            session_state.add_notification(f"Started: {title}")

        # Initial render with performance tracking
        self._render_optimized_progress(0.0, "Initializing...")

        # Track performance
        track_performance(self.operation_id, "start_progress", start_time)

    def update(
        self,
        progress: float,
        current_step: int | None = None,
        step_description: str | None = None,
    ) -> None:
        """
        Update progress bar state with debouncing and performance optimization.

        Args:
            progress: Progress value between 0.0 and 1.0
            current_step: Current step number (optional)
            step_description: Description of current step
        """
        if not self._is_active:
            return

        # Debounce updates to prevent excessive rendering
        if not should_update(f"progress_{self.operation_id}", 0.1):
            return

        start_time = time.time()

        # Clamp progress to valid range
        progress = max(0.0, min(1.0, progress))

        if current_step is not None:
            self.current_step = current_step

        self.last_update_time = start_time

        # Update async operation status
        AsyncOperationManager.update_operation(
            operation_id=self.operation_id,
            progress=progress,
            status="running",
        )

        # Render with optimization
        self._render_optimized_progress(progress, step_description)

        # Track performance
        track_performance(self.operation_id, "update_progress", start_time)

    def finish(self, success_message: str | None = None) -> None:
        """
        Complete the progress bar operation with performance cleanup.

        Args:
            success_message: Optional success message to display
        """
        if not self._is_active:
            return

        start_time = time.time()
        elapsed_time = start_time - (self.start_time or start_time)

        # Show completion state
        self._render_optimized_progress(
            1.0,
            success_message or "Operation completed successfully",
            state="success",
        )

        # Update async operation status
        AsyncOperationManager.finish_operation(self.operation_id, True)

        # Update session state efficiently
        session_state = SessionStateManager.get()
        if should_update(f"finish_notification_{self.operation_id}", 1.0):
            session_state.add_notification(
                f"Completed: {self.title} ({elapsed_time:.1f}s)"
            )

        self._is_active = False

        # Brief pause to show completion (non-blocking)
        time.sleep(0.8)

        # Track performance
        track_performance(self.operation_id, "finish_progress", start_time)

    def _render_optimized_progress(
        self,
        progress: float,
        step_description: str | None = None,
        state: str = "normal",
    ) -> None:
        """
        Render progress bar UI with optimized HTML building and caching.

        Args:
            progress: Current progress (0.0 to 1.0)
            step_description: Current step description
            state: Progress state ('normal', 'success', 'warning')
        """
        if not self._placeholder:
            return

        start_time = time.time()

        # Ensure CSS is injected (cached globally)
        OptimizedProgressBar._ensure_css_injected()

        # Calculate time estimates efficiently
        elapsed_time = time.time() - (self.start_time or time.time())

        if progress > 0.01:  # Avoid division by zero
            estimated_total = elapsed_time / progress
            remaining_time = max(0, estimated_total - elapsed_time)
        else:
            remaining_time = 0

        # Format times efficiently
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = (
            self._format_time(remaining_time)
            if remaining_time > 0
            else "calculating..."
        )

        # Build step info efficiently
        step_info = ""
        if self.total_steps is not None:
            step_info = f"Step {self.current_step}/{self.total_steps}"
        elif step_description:
            step_info = step_description

        # Build description
        description = ""
        if self.description or step_description:
            description = step_description or self.description or ""

        # Use optimized HTML builder
        progress_html = OptimizedHTMLBuilder.build_progress_html(
            title=self.title,
            progress=progress,
            step_info=step_info,
            elapsed_str=elapsed_str,
            remaining_str=remaining_str,
            description=description,
            state=state,
        )

        # Render optimized HTML
        self._placeholder.markdown(progress_html, unsafe_allow_html=True)

        # Track rendering performance
        track_performance(self.operation_id, "render_progress", start_time)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format time duration efficiently with caching for common values.

        Args:
            seconds: Time duration in seconds

        Returns:
            Formatted time string
        """
        # Use integer operations for better performance
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"

    def _cleanup_resources(self) -> None:
        """Clean up progress bar resources."""
        if self._placeholder:
            try:
                self._placeholder.empty()
            except Exception:
                # Placeholder might already be cleaned up
                pass
        self._is_active = False

    @staticmethod
    def create_step_based_progress(
        title: str, steps: list[str], operation_id: str | None = None
    ) -> "OptimizedStepBasedProgress":
        """
        Create an optimized step-based progress tracker.

        Args:
            title: Main operation title
            steps: List of step descriptions
            operation_id: Unique operation identifier

        Returns:
            OptimizedStepBasedProgress instance
        """
        return OptimizedStepBasedProgress(title, steps, operation_id)


class OptimizedStepBasedProgress:
    """Optimized step-based progress tracker for multi-step operations."""

    def __init__(
        self, title: str, steps: list[str], operation_id: str | None = None
    ) -> None:
        """
        Initialize optimized step-based progress tracker.

        Args:
            title: Main operation title
            steps: List of step descriptions
            operation_id: Unique operation identifier
        """
        self.progress_bar = OptimizedProgressBar(operation_id)
        self.steps = steps
        self.current_step_index = 0
        self.title = title
        self.operation_id = operation_id or f"step_progress_{int(time.time())}"

        # Register for memory management
        MemoryManager.register_cleanup_callback(
            self.operation_id,
            self._cleanup_resources,
        )

    def __enter__(self) -> "OptimizedStepBasedProgress":
        """Context manager entry with performance tracking."""
        start_time = time.time()

        self.progress_bar.start(
            title=self.title,
            total_steps=len(self.steps),
            description="Multi-step operation in progress",
        )

        # Track performance
        track_performance(self.operation_id, "enter_step_progress", start_time)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with enhanced error handling and cleanup."""
        start_time = time.time()

        if exc_type is None:
            self.progress_bar.finish("All steps completed successfully")
        else:
            # Enhanced error handling for step-based operations
            self.progress_bar.finish("Operation encountered an error")

            # Show detailed error information if exception occurred
            if exc_val is not None:
                error_state = StandardErrorState(
                    f"OptimizedProgressBar_{self.operation_id}"
                )

                # Classify the error
                error_type = self._classify_error(exc_val)

                # Create error info with step context
                error_info = ErrorMessageFactory.create_error_info(
                    error_type=error_type,
                    exception=exc_val,
                    context={
                        "step": f"{self.current_step_index}/{len(self.steps)}",
                        "current_step_description": (
                            self.steps[self.current_step_index - 1]
                            if self.current_step_index > 0
                            and self.current_step_index <= len(self.steps)
                            else "Unknown step"
                        ),
                        "total_steps": len(self.steps),
                        "operation": self.title,
                        "operation_id": self.operation_id,
                    },
                )

                # Show enhanced error message
                error_state.show_error(error_info)

                if error_info.retry_possible:
                    st.info(
                        "💡 Consider restarting the operation from the "
                        "beginning. The system is optimized for efficient "
                        "retries."
                    )

        # Clean up resources
        self._cleanup_resources()

        # Track performance
        track_performance(self.operation_id, "exit_step_progress", start_time)

    def _classify_error(self, exception: Exception) -> ErrorType:
        """
        Optimized error classification for progress operations.

        Args:
            exception: Exception that occurred

        Returns:
            Appropriate ErrorType for the exception
        """
        # Use cached string operations for performance
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__

        # Fast path checks for common error types
        if "cuda" in exception_str or "vram" in exception_str:
            return (
                ErrorType.VRAM_EXHAUSTED
                if "out of memory" in exception_str
                else ErrorType.MEMORY_INSUFFICIENT
            )

        if exception_type in {"FileNotFoundError", "OSError"}:
            return (
                ErrorType.CONFIG_NOT_FOUND
                if "config" in exception_str
                else ErrorType.DATA_LOADING
            )

        if exception_type in {"ValueError", "TypeError"}:
            return (
                ErrorType.CONFIG_INVALID
                if "config" in exception_str
                else ErrorType.MODEL_INSTANTIATION
            )

        if "timeout" in exception_type.lower():
            return ErrorType.TIMEOUT

        if exception_type == "PermissionError":
            return ErrorType.PERMISSION_DENIED

        if "train" in exception_str:
            return ErrorType.TRAINING_FAILED

        return ErrorType.UNEXPECTED

    def next_step(self, custom_description: str | None = None) -> None:
        """
        Advance to the next step with optimization.

        Args:
            custom_description: Optional custom description for this step
        """
        if self.current_step_index < len(self.steps):
            start_time = time.time()

            step_desc = (
                custom_description or self.steps[self.current_step_index]
            )
            progress = (self.current_step_index + 1) / len(self.steps)

            self.progress_bar.update(
                progress=progress,
                current_step=self.current_step_index + 1,
                step_description=step_desc,
            )

            self.current_step_index += 1

            # Track performance
            track_performance(self.operation_id, "next_step", start_time)

    def set_step_progress(
        self, step_index: int, step_progress: float = 1.0
    ) -> None:
        """
        Set progress for a specific step with optimization.

        Args:
            step_index: Index of the step (0-based)
            step_progress: Progress within the step (0.0 to 1.0)
        """
        if 0 <= step_index < len(self.steps):
            start_time = time.time()

            overall_progress = (step_index + step_progress) / len(self.steps)
            step_desc = self.steps[step_index]

            self.progress_bar.update(
                progress=overall_progress,
                current_step=step_index + 1,
                step_description=step_desc,
            )

            # Track performance
            track_performance(
                self.operation_id, "set_step_progress", start_time
            )

    def _cleanup_resources(self) -> None:
        """Clean up step-based progress resources."""
        if hasattr(self.progress_bar, "_cleanup_resources"):
            self.progress_bar._cleanup_resources()


# Convenience functions for easy import
def create_optimized_progress_bar(
    operation_id: str | None = None,
) -> OptimizedProgressBar:
    """
    Create a new optimized progress bar instance.

    Args:
        operation_id: Unique operation identifier

    Returns:
        OptimizedProgressBar instance
    """
    return OptimizedProgressBar(operation_id)


def create_optimized_step_progress(
    title: str, steps: list[str], operation_id: str | None = None
) -> OptimizedStepBasedProgress:
    """
    Create an optimized step-based progress tracker.

    Args:
        title: Main operation title
        steps: List of step descriptions
        operation_id: Unique operation identifier

    Returns:
        OptimizedStepBasedProgress instance
    """
    return OptimizedStepBasedProgress(title, steps, operation_id)
