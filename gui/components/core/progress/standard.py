"""
Progress bar component for long-running operations in the CrackSeg application.

This module provides advanced progress indicators with step-based tracking,
time estimation, and contextual messaging for operations expected to take
longer than 10 seconds. Complements the LoadingSpinner for shorter operations.
"""

import time
from typing import Any

import streamlit as st

from gui.utils.error_state import (
    ErrorMessageFactory,
    ErrorType,
    StandardErrorState,
)
from gui.utils.session_state import SessionStateManager


class ProgressBar:
    """Advanced progress bar component for long-running operations."""

    # Brand-aligned color scheme matching LoadingSpinner
    _BRAND_COLORS = {
        "primary": "#2E2E2E",
        "secondary": "#F0F0F0",
        "accent": "#FF4444",
        "success": "#00FF64",
        "warning": "#FFB800",
        "error": "#FF4444",
    }

    def __init__(self, operation_id: str | None = None) -> None:
        """
        Initialize progress bar instance.

        Args:
            operation_id: Unique identifier for the operation (optional)
        """
        self.operation_id = operation_id or f"progress_{int(time.time())}"
        self.start_time: float | None = None
        self.last_update_time: float | None = None
        self._placeholder: Any = None
        self._is_active = False

        # Initialize attributes that might be accessed before start()
        self.total_steps: int | None = None
        self.current_step: int = 0
        self.title: str = ""
        self.description: str | None = None

    @staticmethod
    def _inject_custom_css() -> None:
        """Inject custom CSS for brand-aligned progress bar styling."""
        css = """
        <style>
        /* CrackSeg Progress Bar Styles */
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

        /* Success state */
        .crackseg-progress-success .crackseg-progress-bar-fill {
            background: linear-gradient(90deg, #00FF64 0%, #28a745 100%);
        }

        .crackseg-progress-success .crackseg-progress-percentage {
            color: #00FF64;
            border-color: #00FF64;
        }

        /* Warning state */
        .crackseg-progress-warning .crackseg-progress-bar-fill {
            background: linear-gradient(90deg, #FFB800 0%, #ffc107 100%);
        }

        .crackseg-progress-warning .crackseg-progress-percentage {
            color: #FFB800;
            border-color: #FFB800;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    def start(
        self,
        title: str,
        total_steps: int | None = None,
        description: str | None = None,
    ) -> None:
        """
        Start the progress bar operation.

        Args:
            title: Main title for the operation
            total_steps: Total number of steps (None for indeterminate)
            description: Optional description of what's being processed
        """
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self._is_active = True
        self.total_steps = total_steps
        self.current_step = 0
        self.title = title
        self.description = description

        # Create placeholder for updates
        self._placeholder = st.empty()

        # Update session state
        session_state = SessionStateManager.get()
        session_state.add_notification(f"Started: {title}")

        # Initial render
        self._render_progress(0.0, "Initializing...")

    def update(
        self,
        progress: float,
        current_step: int | None = None,
        step_description: str | None = None,
    ) -> None:
        """
        Update progress bar state.

        Args:
            progress: Progress value between 0.0 and 1.0
            current_step: Current step number (optional)
            step_description: Description of current step
        """
        if not self._is_active:
            return

        # Clamp progress to valid range
        progress = max(0.0, min(1.0, progress))

        if current_step is not None:
            self.current_step = current_step

        self.last_update_time = time.time()
        self._render_progress(progress, step_description)

    def finish(self, success_message: str | None = None) -> None:
        """
        Complete the progress bar operation.

        Args:
            success_message: Optional success message to display
        """
        if not self._is_active:
            return

        elapsed_time = time.time() - (self.start_time or time.time())

        # Show completion state
        self._render_progress(
            1.0,
            success_message or "Operation completed successfully",
            state="success",
        )

        # Update session state
        session_state = SessionStateManager.get()
        session_state.add_notification(
            f"Completed: {self.title} ({elapsed_time:.1f}s)"
        )

        self._is_active = False

        # Brief pause to show completion
        time.sleep(1)

    def _render_progress(
        self,
        progress: float,
        step_description: str | None = None,
        state: str = "normal",
    ) -> None:
        """
        Render the progress bar UI.

        Args:
            progress: Current progress (0.0 to 1.0)
            step_description: Current step description
            state: Progress state ('normal', 'success', 'warning')
        """
        if not self._placeholder:
            return

        self._inject_custom_css()

        # Calculate time estimates
        elapsed_time = time.time() - (self.start_time or time.time())

        if progress > 0.01:  # Avoid division by zero
            estimated_total = elapsed_time / progress
            remaining_time = max(0, estimated_total - elapsed_time)
        else:
            remaining_time = 0

        # Format times
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = (
            self._format_time(remaining_time)
            if remaining_time > 0
            else "calculating..."
        )

        # Build step info
        step_info = ""
        if self.total_steps is not None:
            step_info = f"Step {self.current_step}/{self.total_steps}"
        elif step_description:
            step_info = step_description

        # Build description
        description_html = ""
        if self.description or step_description:
            desc_text = step_description or self.description or ""
            description_html = (
                f'<div class="crackseg-progress-description">{desc_text}</div>'
            )

        # State-specific CSS classes
        container_class = (
            f"crackseg-progress-container crackseg-progress-{state}"
        )

        # Render HTML
        progress_percentage = f"{progress:.1%}"
        progress_width = f"{progress * 100}%"

        # Build HTML components
        title_span = f'<h4 class="crackseg-progress-title">{self.title}</h4>'
        percentage_span = (
            f'<span class="crackseg-progress-percentage">'
            f"{progress_percentage}</span>"
        )
        fill_div = (
            f'<div class="crackseg-progress-bar-fill" '
            f'style="width: {progress_width};"></div>'
        )

        header_html = f"""
            <div class="crackseg-progress-header">
                {title_span}
                {percentage_span}
            </div>"""

        bar_html = f"""
            <div class="crackseg-progress-bar-container">
                {fill_div}
            </div>"""

        info_html = f"""
            <div class="crackseg-progress-info">
                <span class="crackseg-progress-step">{step_info}</span>
                <span class="crackseg-progress-time">
                    Elapsed: {elapsed_str} | Remaining: ~{remaining_str}
                </span>
            </div>"""

        progress_html = f"""
        <div class="{container_class}">
            {header_html}
            {bar_html}
            {info_html}
            {description_html}
        </div>
        """

        self._placeholder.markdown(progress_html, unsafe_allow_html=True)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format time duration in human-readable format.

        Args:
            seconds: Time duration in seconds

        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"

    @staticmethod
    def create_step_based_progress(
        title: str, steps: list[str], operation_id: str | None = None
    ) -> "StepBasedProgress":
        """
        Create a step-based progress tracker.

        Args:
            title: Main operation title
            steps: List of step descriptions
            operation_id: Unique operation identifier

        Returns:
            StepBasedProgress instance
        """
        return StepBasedProgress(title, steps, operation_id)


class StepBasedProgress:
    """Step-based progress tracker for multi-step operations."""

    def __init__(
        self, title: str, steps: list[str], operation_id: str | None = None
    ) -> None:
        """
        Initialize step-based progress tracker.

        Args:
            title: Main operation title
            steps: List of step descriptions
            operation_id: Unique operation identifier
        """
        self.progress_bar = ProgressBar(operation_id)
        self.steps = steps
        self.current_step_index = 0
        self.title = title

    def __enter__(self) -> "StepBasedProgress":
        """Context manager entry."""
        self.progress_bar.start(
            title=self.title,
            total_steps=len(self.steps),
            description="Multi-step operation in progress",
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with enhanced error handling."""
        if exc_type is None:
            self.progress_bar.finish("All steps completed successfully")
        else:
            # Enhanced error handling for step-based operations
            self.progress_bar.finish("Operation encountered an error")

            # Show detailed error information if exception occurred
            if exc_val is not None:
                error_state = StandardErrorState("ProgressBar")

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
                    },
                )

                # Show enhanced error message
                error_state.show_error(error_info)

                # Show retry option for step-based operations
                if error_info.retry_possible:
                    st.info(
                        "💡 Consider restarting the operation from the "
                        "beginning."
                    )

    def _classify_error(self, exception: Exception) -> ErrorType:
        """
        Classify exception into appropriate ErrorType for progress operations.

        Args:
            exception: Exception that occurred

        Returns:
            Appropriate ErrorType for the exception
        """
        # Check exception type and message for classification
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__

        if (
            "cuda" in exception_str
            or "vram" in exception_str
            or "memory" in exception_str
        ):
            if "out of memory" in exception_str:
                return ErrorType.VRAM_EXHAUSTED
            return ErrorType.MEMORY_INSUFFICIENT

        if exception_type in ("FileNotFoundError", "OSError"):
            if "config" in exception_str:
                return ErrorType.CONFIG_NOT_FOUND
            return ErrorType.DATA_LOADING

        if exception_type in ("ValueError", "TypeError"):
            if "config" in exception_str:
                return ErrorType.CONFIG_INVALID
            return ErrorType.MODEL_INSTANTIATION

        if exception_type in ("TimeoutError", "asyncio.TimeoutError"):
            return ErrorType.TIMEOUT

        if exception_type in ("PermissionError",):
            return ErrorType.PERMISSION_DENIED

        if "training" in exception_str or "train" in exception_str:
            return ErrorType.TRAINING_FAILED

        # Default to unexpected for unclassified errors
        return ErrorType.UNEXPECTED

    def next_step(self, custom_description: str | None = None) -> None:
        """
        Advance to the next step.

        Args:
            custom_description: Optional custom description for this step
        """
        if self.current_step_index < len(self.steps):
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

    def set_step_progress(
        self, step_index: int, step_progress: float = 1.0
    ) -> None:
        """
        Set progress for a specific step.

        Args:
            step_index: Index of the step (0-based)
            step_progress: Progress within the step (0.0 to 1.0)
        """
        if 0 <= step_index < len(self.steps):
            overall_progress = (step_index + step_progress) / len(self.steps)
            step_desc = self.steps[step_index]

            self.progress_bar.update(
                progress=overall_progress,
                current_step=step_index + 1,
                step_description=step_desc,
            )


# Convenience functions for easy import
def create_progress_bar(operation_id: str | None = None) -> ProgressBar:
    """
    Create a new progress bar instance.

    Args:
        operation_id: Unique operation identifier

    Returns:
        ProgressBar instance
    """
    return ProgressBar(operation_id)


def create_step_progress(
    title: str, steps: list[str], operation_id: str | None = None
) -> StepBasedProgress:
    """
    Create a step-based progress tracker.

    Args:
        title: Main operation title
        steps: List of step descriptions
        operation_id: Unique operation identifier

    Returns:
        StepBasedProgress instance
    """
    return StepBasedProgress(title, steps, operation_id)
