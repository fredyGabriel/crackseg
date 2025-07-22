"""
Error state management system for GUI components.

This module provides a unified error handling interface with user-friendly
messages, retry mechanisms, and context-aware error states for LoadingSpinner,
ProgressBar, and other GUI components.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import streamlit as st

from gui.utils.session_state import SessionStateManager


class ErrorType(Enum):
    """Standard error types for GUI components."""

    # Configuration errors
    CONFIG_INVALID = "config_invalid"
    CONFIG_NOT_FOUND = "config_not_found"
    CONFIG_PARSING = "config_parsing"

    # Model errors
    MODEL_INSTANTIATION = "model_instantiation"
    MODEL_LOADING = "model_loading"
    MODEL_ARCHITECTURE = "model_architecture"

    # Resource errors
    VRAM_EXHAUSTED = "vram_exhausted"
    MEMORY_INSUFFICIENT = "memory_insufficient"
    DISK_SPACE = "disk_space"

    # Operation errors
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    PERMISSION_DENIED = "permission_denied"

    # Training errors
    TRAINING_FAILED = "training_failed"
    DATA_LOADING = "data_loading"
    CHECKPOINT_CORRUPT = "checkpoint_corrupt"

    # General errors
    UNEXPECTED = "unexpected"
    VALIDATION_FAILED = "validation_failed"
    OPERATION_CANCELLED = "operation_cancelled"


@dataclass
class ErrorInfo:
    """Information about an error that occurred."""

    error_type: ErrorType
    title: str
    message: str
    details: str | None = None
    technical_info: str | None = None
    recovery_suggestions: list[str] | None = None
    retry_possible: bool = True

    def __post_init__(self) -> None:
        """Validate error info after initialization."""
        if not self.title.strip():
            raise ValueError("Error title cannot be empty")
        if not self.message.strip():
            raise ValueError("Error message cannot be empty")


class ErrorState(Protocol):
    """Protocol for error state management in GUI components."""

    def show_error(self, error_info: ErrorInfo) -> None:
        """Display error to user with appropriate styling."""
        ...

    def show_retry_option(
        self, retry_callback: Callable[[], None], retry_text: str = "Retry"
    ) -> None:
        """Display retry option to user."""
        ...

    def clear_error(self) -> None:
        """Clear current error state."""
        ...


class ErrorMessageFactory:
    """Factory for creating user-friendly error messages."""

    # Brand colors matching LoadingSpinner
    _BRAND_COLORS = {
        "error": "#FF4444",
        "warning": "#FFB800",
        "info": "#17a2b8",
        "success": "#00FF64",
    }

    @staticmethod
    def create_error_info(
        error_type: ErrorType,
        exception: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> ErrorInfo:
        """
        Create ErrorInfo from error type and exception.

        Args:
            error_type: Type of error that occurred
            exception: Original exception (if any)
            context: Additional context information

        Returns:
            ErrorInfo with user-friendly messages and recovery suggestions
        """
        context = context or {}

        # Get base error template
        error_template = ErrorMessageFactory._get_error_template(error_type)

        # Fill in context-specific details
        title = error_template["title"]
        message = error_template["message"]

        # Add technical details if exception is provided
        technical_info = str(exception) if exception else None

        # Add context-specific information
        details = error_template.get("details", "")
        if context:
            details += f"\n\nContext: {context}"

        return ErrorInfo(
            error_type=error_type,
            title=title,
            message=message,
            details=details.strip() if details.strip() else None,
            technical_info=technical_info,
            recovery_suggestions=error_template.get(
                "recovery_suggestions", []
            ),
            retry_possible=error_template.get("retry_possible", True),
        )

    @staticmethod
    def _get_error_template(error_type: ErrorType) -> dict[str, Any]:
        """Get error message template for given error type."""
        templates = {
            ErrorType.CONFIG_INVALID: {
                "title": "Configuration Error",
                "message": "The configuration file contains invalid settings.",
                "details": (
                    "Please check the configuration file for syntax errors "
                    "or missing required fields."
                ),
                "recovery_suggestions": [
                    "Verify YAML syntax is correct",
                    "Check all required fields are present",
                    "Use the configuration validator",
                    "Restore from a backup configuration",
                ],
                "retry_possible": True,
            },
            ErrorType.CONFIG_NOT_FOUND: {
                "title": "Configuration File Not Found",
                "message": (
                    "The specified configuration file could not be located."
                ),
                "details": (
                    "Make sure the file path is correct and the file exists."
                ),
                "recovery_suggestions": [
                    "Check the file path is correct",
                    "Verify the file exists",
                    "Use the file browser to select the configuration",
                    "Create a new configuration file",
                ],
                "retry_possible": True,
            },
            ErrorType.MODEL_INSTANTIATION: {
                "title": "Model Creation Failed",
                "message": "Unable to create the selected model architecture.",
                "details": (
                    "This may be due to incompatible configuration "
                    "or missing dependencies."
                ),
                "recovery_suggestions": [
                    "Check model configuration parameters",
                    "Verify all required dependencies are installed",
                    "Try a different model architecture",
                    "Check available GPU memory",
                ],
                "retry_possible": True,
            },
            ErrorType.VRAM_EXHAUSTED: {
                "title": "GPU Memory Insufficient",
                "message": (
                    "Not enough graphics card memory to complete the "
                    "operation."
                ),
                "details": (
                    "Your RTX 3070 Ti has 8GB VRAM. The current operation "
                    "requires more memory."
                ),
                "recovery_suggestions": [
                    "Reduce batch size in configuration",
                    "Use a smaller model architecture",
                    "Close other GPU-intensive applications",
                    "Enable gradient checkpointing if available",
                ],
                "retry_possible": True,
            },
            ErrorType.TIMEOUT: {
                "title": "Operation Timed Out",
                "message": (
                    "The operation took longer than expected to complete."
                ),
                "details": (
                    "This may be due to system load or resource constraints."
                ),
                "recovery_suggestions": [
                    "Try the operation again",
                    "Check system resources (CPU, memory, disk)",
                    "Close other applications to free resources",
                    "Consider using a smaller dataset or model",
                ],
                "retry_possible": True,
            },
            ErrorType.TRAINING_FAILED: {
                "title": "Training Process Failed",
                "message": "The model training process encountered an error.",
                "details": (
                    "This could be due to data issues, configuration problems,"
                    " or resource constraints."
                ),
                "recovery_suggestions": [
                    "Check training data is valid",
                    "Verify configuration parameters",
                    "Monitor system resources",
                    "Check training logs for specific errors",
                ],
                "retry_possible": True,
            },
            ErrorType.DATA_LOADING: {
                "title": "Data Loading Error",
                "message": "Unable to load training or validation data.",
                "details": (
                    "Check that data files exist and are in the correct "
                    "format."
                ),
                "recovery_suggestions": [
                    "Verify data files exist",
                    "Check file permissions",
                    "Validate data format",
                    "Check available disk space",
                ],
                "retry_possible": True,
            },
            ErrorType.UNEXPECTED: {
                "title": "Unexpected Error",
                "message": (
                    "An unexpected error occurred during the operation."
                ),
                "details": (
                    "This is usually a temporary issue that can be resolved "
                    "by retrying."
                ),
                "recovery_suggestions": [
                    "Try the operation again",
                    "Check system resources",
                    "Restart the application if the problem persists",
                    "Report the issue if it continues",
                ],
                "retry_possible": True,
            },
        }

        return templates.get(error_type, templates[ErrorType.UNEXPECTED])


class StandardErrorState:
    """Standard implementation of ErrorState for GUI components."""

    def __init__(self, component_name: str) -> None:
        """
        Initialize error state for a specific component.

        Args:
            component_name: Name of the component for logging
        """
        self.component_name = component_name
        self._error_placeholder: Any = None
        self._current_error: ErrorInfo | None = None

    def show_error(self, error_info: ErrorInfo) -> None:
        """Display error to user with appropriate styling."""
        self._current_error = error_info

        # Create error placeholder if not exists
        if self._error_placeholder is None:
            self._error_placeholder = st.empty()

        # Inject custom CSS for error styling
        self._inject_error_css()

        # Create error content
        error_html = self._create_error_html(error_info)

        # Display error
        self._error_placeholder.markdown(error_html, unsafe_allow_html=True)

        # Log error to session state
        session_state = SessionStateManager.get()
        session_state.add_notification(
            f"Error in {self.component_name}: {error_info.title}"
        )

    def show_retry_option(
        self, retry_callback: Callable[[], None], retry_text: str = "Retry"
    ) -> None:
        """Display retry option to user."""
        if self._current_error and self._current_error.retry_possible:
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button(
                    f"🔄 {retry_text}", key=f"retry_{self.component_name}"
                ):
                    self.clear_error()
                    retry_callback()

            with col2:
                if st.button(
                    "❌ Dismiss", key=f"dismiss_{self.component_name}"
                ):
                    self.clear_error()

    def clear_error(self) -> None:
        """Clear current error state."""
        if self._error_placeholder:
            self._error_placeholder.empty()
            self._error_placeholder = None
        self._current_error = None

    def _inject_error_css(self) -> None:
        """Inject custom CSS for error styling."""
        css = """
        <style>
        .crackseg-error-container {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
            border-left: 4px solid #FF4444;
        }

        .crackseg-error-title {
            color: #721c24;
            font-size: 16px;
            font-weight: 600;
            margin: 0 0 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .crackseg-error-message {
            color: #721c24;
            font-size: 14px;
            margin: 0 0 12px 0;
            line-height: 1.4;
        }

        .crackseg-error-details {
            background: #fff;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 12px;
            margin: 8px 0;
            font-size: 13px;
            color: #495057;
        }

        .crackseg-error-suggestions {
            margin-top: 12px;
        }

        .crackseg-error-suggestions h4 {
            color: #721c24;
            font-size: 14px;
            margin: 0 0 8px 0;
        }

        .crackseg-error-suggestions ul {
            margin: 0;
            padding-left: 20px;
            color: #495057;
            font-size: 13px;
        }

        .crackseg-error-suggestions li {
            margin: 4px 0;
        }

        .crackseg-technical-info {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 8px;
            margin: 8px 0;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: #6c757d;
            max-height: 100px;
            overflow-y: auto;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    def _create_error_html(self, error_info: ErrorInfo) -> str:
        """Create HTML for error display."""
        # Error icon based on type
        icon = "⚠️" if error_info.error_type == ErrorType.TIMEOUT else "❌"

        # Build main error content
        title_html = f"""
        <div class="crackseg-error-title">
            {icon} {error_info.title}
        </div>
        """

        message_html = f"""
        <div class="crackseg-error-message">
            {error_info.message}
        </div>
        """

        # Add details if available
        details_html = ""
        if error_info.details:
            details_html = f"""
            <div class="crackseg-error-details">
                {error_info.details}
            </div>
            """

        # Add recovery suggestions if available
        suggestions_html = ""
        if error_info.recovery_suggestions:
            suggestions_list = "".join(
                f"<li>{suggestion}</li>"
                for suggestion in error_info.recovery_suggestions
            )
            suggestions_html = f"""
            <div class="crackseg-error-suggestions">
                <h4>💡 Suggested Solutions:</h4>
                <ul>{suggestions_list}</ul>
            </div>
            """

        # Add technical info if available (collapsed by default)
        technical_html = ""
        if error_info.technical_info:
            technical_html = f"""
            <details>
                <summary style="cursor: pointer; color: #6c757d;
                         font-size: 12px;">
                    🔧 Technical Details
                </summary>
                <div class="crackseg-technical-info">
                    {error_info.technical_info}
                </div>
            </details>
            """

        return f"""
        <div class="crackseg-error-container">
            {title_html}
            {message_html}
            {details_html}
            {suggestions_html}
            {technical_html}
        </div>
        """
