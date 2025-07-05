"""
Loading spinner component for the CrackSeg application.

This module provides a professional loading spinner with contextual messaging,
brand-aligned styling, and timeout handling for enhanced user experience.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager

import streamlit as st

from scripts.gui.utils.error_state import (
    ErrorMessageFactory,
    ErrorType,
    StandardErrorState,
)
from scripts.gui.utils.session_state import SessionStateManager


class LoadingSpinner:
    """Professional loading spinner component with brand alignment and timeout
    handling."""

    # Brand-aligned color scheme matching LogoComponent
    _BRAND_COLORS = {
        "primary": "#2E2E2E",
        "secondary": "#F0F0F0",
        "accent": "#FF4444",
        "success": "#00FF64",
        "warning": "#FFB800",
        "error": "#FF4444",
    }

    # Spinner animation styles
    _SPINNER_STYLES = {
        "crack_pattern": {
            "animation": "crack-pulse",
            "color": _BRAND_COLORS["accent"],
            "description": "Crack pattern animation",
        },
        "road_analysis": {
            "animation": "road-scan",
            "color": _BRAND_COLORS["primary"],
            "description": "Road analysis pattern",
        },
        "ai_processing": {
            "animation": "ai-pulse",
            "color": _BRAND_COLORS["success"],
            "description": "AI processing pattern",
        },
        "default": {
            "animation": "rotate",
            "color": _BRAND_COLORS["primary"],
            "description": "Default spinner",
        },
    }

    @staticmethod
    def _inject_custom_css() -> None:
        """Inject custom CSS for brand-aligned spinner styling."""
        css = """
        <style>
        /* CrackSeg Brand Spinner Styles */
        .crackseg-spinner-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }

        .crackseg-spinner-icon {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #FF4444;
            border-radius: 50%;
            animation: crackseg-rotate 1s linear infinite;
        }

        .crackseg-spinner-text {
            margin-top: 15px;
            font-size: 16px;
            color: #2E2E2E;
            font-weight: 500;
            text-align: center;
            line-height: 1.4;
        }

        .crackseg-spinner-subtext {
            margin-top: 5px;
            font-size: 12px;
            color: #666;
            text-align: center;
            opacity: 0.8;
        }

        @keyframes crackseg-rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes crack-pulse {
            0%, 100% { opacity: 1; border-color: #FF4444; }
            50% { opacity: 0.6; border-color: #FF8888; }
        }

        @keyframes road-scan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        @keyframes ai-pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.05); }
        }

        /* Timeout warning styles */
        .crackseg-timeout-warning {
            background: #FFF3CD;
            border: 1px solid #FFB800;
            color: #856404;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 14px;
        }

        /* Success completion styles */
        .crackseg-success-message {
            background: #D4EDDA;
            border: 1px solid #00FF64;
            color: #155724;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 14px;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    @staticmethod
    def _render_custom_spinner(
        message: str, subtext: str | None = None, spinner_type: str = "default"
    ) -> None:
        """Render custom branded spinner with contextual messaging."""
        LoadingSpinner._inject_custom_css()

        # Create spinner HTML with optional subtext
        subtext_html = (
            f'<div class="crackseg-spinner-subtext">{subtext}</div>'
            if subtext
            else ""
        )

        spinner_html = f"""
        <div class="crackseg-spinner-container">
            <div class="crackseg-spinner-icon"></div>
            <div class="crackseg-spinner-text">{message}</div>
            {subtext_html}
        </div>
        """

        st.markdown(spinner_html, unsafe_allow_html=True)

    @staticmethod
    @contextmanager
    def spinner(
        message: str,
        subtext: str | None = None,
        spinner_type: str = "default",
        timeout_seconds: int = 30,
        show_custom_ui: bool = True,
    ) -> Generator[None, None, None]:
        """
        Context manager for displaying loading spinner with timeout handling.

        Args:
            message: Primary loading message to display
            subtext: Optional secondary message for additional context
            spinner_type: Type of spinner animation
                ('crack_pattern', 'road_analysis', 'ai_processing', 'default')
            timeout_seconds: Maximum time before showing timeout warning
            show_custom_ui: Whether to show custom UI alongside native spinner

        Example:
            >>> with LoadingSpinner.spinner(
            ...     "Loading configuration...", "Validating YAML structure"
            ... ):
            ...     # Your loading operation here
            ...     time.sleep(2)
        """
        start_time = time.time()

        # Update session state to reflect loading
        session_state = SessionStateManager.get()
        session_state.add_notification(f"Loading: {message}")

        try:
            # Use native Streamlit spinner for reliability
            with st.spinner(message):
                # Show custom UI if requested
                if show_custom_ui:
                    # Create placeholder for custom spinner
                    spinner_placeholder = st.empty()

                    # Show custom spinner initially
                    with spinner_placeholder.container():
                        LoadingSpinner._render_custom_spinner(
                            message, subtext, spinner_type
                        )

                    # Yield control to the calling code
                    yield

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time

                    # Show timeout warning if operation took too long
                    if elapsed_time > timeout_seconds:
                        timeout_msg = (
                            f"⚠️ Operation took longer than expected "
                            f"({elapsed_time:.1f}s)."
                        )
                        background_msg = (
                            "The process may still be running "
                            "in the background."
                        )
                        with spinner_placeholder.container():
                            st.markdown(
                                f"""
                                <div class="crackseg-timeout-warning">
                                    {timeout_msg}<br/>
                                    {background_msg}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        # Show success message briefly
                        success_msg = (
                            f"✅ {message} completed successfully "
                            f"({elapsed_time:.1f}s)"
                        )
                        with spinner_placeholder.container():
                            st.markdown(
                                f"""
                                <div class="crackseg-success-message">
                                    {success_msg}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        time.sleep(1)  # Brief success message display

                    # Clear the spinner placeholder
                    spinner_placeholder.empty()
                else:
                    # Just yield for native spinner only
                    yield

        except Exception as e:
            # Handle errors with enhanced error messaging
            error_state = StandardErrorState("LoadingSpinner")

            # Determine error type based on exception
            error_type = LoadingSpinner._classify_error(e)

            # Create error info with context
            error_info = ErrorMessageFactory.create_error_info(
                error_type=error_type,
                exception=e,
                context={
                    "operation": message,
                    "timeout_seconds": timeout_seconds,
                    "elapsed_time": time.time() - start_time,
                },
            )

            # Show enhanced error message
            error_state.show_error(error_info)

            # Add retry option for recoverable errors
            if error_info.retry_possible:
                st.info(
                    "💡 You can try the operation again after addressing "
                    "the suggested solutions."
                )

            raise
        finally:
            # Update session state completion
            elapsed_time = time.time() - start_time
            session_state.add_notification(
                f"Completed: {message} ({elapsed_time:.1f}s)"
            )

    @staticmethod
    def show_progress_with_spinner(
        message: str,
        progress: float,
        subtext: str | None = None,
        spinner_type: str = "default",
    ) -> None:
        """
        Display progress bar with spinner for longer operations.

        Args:
            message: Primary progress message
            progress: Progress value between 0.0 and 1.0
            subtext: Optional secondary message
            spinner_type: Type of spinner animation
        """
        LoadingSpinner._inject_custom_css()

        # Create progress display
        col1, col2 = st.columns([1, 4])

        with col1:
            LoadingSpinner._render_custom_spinner(
                f"{progress:.0%}", "Complete", spinner_type
            )

        with col2:
            st.markdown(f"**{message}**")
            if subtext:
                st.caption(subtext)
            st.progress(progress)

    @staticmethod
    def get_contextual_message(operation_type: str) -> tuple[str, str, str]:
        """
        Get contextual loading messages for different operation types.

        Args:
            operation_type: Type of operation
                ('config', 'model', 'training', 'results', 'tensorboard')

        Returns:
            Tuple of (message, subtext, spinner_type)
        """
        messages = {
            "config": (
                "Loading configuration...",
                "Validating YAML structure and parameters",
                "default",
            ),
            "model": (
                "Instantiating model architecture...",
                "Building encoder-decoder components",
                "ai_processing",
            ),
            "training": (
                "Starting training process...",
                "Initializing data loaders and optimizers",
                "crack_pattern",
            ),
            "results": (
                "Scanning results directory...",
                "Analyzing predictions and metrics",
                "road_analysis",
            ),
            "tensorboard": (
                "Starting TensorBoard server...",
                "Initializing visualization dashboard",
                "ai_processing",
            ),
            "export": (
                "Exporting results...",
                "Generating reports and visualizations",
                "default",
            ),
        }

        return messages.get(
            operation_type,
            (
                "Processing...",
                "Please wait while the operation completes",
                "default",
            ),
        )

    @staticmethod
    def _classify_error(exception: Exception) -> ErrorType:
        """
        Classify exception into appropriate ErrorType.

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

    @staticmethod
    def demo_spinner_types() -> None:
        """Demonstrate different spinner types for development/testing."""
        st.header("CrackSeg Loading Spinner Demo")

        spinner_types = list(LoadingSpinner._SPINNER_STYLES.keys())

        for spinner_type in spinner_types:
            st.subheader(f"Spinner Type: {spinner_type}")

            if st.button(f"Test {spinner_type} spinner"):
                message, subtext, _ = LoadingSpinner.get_contextual_message(
                    "model" if spinner_type == "ai_processing" else "config"
                )

                with LoadingSpinner.spinner(
                    message=message,
                    subtext=subtext,
                    spinner_type=spinner_type,
                    timeout_seconds=5,
                ):
                    # Simulate operation
                    time.sleep(2)

                st.success(f"✅ {spinner_type} spinner demo completed!")


# Helper function for easy import
def loading_spinner(
    message: str,
    subtext: str | None = None,
    spinner_type: str = "default",
    timeout_seconds: int = 30,
):
    """
    Convenience function for creating loading spinner context manager.

    Args:
        message: Primary loading message
        subtext: Optional secondary message
        spinner_type: Type of spinner animation
        timeout_seconds: Timeout threshold in seconds

    Returns:
        Context manager for loading spinner
    """
    return LoadingSpinner.spinner(
        message=message,
        subtext=subtext,
        spinner_type=spinner_type,
        timeout_seconds=timeout_seconds,
    )
