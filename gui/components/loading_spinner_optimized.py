"""
Optimized loading spinner component for the CrackSeg application.

This module provides a high-performance loading spinner with advanced
optimizations including CSS caching, update debouncing, memory management, and
performance monitoring for superior user experience.
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
from scripts.gui.utils.performance_optimizer import (
    AsyncOperationManager,
    MemoryManager,
    OptimizedHTMLBuilder,
    inject_css_once,
    should_update,
    track_performance,
)
from scripts.gui.utils.session_state import SessionStateManager


class OptimizedLoadingSpinner:
    """
    High-performance loading spinner component with advanced optimizations.
    """

    # Brand-aligned color scheme
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

    # CSS is cached globally - injected only once per session
    _CSS_CONTENT = """
    <style>
    /* CrackSeg Optimized Spinner Styles */
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

    /* Optimized timeout and success styles */
    .crackseg-timeout-warning {
        background: #FFF3CD;
        border: 1px solid #FFB800;
        color: #856404;
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 14px;
        animation: fadeIn 0.3s ease-in;
    }

    .crackseg-success-message {
        background: #D4EDDA;
        border: 1px solid #00FF64;
        color: #155724;
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 14px;
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Performance optimization: GPU acceleration */
    .crackseg-spinner-icon,
    .crackseg-spinner-container {
        will-change: transform;
        transform: translateZ(0);
    }
    </style>
    """

    @staticmethod
    def _ensure_css_injected() -> None:
        """
        Ensure CSS is injected only once per session for optimal performance.
        """
        inject_css_once(
            "crackseg_spinner_optimized", OptimizedLoadingSpinner._CSS_CONTENT
        )

    @staticmethod
    def _render_optimized_spinner(
        message: str,
        subtext: str | None = None,
        spinner_type: str = "default",
        component_id: str = "default_spinner",
    ) -> None:
        """
        Render optimized spinner with performance tracking and caching.

        Args:
            message: Primary spinner message
            subtext: Optional secondary message
            spinner_type: Type of spinner animation
            component_id: Unique component identifier for performance tracking
        """
        start_time = time.time()

        # Ensure CSS is injected (cached globally)
        OptimizedLoadingSpinner._ensure_css_injected()

        # Use optimized HTML builder for efficient rendering
        spinner_html = OptimizedHTMLBuilder.build_spinner_html(
            message=message,
            subtext=subtext or "",
            spinner_type=spinner_type,
        )

        # Render with performance monitoring
        st.markdown(spinner_html, unsafe_allow_html=True)

        # Track performance metrics
        track_performance(component_id, "render_spinner", start_time)

    @staticmethod
    @contextmanager
    def spinner(
        message: str,
        subtext: str | None = None,
        spinner_type: str = "default",
        timeout_seconds: int = 30,
        show_custom_ui: bool = True,
        operation_id: str | None = None,
    ) -> Generator[None, None, None]:
        """
        Optimized context manager for loading spinner with advanced performance
        features.

        Args:
            message: Primary loading message
            subtext: Optional secondary message
            spinner_type: Spinner animation type
            timeout_seconds: Maximum operation time before timeout warning
            show_custom_ui: Whether to show custom UI
            operation_id: Unique operation identifier for tracking

        Yields:
            None: Context for the operation
        """
        # Generate unique operation ID if not provided
        if operation_id is None:
            operation_id = f"spinner_{int(time.time())}"

        start_time = time.time()

        # Start async operation tracking
        AsyncOperationManager.start_operation(
            operation_id=operation_id,
            title=message,
            estimated_duration=timeout_seconds,
        )

        # Efficient session state management
        session_state = SessionStateManager.get()

        # Register memory cleanup callback
        MemoryManager.register_cleanup_callback(
            operation_id,
            lambda: AsyncOperationManager.finish_operation(
                operation_id, False
            ),
        )

        # Create placeholder for updates (registered for cleanup)
        spinner_placeholder = None

        try:
            # Use native Streamlit spinner for reliability
            with st.spinner(message):
                if show_custom_ui:
                    # Create optimized placeholder
                    spinner_placeholder = st.empty()

                    # Render initial spinner state
                    with spinner_placeholder.container():
                        OptimizedLoadingSpinner._render_optimized_spinner(
                            message=message,
                            subtext=subtext,
                            spinner_type=spinner_type,
                            component_id=operation_id,
                        )

                    # Update operation status
                    AsyncOperationManager.update_operation(
                        operation_id=operation_id,
                        progress=0.0,
                        status="running",
                    )

                # Yield control to calling code
                yield

                # Calculate elapsed time
                elapsed_time = time.time() - start_time

                # Update final operation status
                AsyncOperationManager.update_operation(
                    operation_id=operation_id,
                    progress=1.0,
                    status="completed",
                )

                # Show completion or timeout message (only if update is needed)
                if show_custom_ui and spinner_placeholder:
                    if elapsed_time > timeout_seconds:
                        # Show timeout warning with optimized rendering
                        timeout_html = f"""
                        <div class="crackseg-timeout-warning">
                            ⚠️ Operation took longer than expected
                            ({elapsed_time:.1f}s).<br/>
                            The process may still be running in the background.
                        </div>
                        """
                        spinner_placeholder.markdown(
                            timeout_html, unsafe_allow_html=True
                        )
                    else:
                        # Show success message with optimized rendering
                        success_html = f"""
                        <div class="crackseg-success-message">
                            ✅ {message} completed successfully
                            ({elapsed_time:.1f}s)
                        </div>
                        """
                        spinner_placeholder.markdown(
                            success_html, unsafe_allow_html=True
                        )

                        # Brief non-blocking display
                        time.sleep(0.8)

                    # Clean up placeholder
                    spinner_placeholder.empty()

        except Exception as e:
            # Enhanced error handling with performance tracking
            error_start = time.time()

            # Update operation status
            AsyncOperationManager.finish_operation(operation_id, False)

            # Handle errors with optimized error messaging
            error_state = StandardErrorState(
                f"OptimizedSpinner_{operation_id}"
            )
            error_type = OptimizedLoadingSpinner._classify_error(e)

            # Create error info with performance context
            error_info = ErrorMessageFactory.create_error_info(
                error_type=error_type,
                exception=e,
                context={
                    "operation": message,
                    "operation_id": operation_id,
                    "timeout_seconds": timeout_seconds,
                    "elapsed_time": time.time() - start_time,
                    "spinner_type": spinner_type,
                },
            )

            # Show error with retry information
            error_state.show_error(error_info)

            if error_info.retry_possible:
                st.info(
                    "💡 You can retry this operation. The system has been "
                    "optimized to handle retries efficiently."
                )

            # Track error handling performance
            track_performance(operation_id, "error_handling", error_start)

            raise

        finally:
            # Comprehensive cleanup
            elapsed_time = time.time() - start_time

            # Update session state efficiently
            if should_update(f"notification_{operation_id}", 0.5):
                session_state.add_notification(
                    f"Completed: {message} ({elapsed_time:.1f}s)"
                )

            # Finish operation tracking
            AsyncOperationManager.finish_operation(operation_id, True)

            # Clean up resources
            MemoryManager.cleanup_component(operation_id)

            # Track overall performance
            track_performance(operation_id, "complete_operation", start_time)

    @staticmethod
    def show_progress_with_spinner(
        message: str,
        progress: float,
        subtext: str | None = None,
        spinner_type: str = "default",
        component_id: str | None = None,
    ) -> None:
        """
        Display optimized progress bar with spinner for longer operations.

        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
            subtext: Optional secondary message
            spinner_type: Spinner animation type
            component_id: Unique component identifier
        """
        if component_id is None:
            component_id = f"progress_spinner_{int(time.time())}"

        # Check if update is needed (debouncing)
        if not should_update(component_id, 0.1):
            return

        start_time = time.time()

        # Ensure CSS is injected
        OptimizedLoadingSpinner._ensure_css_injected()

        # Create optimized layout
        col1, col2 = st.columns([1, 4])

        with col1:
            OptimizedLoadingSpinner._render_optimized_spinner(
                message=f"{progress:.0%}",
                subtext="Complete",
                spinner_type=spinner_type,
                component_id=f"{component_id}_icon",
            )

        with col2:
            st.markdown(f"**{message}**")
            if subtext:
                st.caption(subtext)
            st.progress(progress)

        # Track performance
        track_performance(component_id, "progress_with_spinner", start_time)

    @staticmethod
    def get_contextual_message(operation_type: str) -> tuple[str, str, str]:
        """
        Get contextual loading messages with performance optimization.

        Args:
            operation_type: Type of operation

        Returns:
            Tuple of (message, subtext, spinner_type)
        """
        # Cached message lookup for performance
        _message_cache = {
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

        return _message_cache.get(
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
        Optimized error classification with caching for performance.

        Args:
            exception: Exception to classify

        Returns:
            Appropriate ErrorType
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

    @staticmethod
    def demo_spinner_types() -> None:
        """
        Demonstrate different spinner types with performance monitoring.
        """
        st.header("CrackSeg Optimized Loading Spinner Demo")

        # Performance monitoring setup
        demo_start = time.time()

        st.info(
            "This demo uses the optimized spinner component with CSS caching, "
            "debouncing, and performance tracking."
        )

        spinner_types = list(OptimizedLoadingSpinner._SPINNER_STYLES.keys())

        for spinner_type in spinner_types:
            st.subheader(f"Spinner Type: {spinner_type}")

            if st.button(f"Test {spinner_type} spinner"):
                message, subtext, _ = (
                    OptimizedLoadingSpinner.get_contextual_message(
                        "model"
                        if spinner_type == "ai_processing"
                        else "config"
                    )
                )

                with OptimizedLoadingSpinner.spinner(
                    message=message,
                    subtext=subtext,
                    spinner_type=spinner_type,
                    timeout_seconds=5,
                    operation_id=f"demo_{spinner_type}",
                ):
                    # Simulate operation
                    time.sleep(2)

                st.success(f"✅ {spinner_type} spinner demo completed!")

        # Track demo performance
        track_performance("spinner_demo", "complete_demo", demo_start)


# Convenience function for easy import
def optimized_loading_spinner(
    message: str,
    subtext: str | None = None,
    spinner_type: str = "default",
    timeout_seconds: int = 30,
    operation_id: str | None = None,
):
    """
    Convenience function for creating optimized loading spinner.

    Args:
        message: Primary loading message
        subtext: Optional secondary message
        spinner_type: Spinner animation type
        timeout_seconds: Timeout threshold in seconds
        operation_id: Unique operation identifier

    Returns:
        Optimized context manager for loading spinner
    """
    return OptimizedLoadingSpinner.spinner(
        message=message,
        subtext=subtext,
        spinner_type=spinner_type,
        timeout_seconds=timeout_seconds,
        operation_id=operation_id,
    )
