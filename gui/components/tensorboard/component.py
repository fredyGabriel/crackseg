"""Refactored TensorBoard component for CrackSeg GUI.

This is the main component class that orchestrates TensorBoard integration
by delegating specific responsibilities to specialized modules. This follows
the single responsibility principle and keeps the main component focused.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from scripts.gui.utils.tb_manager import (
    TensorBoardManager,
    get_default_tensorboard_manager,
    get_global_lifecycle_manager,
)

from .state.session_manager import SessionStateManager
from .utils.validators import (
    validate_iframe_dimensions,
    validate_log_directory,
)


class TensorBoardComponent:
    """Main TensorBoard component for Streamlit integration.

    This component provides a clean interface for TensorBoard integration
    within Streamlit applications. It delegates specific responsibilities
    to specialized modules for better maintainability.

    Features:
    - Automatic TensorBoard startup/shutdown management
    - Session state management with validation
    - Error handling and recovery strategies
    - Responsive iframe embedding
    - Progress tracking and status indicators

    Example:
        >>> component = TensorBoardComponent()
        >>> log_path = Path("outputs/experiment_1/logs/tensorboard")
        >>> component.render(log_dir=log_path)
    """

    def __init__(
        self,
        manager: TensorBoardManager | None = None,
        default_height: int = 600,
        default_width: int | None = None,
        auto_startup: bool = True,
        show_controls: bool = True,
        show_status: bool = True,
        enable_lifecycle_management: bool = True,
        startup_timeout: float = 30.0,
        max_startup_attempts: int = 3,
    ) -> None:
        """Initialize TensorBoard component.

        Args:
            manager: TensorBoard manager instance (uses default if None)
            default_height: Default iframe height in pixels
            default_width: Default iframe width (None for responsive)
            auto_startup: Automatically start TensorBoard when log dir exists
            show_controls: Show start/stop controls in UI
            show_status: Show status indicators in UI
            enable_lifecycle_management: Enable automatic lifecycle management
            startup_timeout: Maximum seconds to wait for startup
            max_startup_attempts: Maximum automatic retry attempts
        """
        # Core dependencies
        self._manager = manager or get_default_tensorboard_manager()
        self._session_manager = SessionStateManager()

        # Configuration
        self._default_height = default_height
        self._default_width = default_width
        self._auto_startup = auto_startup
        self._show_controls = show_controls
        self._show_status = show_status
        self._startup_timeout = startup_timeout
        self._max_startup_attempts = max_startup_attempts

        # Lifecycle management integration
        if enable_lifecycle_management:
            self._lifecycle_manager = get_global_lifecycle_manager()
        else:
            self._lifecycle_manager = None

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate component configuration."""
        config = {
            "default_height": self._default_height,
            "auto_startup": self._auto_startup,
            "show_controls": self._show_controls,
            "show_status": self._show_status,
            "startup_timeout": self._startup_timeout,
        }

        # Import here to avoid circular imports
        from .utils.validators import validate_component_config

        is_valid, error_msg = validate_component_config(config)
        if not is_valid:
            raise ValueError(f"Invalid component configuration: {error_msg}")

    def render(
        self,
        log_dir: Path | None = None,
        height: int | None = None,
        width: int | None = None,
        title: str = "TensorBoard",
        show_refresh: bool = True,
    ) -> bool:
        """Render TensorBoard component in Streamlit.

        Args:
            log_dir: Path to TensorBoard log directory
            height: Iframe height override
            width: Iframe width override
            title: Component title
            show_refresh: Show refresh button

        Returns:
            True if TensorBoard is running and embedded, False otherwise
        """
        st.subheader(title)

        # Validate inputs
        if not self._validate_render_inputs(log_dir, height, width):
            return False

        # Handle log directory availability
        if not self._handle_log_directory(log_dir):
            return False

        # At this point log_dir is guaranteed to be valid Path
        assert log_dir is not None  # Type narrowing for mypy/basedpyright

        # Handle auto-startup logic
        self._handle_auto_startup(log_dir)

        # Render status and controls
        self._render_ui_sections(show_refresh, log_dir)

        # Render main content
        return self._render_main_content(log_dir, height, width)

    def _validate_render_inputs(
        self, log_dir: Path | None, height: int | None, width: int | None
    ) -> bool:
        """Validate render method inputs."""
        # Validate dimensions
        is_valid, error_msg = validate_iframe_dimensions(height, width)
        if not is_valid:
            st.error(f"❌ Invalid dimensions: {error_msg}")
            return False

        return True

    def _handle_log_directory(self, log_dir: Path | None) -> bool:
        """Handle log directory validation and state updates."""
        if log_dir is None:
            self._render_no_logs_available(None)
            return False

        # Validate log directory
        is_valid, error_msg = validate_log_directory(log_dir)
        if not is_valid:
            self._render_no_logs_available(log_dir, error_msg)
            return False

        # Update session state with new log directory
        self._session_manager.set_log_directory(log_dir)
        return True

    def _handle_auto_startup(self, log_dir: Path) -> None:
        """Handle automatic TensorBoard startup logic."""
        # Use lifecycle manager if available
        if self._lifecycle_manager:
            handled = self._lifecycle_manager.handle_log_directory_available(
                log_dir
            )
            if handled:
                self._session_manager.update_state(
                    startup_attempted=True,
                    error_message=None,
                    error_type=None,
                )
                return

        # Standard auto-startup logic
        if (
            self._auto_startup
            and not self._manager.is_running
            and self._session_manager.should_attempt_startup(log_dir)
        ):
            self._attempt_startup(log_dir)

    def _attempt_startup(self, log_dir: Path) -> None:
        """Attempt TensorBoard startup with progress tracking."""
        # Import rendering modules here to avoid circular imports
        from .rendering.startup_renderer import render_startup_progress

        # Track startup attempt
        attempts = self._session_manager.increment_startup_attempts()

        # Show startup progress
        progress_container = st.empty()
        render_startup_progress(
            progress_container, attempts, self._max_startup_attempts
        )

        try:
            # Attempt startup
            success = self._manager.start_tensorboard(log_dir)

            if success:
                self._session_manager.update_state(
                    startup_attempted=True,
                    error_message=None,
                    error_type=None,
                    startup_progress=1.0,
                )
                progress_container.success(
                    "✅ TensorBoard started successfully!"
                )
                st.rerun()
            else:
                progress_container.empty()
                self._handle_startup_failure(
                    "Failed to start TensorBoard process"
                )

        except Exception as e:
            progress_container.empty()
            self._handle_startup_failure(str(e), type(e).__name__)

    def _handle_startup_failure(
        self, error_message: str, error_type: str | None = None
    ) -> None:
        """Handle startup failure with error tracking."""
        self._session_manager.set_error(error_message, error_type)

        # Import recovery module here to avoid circular imports
        from .recovery.recovery_strategies import attempt_automatic_recovery

        # Try automatic recovery if possible
        if (
            self._session_manager.get_value("startup_attempts")
            < self._max_startup_attempts
        ):
            attempt_automatic_recovery(error_type, self._session_manager)

    def _render_ui_sections(self, show_refresh: bool, log_dir: Path) -> None:
        """Render status and control sections."""
        # Import rendering modules here to avoid circular imports
        from .rendering.advanced_status_renderer import (
            render_advanced_status_section,
        )
        from .rendering.control_renderer import render_control_section

        if self._show_status:
            # Use advanced status indicators for comprehensive monitoring
            render_advanced_status_section(
                self._manager,
                self._session_manager,
                show_refresh=show_refresh,
                show_diagnostics=True,
                compact_mode=False,
            )

        if self._show_controls:
            render_control_section(
                self._manager, self._session_manager, log_dir
            )

    def _render_main_content(
        self, log_dir: Path, height: int | None, width: int | None
    ) -> bool:
        """Render the main TensorBoard iframe content."""
        if not self._manager.is_running:
            # Import rendering modules here to avoid circular imports
            from .rendering.error_renderer import render_not_running_state

            render_not_running_state(
                self._session_manager,
                log_dir,
                self._show_controls,
                self._max_startup_attempts,
            )
            return False

        # Import iframe renderer here to avoid circular imports
        from .rendering.iframe_renderer import render_tensorboard_iframe

        return render_tensorboard_iframe(
            self._manager.get_url(),
            log_dir,
            height or self._default_height,
            width or self._default_width,
        )

    def _render_no_logs_available(
        self, log_dir: Path | None, error_msg: str | None = None
    ) -> None:
        """Render UI when no log directory is available."""
        # Import rendering modules here to avoid circular imports
        from .rendering.error_renderer import render_no_logs_available

        render_no_logs_available(log_dir, error_msg)

    # Public interface methods

    def get_manager(self) -> TensorBoardManager:
        """Get the underlying TensorBoard manager."""
        return self._manager

    def is_running(self) -> bool:
        """Check if TensorBoard is currently running."""
        return self._manager.is_running

    def get_url(self) -> str | None:
        """Get the current TensorBoard URL."""
        return self._manager.get_url()

    def get_startup_progress(self) -> float:
        """Get current startup progress (0.0 to 1.0)."""
        return self._session_manager.get_value("startup_progress", 0.0)

    def has_error(self) -> bool:
        """Check if component is in error state."""
        return (
            self._session_manager.has_error() or self._manager.info.has_error()
        )

    def get_error_info(self) -> dict[str, Any]:
        """Get detailed error information."""
        return {
            "session_error": self._session_manager.get_value("error_message"),
            "session_error_type": self._session_manager.get_value(
                "error_type"
            ),
            "startup_attempts": self._session_manager.get_value(
                "startup_attempts", 0
            ),
            "recovery_attempted": self._session_manager.get_value(
                "recovery_attempted", False
            ),
            "manager_error": self._manager.info.error_message,
        }

    def reset_state(self) -> None:
        """Reset component state for fresh start."""
        self._session_manager.reset_startup_state()

    def clear_errors(self) -> None:
        """Clear all error states."""
        self._session_manager.reset_error_state()
