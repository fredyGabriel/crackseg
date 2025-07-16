"""Error rendering for TensorBoard component.

This module handles rendering of error states, diagnostics, and
troubleshooting information for the TensorBoard component.
"""

from pathlib import Path

import streamlit as st

from ..state.session_manager import SessionStateManager


def render_no_logs_available(
    log_dir: Path | None, error_msg: str | None = None
) -> None:
    """Render UI when no log directory is available.

    Args:
        log_dir: Log directory path (None if not specified).
        error_msg: Optional error message to display.
    """
    if log_dir is None:
        st.info("📊 No log directory specified for TensorBoard")
        with st.expander("💡 How to get TensorBoard logs", expanded=False):
            st.markdown(
                """
            **To view TensorBoard logs:**
            1. Start a training run from the Training page
            2. Ensure your model logs to `logs/tensorboard/` directory
            3. TensorBoard will automatically detect and display logs

            **Supported log formats:**
            - Scalar metrics (loss, accuracy, etc.)
            - Images and histograms
            - Model graph visualization
            """
            )
    else:
        if error_msg:
            st.error(f"❌ {error_msg}")
        else:
            st.warning(f"📂 Log directory not found: `{log_dir}`")

        # Check if parent directory exists
        parent_dir = log_dir.parent
        if parent_dir.exists():
            st.info(
                "✅ Run directory exists - TensorBoard will start once "
                "training begins"
            )
        else:
            st.error(f"❌ Run directory does not exist: `{parent_dir}`")
            st.info("Please configure a valid run directory first")

        with st.expander("🔧 Troubleshooting", expanded=False):
            st.markdown(
                f"""
            **Expected log directory:** `{log_dir}`

            **Common solutions:**
            - Ensure training has started and is writing logs
            - Check that the run directory is correctly configured
            - Verify that TensorBoard logging is enabled in your training
              config
            - Wait a few seconds after training starts for logs to appear
            """
            )


def render_not_running_state(
    session_manager: SessionStateManager,
    log_dir: Path,
    show_controls: bool,
    max_startup_attempts: int,
) -> None:
    """Render UI when TensorBoard is not running.

    Args:
        session_manager: Session state manager instance.
        log_dir: Current log directory.
        show_controls: Whether controls are shown.
        max_startup_attempts: Maximum startup attempts allowed.
    """
    # Show error information if available
    if session_manager.has_error():
        error_msg = session_manager.get_value("error_message")
        error_type = session_manager.get_value("error_type")
        _render_error_status(error_msg, error_type, session_manager)

    # Main status message
    st.info("🔧 TensorBoard is not currently running")

    # Provide guidance based on context
    if show_controls:
        startup_attempts = session_manager.get_value("startup_attempts", 0)
        if startup_attempts >= max_startup_attempts:
            st.warning(
                f"⚠️ Maximum startup attempts reached ({max_startup_attempts})"
            )
            st.markdown(
                "Try the **🧹 Reset** button above to clear the error state."
            )
        else:
            st.markdown(
                "Click **🚀 Start** above to launch TensorBoard with the "
                "current log directory."
            )
    else:
        st.markdown(
            f"TensorBoard needs to be started to view logs from: `{log_dir}`"
        )

    # Show helpful tips
    with st.expander("💡 Tips for TensorBoard", expanded=False):
        st.markdown(
            """
        **TensorBoard Tips:**
        - TensorBoard automatically refreshes when new logs are written
        - Use the scalars tab to view training metrics
        - The graphs tab shows your model architecture
        - Images tab displays sample predictions (if logged)
        - Use the time series selector to focus on specific metrics
        """
        )


def _render_error_status(
    error_message: str | None,
    error_type: str | None,
    session_manager: SessionStateManager,
) -> None:
    """Render detailed error status with recovery options.

    Args:
        error_message: Error message to display.
        error_type: Type of error for categorization.
        session_manager: Session state manager for recovery actions.
    """
    if not error_message:
        return

    st.error(f"🔴 **Failed**: {error_message}")

    # Show recovery options based on error type
    if error_type == "PortConflictError":
        st.caption(
            "💡 **Solution**: Port is in use. Try restarting or use "
            "different port."
        )
        if st.button(
            "🔄 Try Different Port",
            help="Attempt startup with different port",
        ):
            session_manager.reset_startup_state()
            st.rerun()

    elif error_type == "LogDirectoryError":
        st.caption(
            "💡 **Solution**: Ensure log directory exists and is accessible."
        )

    elif error_type == "ProcessStartupError":
        st.caption(
            "💡 **Solution**: Check system resources and try restarting."
        )
        if st.button("🔄 Retry Startup", help="Attempt startup again"):
            session_manager.reset_startup_state()
            st.rerun()
    else:
        # Generic recovery options
        startup_attempts = session_manager.get_value("startup_attempts", 0)
        max_attempts = 3  # Default max attempts
        if startup_attempts < max_attempts:
            if st.button(
                "🔄 Retry",
                help=(
                    f"Retry startup (attempt {startup_attempts + 1}/"
                    f"{max_attempts})"
                ),
            ):
                session_manager.reset_startup_state()
                st.rerun()
