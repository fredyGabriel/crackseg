"""Control rendering for TensorBoard component."""

from pathlib import Path

import streamlit as st

from scripts.gui.utils.tb_manager import TensorBoardManager

from ..state.session_manager import SessionStateManager


def render_control_section(
    manager: TensorBoardManager,
    session_manager: SessionStateManager,
    log_dir: Path,
) -> None:
    """Render TensorBoard control buttons.

    Args:
        manager: TensorBoard manager instance.
        session_manager: Session state manager.
        log_dir: Current log directory.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "🚀 Start",
            disabled=manager.is_running,
            help="Start TensorBoard with current log directory",
        ):
            _start_with_feedback(manager, session_manager, log_dir)

    with col2:
        if st.button(
            "🔄 Restart",
            disabled=not manager.is_running,
            help="Restart TensorBoard with fresh process",
        ):
            _restart_with_feedback(manager, session_manager, log_dir)

    with col3:
        if st.button(
            "⏹️ Stop",
            disabled=not manager.is_running,
            help="Stop TensorBoard process",
        ):
            _stop_with_feedback(manager, session_manager)

    with col4:
        if st.button(
            "🧹 Reset", help="Reset error state and attempt counters"
        ):
            session_manager.reset_startup_state()
            st.success("✅ State reset successfully")
            st.rerun()


def _start_with_feedback(
    manager: TensorBoardManager,
    session_manager: SessionStateManager,
    log_dir: Path,
) -> None:
    """Start TensorBoard with user feedback."""
    try:
        with st.spinner("Starting TensorBoard..."):
            success = manager.start_tensorboard(log_dir, force_restart=False)

        if success:
            st.success("✅ TensorBoard started successfully!")
            session_manager.update_state(
                user_initiated_stop=False,
                error_message=None,
                error_type=None,
                startup_attempts=0,
            )
            st.rerun()
        else:
            st.error("❌ Failed to start TensorBoard")

    except Exception as e:
        st.error(f"❌ Error starting TensorBoard: {e}")
        session_manager.set_error(str(e), type(e).__name__)


def _restart_with_feedback(
    manager: TensorBoardManager,
    session_manager: SessionStateManager,
    log_dir: Path,
) -> None:
    """Restart TensorBoard with user feedback."""
    try:
        with st.spinner("Restarting TensorBoard..."):
            success = manager.restart_tensorboard(log_dir)

        if success:
            st.success("✅ TensorBoard restarted successfully!")
            session_manager.update_state(
                error_message=None,
                error_type=None,
                startup_attempts=0,
            )
            st.rerun()
        else:
            st.error("❌ Failed to restart TensorBoard")

    except Exception as e:
        st.error(f"❌ Error restarting TensorBoard: {e}")


def _stop_with_feedback(
    manager: TensorBoardManager, session_manager: SessionStateManager
) -> None:
    """Stop TensorBoard with user feedback."""
    with st.spinner("Stopping TensorBoard..."):
        success = manager.stop_tensorboard()

    if success:
        st.info("ℹ️ TensorBoard stopped")
        session_manager.update_state(user_initiated_stop=True)
        st.rerun()
    else:
        st.error("❌ Failed to stop TensorBoard")
