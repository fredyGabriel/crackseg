"""Status rendering for TensorBoard component."""

import streamlit as st

from scripts.gui.utils.tb_manager import TensorBoardManager

from ..state.session_manager import SessionStateManager
from ..utils.formatters import format_uptime


def render_status_section(
    manager: TensorBoardManager,
    session_manager: SessionStateManager,
    show_refresh: bool,
    startup_timeout: float,
) -> None:
    """Render TensorBoard status section.

    Args:
        manager: TensorBoard manager instance.
        session_manager: Session state manager.
        show_refresh: Whether to show refresh button.
        startup_timeout: Startup timeout in seconds.
    """
    col1, col2 = st.columns([2, 1])

    with col1:
        info = manager.info

        if manager.is_running:
            st.success(f"🟢 **Running** on port {info.port}")
            if info.url:
                st.caption(f"🔗 URL: {info.url}")
                uptime = info.get_uptime()
                if uptime:
                    st.caption(f"⏱️ Uptime: {format_uptime(uptime)}")
        else:
            st.info("⚪ **Stopped**")
            if session_manager.has_error():
                error_msg = session_manager.get_value("error_message")
                st.error(f"🔴 **Error**: {error_msg}")

    with col2:
        if show_refresh:
            if st.button("🔄 Refresh", help="Refresh TensorBoard status"):
                st.rerun()
