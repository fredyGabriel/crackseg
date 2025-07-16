"""Diagnostic panel for detailed TensorBoard information."""

import time
from typing import Any

import streamlit as st

from scripts.gui.components.tensorboard.utils.formatters import format_uptime


class DiagnosticPanel:
    """Panel for detailed TensorBoard diagnostic information."""

    def __init__(self) -> None:
        """Initialize diagnostic panel."""
        pass

    def render(self, manager: Any, session_manager: Any) -> None:
        """Render detailed diagnostic information.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.
        """
        with st.expander(
            "📊 **Detailed Status & Diagnostics**", expanded=False
        ):
            self._render_process_info(manager)
            self._render_error_info(manager)
            self._render_session_info(session_manager)

    def _render_process_info(self, manager: Any) -> None:
        """Render process information section.

        Args:
            manager: TensorBoard manager instance.
        """
        st.markdown("#### 🔍 Process Information")

        # Process details
        info = manager.info
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Process State:**")
            process_data = {
                "State": info.state.value.title(),
                "PID": info.pid or "Not assigned",
                "Port": info.port or "Not allocated",
                "Startup Attempts": info.startup_attempts,
            }

            for key, value in process_data.items():
                st.text(f"  {key}: {value}")

        with col2:
            st.markdown("**Timing Information:**")
            timing_data = {}

            if info.start_time:
                timing_data["Started"] = time.ctime(info.start_time)

            uptime = info.get_uptime()
            if uptime:
                timing_data["Uptime"] = format_uptime(uptime)

            health_age = info.get_health_age()
            if health_age is not None:
                timing_data["Last Health Check"] = f"{health_age:.0f}s ago"

            for key, value in timing_data.items():
                st.text(f"  {key}: {value}")

    def _render_error_info(self, manager: Any) -> None:
        """Render error information if present.

        Args:
            manager: TensorBoard manager instance.
        """
        info = manager.info
        if info.error_message:
            st.markdown("#### ⚠️ Error Information")
            st.error(f"**Error**: {info.error_message}")

    def _render_session_info(self, session_manager: Any) -> None:
        """Render session state information.

        Args:
            session_manager: Session state manager.
        """
        st.markdown("#### 📋 Session State")
        session_info = {
            "Log Directory": session_manager.get_value("log_directory")
            or "Not set",
            "Auto-startup": session_manager.get_value(
                "auto_startup_enabled", True
            ),
            "User Stopped": session_manager.get_value(
                "user_initiated_stop", False
            ),
        }

        for key, value in session_info.items():
            st.text(f"  {key}: {value}")
