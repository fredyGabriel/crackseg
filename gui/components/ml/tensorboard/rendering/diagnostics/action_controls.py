"""Action controls for TensorBoard status interface."""

from typing import Any

import streamlit as st


class ActionControls:
    """Panel for TensorBoard action controls and buttons."""

    def __init__(self) -> None:
        """Initialize action controls."""
        pass

    def render(self, manager: Any, session_manager: Any) -> None:
        """Render quick status action buttons.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.
        """
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(
                "🔄 Refresh Status", help="Refresh all status indicators"
            ):
                # Trigger health check
                if manager.is_running:
                    # Force health check update
                    manager.info.update_health(True)
                st.rerun()

        with col2:
            if manager.is_running and manager.info.url:
                if st.button(
                    "🔗 Open TensorBoard", help="Open TensorBoard in new tab"
                ):
                    st.markdown(
                        f'<a href="{manager.info.url}" target="_blank">'
                        "Open TensorBoard</a>",
                        unsafe_allow_html=True,
                    )

        with col3:
            if st.button("📊 View Metrics", help="Show detailed metrics"):
                session_manager.update_state(show_detailed_metrics=True)
                st.rerun()

        with col4:
            if st.button(
                "🧹 Reset State", help="Reset error state and counters"
            ):
                session_manager.reset_startup_state()
                manager.info.reset_startup_attempts()
                st.success("Status state reset successfully")
                st.rerun()

    def render_compact(self, manager: Any, session_manager: Any) -> None:
        """Render compact action controls for limited space.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.
        """
        col1, col2 = st.columns([1, 1])

        with col1:
            if manager.is_running and manager.info.url:
                if st.button(
                    "🔗 Open",
                    help="Open TensorBoard in new tab",
                    key="open_tb",
                ):
                    st.markdown(
                        f'<a href="{manager.info.url}" target="_blank">'
                        "Open TensorBoard</a>",
                        unsafe_allow_html=True,
                    )

        with col2:
            if st.button("🔄", help="Refresh status", key="refresh_status"):
                st.rerun()
