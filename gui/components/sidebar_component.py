"""
Sidebar component for the CrackSeg application.

This module provides professional sidebar navigation with enhanced
functionality and visual status indicators.
"""

from pathlib import Path

import streamlit as st

from scripts.gui.components.logo_component import LogoComponent
from scripts.gui.components.page_router import PageRouter
from scripts.gui.components.theme_component import ThemeComponent
from scripts.gui.utils.gui_config import PAGE_CONFIG
from scripts.gui.utils.session_state import SessionStateManager


def render_sidebar(project_root: Path) -> str:
    """Render sidebar with navigation and logo.

    Args:
        project_root: Project root directory for logo fallback paths

    Returns:
        Currently selected page name
    """
    state = SessionStateManager.get()

    with st.sidebar:
        # Logo
        LogoComponent.render(
            primary_path=state.config_path,
            style=state.theme,
            width=150,
            alt_text="CrackSeg Logo",
            css_class="logo",
            center=True,
            project_root=project_root,
        )

        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Navigation")

        # Get available pages from PageRouter
        available_pages = PageRouter.get_available_pages(state)
        all_pages = list(PAGE_CONFIG.keys())

        # Display page status
        for page in all_pages:
            config = PAGE_CONFIG[page]
            icon = config["icon"]
            description = config["description"]
            is_available = page in available_pages
            is_current = page == state.current_page

            if is_current:
                st.success(f"**➤ {icon} {page}** (Current)")
            elif is_available:
                if st.button(
                    page,
                    key=f"nav_btn_{page}",
                    use_container_width=True,
                    help=str(description) if description else None,
                ):
                    if PageRouter.handle_navigation_change(page, state):
                        st.rerun()
            else:
                st.warning(f"⚠️ {icon} {page} (Needs setup)")
                # Show requirements
                requirements = config.get("requires", [])
                req_messages = []
                for req in requirements:
                    if req == "config_loaded" and not state.config_loaded:
                        req_messages.append("Load configuration")
                    elif req == "run_directory" and not state.run_directory:
                        req_messages.append("Set run directory")
                if req_messages:
                    st.caption(f"Required: {', '.join(req_messages)}")

        st.markdown("---")

        # Status Panel
        st.markdown("### 📋 Status")

        # Configuration status
        if state.config_loaded:
            st.success("📄 Configuration: Loaded")
            if state.config_path:
                st.caption(f"Path: {Path(state.config_path).name}")
        else:
            st.error("📄 Configuration: Not loaded")

        # Run directory status
        if state.run_directory:
            st.success("📁 Run Directory: Set")
            st.caption(f"Path: {Path(state.run_directory).name}")
        else:
            st.error("📁 Run Directory: Not set")

        # Training status
        if state.training_active:
            st.info(f"🚀 Training: Active ({state.training_progress:.1%})")
            st.progress(state.training_progress)
        else:
            st.warning("🚀 Training: Inactive")

        st.markdown("---")

        # Notifications
        if state.notifications:
            st.markdown("### 🔔 Recent Activity")
            for notification in state.notifications[-3:]:  # Show last 3
                st.caption(notification)

            if st.button(
                "🗑️ Clear Notifications",
                key="clear_notifications",
                use_container_width=True,
            ):
                state.clear_notifications()
                st.rerun()

        # Quick Actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🔄 Refresh", key="refresh_app", use_container_width=True
            ):
                st.rerun()

        with col2:
            readiness = state.is_ready_for_training()
            if readiness:
                st.success("✅ Ready")
            else:
                st.error("❌ Not Ready")

        # Theme selector
        st.markdown("---")
        ThemeComponent.render_quick_theme_switcher()

        # App info
        st.markdown("---")
        st.caption("CrackSeg v1.0")
        st.caption("AI-Powered Crack Detection")

    return state.current_page
