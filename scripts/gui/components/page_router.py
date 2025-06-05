"""
Page routing system for the CrackSeg application.

This module provides centralized page routing, navigation validation,
and page metadata management.
"""

from collections.abc import Callable

import streamlit as st

from scripts.gui.utils.gui_config import PAGE_CONFIG
from scripts.gui.utils.session_state import SessionState, SessionStateManager


class PageRouter:
    """Centralized page routing system for the application."""

    # Page component mapping
    _page_components = {
        "Config": "page_config",
        "Advanced Config": "page_advanced_config",
        "Architecture": "page_architecture",
        "Train": "page_train",
        "Results": "page_results",
    }

    # Page metadata for enhanced routing
    _page_metadata = {
        "Config": {
            "title": "🔧 Configuration Manager",
            "subtitle": (
                "Configure your crack segmentation model and training "
                "parameters"
            ),
            "help_text": (
                "Load YAML configurations and set up your training environment"
            ),
        },
        "Advanced Config": {
            "title": "⚙️ Advanced YAML Editor",
            "subtitle": (
                "Edit configurations with syntax highlighting "
                "and live validation"
            ),
            "help_text": (
                "Advanced YAML editor with live validation, templates, "
                "and file management"
            ),
        },
        "Architecture": {
            "title": "🏗️ Model Architecture Viewer",
            "subtitle": "Visualize and understand your model structure",
            "help_text": "Explore model components, layers, and connections",
        },
        "Train": {
            "title": "🚀 Training Dashboard",
            "subtitle": "Launch and monitor model training",
            "help_text": (
                "Start training, view real-time metrics, and manage "
                "checkpoints"
            ),
        },
        "Results": {
            "title": "📊 Results Gallery",
            "subtitle": "Analyze predictions and export reports",
            "help_text": (
                "View segmentation results, metrics, and generate "
                "comprehensive reports"
            ),
        },
    }

    @staticmethod
    def get_available_pages(state: SessionState) -> list[str]:
        """Get list of currently available pages based on state.

        Args:
            state: Current session state

        Returns:
            List of available page names
        """
        available = []

        for page, config in PAGE_CONFIG.items():
            requirements = config.get("requires", [])
            is_available = True

            for req in requirements:
                if req == "config_loaded" and not state.config_loaded:
                    is_available = False
                    break
                elif req == "run_directory" and not state.run_directory:
                    is_available = False
                    break

            if is_available:
                available.append(page)

        return available

    @staticmethod
    def validate_page_transition(
        from_page: str, to_page: str, state: SessionState
    ) -> tuple[bool, str]:
        """Validate if page transition is allowed.

        Args:
            from_page: Current page name
            to_page: Target page name
            state: Current session state

        Returns:
            Tuple of (is_valid, error_message)
        """
        available_pages = PageRouter.get_available_pages(state)

        if to_page not in available_pages:
            requirements = PAGE_CONFIG.get(to_page, {}).get("requires", [])
            missing = []

            for req in requirements:
                if req == "config_loaded" and not state.config_loaded:
                    missing.append("Configuration must be loaded")
                elif req == "run_directory" and not state.run_directory:
                    missing.append("Run directory must be set")

            error_msg = (
                f"Cannot navigate to {to_page}. Missing: {', '.join(missing)}"
            )
            return False, error_msg

        return True, ""

    @staticmethod
    def route_to_page(
        page_name: str,
        state: SessionState,
        page_functions: dict[str, Callable[[], None]],
    ) -> None:
        """Route to the specified page with validation.

        Args:
            page_name: Name of the page to route to
            state: Current session state
            page_functions: Dictionary mapping page names to functions
        """
        # Validate transition
        is_valid, error_msg = PageRouter.validate_page_transition(
            state.current_page, page_name, state
        )

        if not is_valid:
            st.error(error_msg)
            return

        # Update session state
        SessionStateManager.update({"current_page": page_name})
        state.add_notification(f"Navigated to {page_name}")

        # Get page metadata
        metadata = PageRouter._page_metadata.get(page_name, {})

        # Render page header
        if metadata:
            st.title(metadata.get("title", page_name))
            st.markdown(metadata.get("subtitle", ""))

            # Add help expander
            with st.expander("ℹ️ Page Help", expanded=False):
                st.info(metadata.get("help_text", "No help available"))

            st.markdown("---")

        # Route to appropriate page function
        page_function_name = PageRouter._page_components.get(page_name)
        if page_function_name and page_function_name in page_functions:
            page_functions[page_function_name]()
        else:
            st.error(f"Page component not found: {page_name}")

    @staticmethod
    def handle_navigation_change(new_page: str, state: SessionState) -> bool:
        """Handle navigation change with validation and state update.

        Args:
            new_page: New page to navigate to
            state: Current session state

        Returns:
            True if navigation was successful
        """
        if new_page == state.current_page:
            return True

        # Validate transition
        is_valid, error_msg = PageRouter.validate_page_transition(
            state.current_page, new_page, state
        )

        if is_valid:
            SessionStateManager.update({"current_page": new_page})
            state.add_notification(f"Navigated to {new_page}")
            return True
        else:
            st.sidebar.error(error_msg)
            return False

    @staticmethod
    def get_page_breadcrumbs(current_page: str) -> str:
        """Generate breadcrumb navigation for current page.

        Args:
            current_page: Current page name

        Returns:
            Breadcrumb HTML string
        """
        pages = ["Config", "Architecture", "Train", "Results"]
        breadcrumbs = []

        for page in pages:
            if page == current_page:
                breadcrumbs.append(f"**{page}**")
            else:
                breadcrumbs.append(page)

        return " → ".join(breadcrumbs)
