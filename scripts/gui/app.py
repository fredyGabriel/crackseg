"""
CrackSeg - Pavement Crack Segmentation GUI
Main entry point for the Streamlit application (Refactored)

This is the refactored version with modular architecture for better
maintainability.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PyTorch & Streamlit Hotfix
#
# Addresses a conflict between Streamlit's file watcher and PyTorch's
# JIT compiler when using `torch.classes`. This can cause a
# `RuntimeError: no running event loop` on startup.
#
# This workaround explicitly sets the path for `torch.classes`, preventing
# the watcher from triggering the problematic code path.
#
# See related issues:
# - https://discuss.streamlit.io/t/runtimeerror-no-running-event-loop/27287
# - https://blog.csdn.net/m0_53115174/article/details/146381953
# =============================================================================
import os

import torch

try:
    # This attribute is dynamically created and may not exist in all versions
    if (
        hasattr(torch, "classes")
        and hasattr(torch.classes, "__file__")
        and torch.classes.__file__ is not None
    ):
        torch.classes.__path__ = [
            os.path.join(torch.__path__[0], torch.classes.__file__)
        ]
except (AttributeError, FileNotFoundError):
    # If the attributes don't exist, this patch is likely not needed.
    # We pass silently to avoid breaking on future PyTorch/Streamlit versions.
    pass
# =============================================================================


# Import modular components  # ruff: noqa: E402
from scripts.gui.components.page_router import PageRouter
from scripts.gui.components.sidebar_component import render_sidebar
from scripts.gui.components.theme_component import ThemeComponent
from scripts.gui.pages import (
    page_advanced_config,
    page_architecture,
    page_config,
    page_results,
    page_train,
)
from scripts.gui.utils.session_state import SessionStateManager

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="CrackSeg - Crack Segmentation",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/crackseg",
        "Report a bug": "https://github.com/yourusername/crackseg/issues",
        "About": (
            "# CrackSeg\nA deep learning application for pavement "
            "crack segmentation"
        ),
    },
)


def initialize_session_state() -> None:
    """Initialize session state variables using the SessionStateManager."""
    SessionStateManager.initialize()


def main() -> None:
    """Main application entry point with enhanced routing."""
    # Initialize session state
    SessionStateManager.initialize()

    # Apply current theme
    ThemeComponent.apply_current_theme()

    # Get current state
    state = SessionStateManager.get()

    # Render sidebar and get current page
    current_page = render_sidebar(PROJECT_ROOT)

    # Display breadcrumb navigation
    breadcrumbs = PageRouter.get_page_breadcrumbs(current_page)
    st.markdown(f"**Navigation:** {breadcrumbs}")
    st.markdown("---")

    # Page function mapping
    page_functions = {
        "page_config": page_config,
        "page_advanced_config": page_advanced_config,
        "page_architecture": page_architecture,
        "page_train": page_train,
        "page_results": page_results,
    }

    # Use PageRouter to handle page rendering
    PageRouter.route_to_page(current_page, state, page_functions)


if __name__ == "__main__":
    main()
