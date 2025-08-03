"""
GUI Home Page - Dashboard This module defines the layout and content
of the application's main dashboard.
"""

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from gui.components.header_component import render_header
from gui.utils.data_stats import get_dataset_image_counts
from gui.utils.gui_config import PAGE_CONFIG
from gui.utils.session_state import SessionState, SessionStateManager


def render_project_overview(state: SessionState) -> None:
    """
    Renders the main project overview section with title and description.
    Args: state: The current session state (unused, for consistency).
    """
    st.title("CrackSeg: Pavement Crack Segmentation")
    st.markdown(
        """
Welcome to the central dashboard for the CrackSeg project. This GUI
allows you to configure experiments, launch training runs, and analyze
results.
"""
    )
    st.markdown("---")


def render_quick_actions(
    state: SessionState, navigate_to: Callable[[str], None]
) -> None:
    """
    Renders a section with quick action buttons for core workflows. Args:
    state: The current session state. navigate_to: A callback function to
    handle page navigation, which takes the target page name as a string.
    """
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start New Training", use_container_width=True):
            navigate_to("Train")
    with col2:
        if st.button("View Latest Results", use_container_width=True):
            navigate_to("Results")
    with col3:
        if st.button("Configure Architecture", use_container_width=True):
            navigate_to("Config")


def render_statistics(state: SessionState) -> None:
    """
    Renders a section for key dataset statistics. This function scans the
    `data/` directory to count images in the train, validation, and test
    sets. Args: state: The current session state (unused, for
    consistency).
    """
    st.subheader("📊 Dataset Statistics")

    data_root = Path("data")  # Assuming the app runs from the project root
    if not data_root.is_dir():
        st.warning("`data` directory not found. Cannot calculate stats.")
        return

    counts = get_dataset_image_counts(data_root)
    total_images = sum(counts.values())

    st.metric(label="Total Images", value=total_images)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Training Images", value=counts.get("train", 0))
    with col2:
        st.metric(label="Validation Images", value=counts.get("val", 0))
    with col3:
        st.metric(label="Test Images", value=counts.get("test", 0))


def page_home() -> None:
    """
    Renders the entire home page dashboard, including overview, actions,
    and stats. It defines the navigation logic and orchestrates the layout
    of the page. Gets the session state internally for consistency with
    other pages.
    """
    # Get state internally like other pages
    state = SessionStateManager.get()

    st.title(PAGE_CONFIG["Home"]["title"])
    st.markdown(
        "Welcome to the CrackSeg project. "
        "Use the sidebar to navigate through the application."
    )
    render_header("Home Dashboard")
    render_project_overview(state)

    def navigate_to_page(page_name: str) -> None:
        """
        Callback function to handle navigation. Updates the session state and
        triggers a rerun.
        """
        if st.session_state:
            state.current_page = page_name
            st.rerun()

    col1, col2 = st.columns((2, 1))

    with col1:
        render_quick_actions(state, navigate_to_page)

    with col2:
        render_statistics(state)


if __name__ == "__main__":
    # This block is for isolated testing of the home page component.
    # It creates a mock session state and renders the page.
    st.set_page_config(layout="wide")

    # Add project root to path for local testing of data stats
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT))

    from gui.utils.session_state import SessionState

    # Initialize a mock session state
    if "session_state_initialized" not in st.session_state:
        mock_state = SessionState(current_page="Home")
        st.session_state.app_state = mock_state
        st.session_state.session_state_initialized = True

    app_state = st.session_state.app_state

    page_home()

    st.write("Current Page in State:", app_state.current_page)
