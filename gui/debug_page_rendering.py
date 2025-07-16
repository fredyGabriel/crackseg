"""
Debug script for rendering individual GUI pages.

This script allows calling a page's rendering function directly
to isolate errors and exceptions that might cause Streamlit to crash.

Usage:
    streamlit run scripts/gui/debug_page_rendering.py -- <page_name>

Example:
    streamlit run scripts/gui/debug_page_rendering.py -- architecture
"""

import sys
import traceback
from collections.abc import Callable
from pathlib import Path

import streamlit as st

# Import page functions and other necessary components
from scripts.gui.pages import (
    page_advanced_config,
    page_architecture,
    page_config,
    page_home,
    page_results,
    page_train,
)
from scripts.gui.utils.session_state import SessionStateManager

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """
    Main function to select and render a specific page for debugging.
    """
    st.title("GUI Page Debugger")

    # Get page name from command line arguments
    try:
        page_name_to_debug = sys.argv[1]
    except IndexError:
        st.error(
            "Please provide a page name to debug as a command-line argument."
        )
        st.code(
            "streamlit run scripts/gui/debug_page_rendering.py -- <page_name>"
        )
        st.stop()

    # All page functions now have consistent signatures (no parameters)
    pages_without_state: dict[str, Callable[[], None]] = {
        "home": page_home,
        "config": page_config,
        "advanced_config": page_advanced_config,
        "architecture": page_architecture,
        "train": page_train,
        "results": page_results,
    }

    all_pages = list(pages_without_state.keys())

    if page_name_to_debug not in all_pages:
        st.error(f"Page '{page_name_to_debug}' not found.")
        st.write("Available pages:", all_pages)
        st.stop()

    st.info(f"Attempting to render page: '{page_name_to_debug}'")

    try:
        # Initialize session state for the pages to use internally
        SessionStateManager.initialize()

        # All pages now get state internally, so we just call them
        render_func = pages_without_state[page_name_to_debug]
        render_func()

        st.success(
            f"Page '{page_name_to_debug}' rendered successfully "
            "(no exceptions)."
        )

    except Exception as e:
        st.error(
            f"An exception occurred while rendering page "
            f"'{page_name_to_debug}':"
        )
        st.error(f"**Error Type:** {type(e).__name__}")
        st.error(f"**Error Details:** {e}")
        st.subheader("Full Traceback:")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
