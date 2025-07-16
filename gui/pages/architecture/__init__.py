"""
Architecture page for the CrackSeg application.

This module contains the architecture visualization page that allows users to:
- Load model configurations
- Instantiate models from configurations
- Generate and display architecture diagrams
- View model summaries and statistics

Includes proper error handling for Graphviz installation and model
instantiation.
"""

import streamlit as st

from scripts.gui.utils.gui_config import PAGE_CONFIG
from scripts.gui.utils.session_state import SessionStateManager

from .config_section import render_configuration_selection
from .info_section import render_model_information_section
from .model_section import render_model_instantiation_section
from .visualization_section import render_architecture_visualization_section


def page_architecture() -> None:
    """Architecture visualization page content."""
    st.title(PAGE_CONFIG["Architecture"]["title"])
    st.header("Model Architecture Viewer")

    # Get session state
    state = SessionStateManager.get()

    # Configuration selection section
    render_configuration_selection()

    # Model instantiation section
    if state.config_path:
        render_model_instantiation_section()
    else:
        st.info("Please select a configuration file to proceed")
        return

    # Architecture visualization section
    if state.model_loaded and state.current_model is not None:
        render_architecture_visualization_section()

    # Model information section
    if state.model_loaded and state.current_model is not None:
        render_model_information_section()


# Export main function for backward compatibility
__all__ = ["page_architecture"]
