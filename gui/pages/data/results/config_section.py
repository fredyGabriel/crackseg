"""
Configuration section for the results page. This module handles the
configuration panel for results display settings, including directory
selection, validation settings, and display options.
"""

from pathlib import Path

import streamlit as st

from gui.utils.results import ValidationLevel


def render_config_panel(predictions_dir: Path | None) -> None:
    """Render configuration panel for results display."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### **Scan Directory**")
        if predictions_dir:
            st.success(f"Auto-detected: `{predictions_dir}`")
        else:
            st.warning("No predictions directory found")

        # Manual directory input
        custom_dir = st.text_input(
            "Custom Directory",
            value=str(predictions_dir) if predictions_dir else "",
            help="Override auto-detected predictions directory",
        )

        if custom_dir and Path(custom_dir).exists():
            st.session_state["custom_predictions_dir"] = custom_dir
            st.success(f"Custom directory set: `{custom_dir}`")
        elif custom_dir:
            st.error("Directory does not exist")

    with col2:
        st.markdown("### **Validation Settings**")

        validation_level = st.selectbox(
            "Validation Level",
            options=[level.name for level in ValidationLevel],
            index=1,  # Default to STANDARD
            help="Choose validation thoroughness for triplet scanning",
        )
        st.session_state["global_validation_level"] = ValidationLevel[
            validation_level
        ]

        max_triplets = st.number_input(
            "Max Triplets",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Maximum number of triplets to display",
        )
        st.session_state["global_max_triplets"] = max_triplets

    with col3:
        st.markdown("### **Display Settings**")

        grid_columns = st.selectbox(
            "Grid Columns",
            options=[2, 3, 4, 5],
            index=1,  # Default to 3
            help="Number of columns in gallery grid",
        )
        st.session_state["global_grid_columns"] = grid_columns

        enable_real_time = st.checkbox(
            "Real-time Updates",
            value=True,
            help="Enable real-time gallery updates during scanning",
        )
        st.session_state["global_real_time"] = enable_real_time
