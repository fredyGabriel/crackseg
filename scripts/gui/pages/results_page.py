"""
Results page for the CrackSeg application.

This module contains the results visualization page content
for analyzing predictions and exporting reports.
"""

import streamlit as st

from scripts.gui.utils.session_state import SessionStateManager


def page_results() -> None:
    """Results visualization page content."""
    state = SessionStateManager.get()

    if not state.is_ready_for_results():
        st.warning("Please complete training or set a run directory first.")
        return

    # Results tabs
    tab1, tab2, tab3 = st.tabs(
        ["Metrics", "Visualizations", "Model Comparison"]
    )

    with tab1:
        st.subheader("Training Metrics")
        if state.last_evaluation:
            st.json(state.last_evaluation)
        else:
            st.info(
                "Training metrics visualization will be implemented in future "
                "updates"
            )

    with tab2:
        st.subheader("Segmentation Results")
        st.info(
            "Segmentation visualization will be implemented in future updates"
        )

    with tab3:
        st.subheader("Model Comparison")
        st.info("Model comparison tools will be implemented in future updates")
