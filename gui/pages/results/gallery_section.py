"""
Gallery section for the results page.

This module handles the results gallery tab with triplet visualization,
including gallery rendering, statistics display, and quick actions.
"""

from pathlib import Path

import streamlit as st

from scripts.gui.components.results_gallery_component import (
    ResultsGalleryComponent,
)
from scripts.gui.utils.results import ValidationLevel

from .utils import show_gallery_summary


def render_gallery_tab(predictions_dir: Path | None) -> None:
    """Render the results gallery tab with triplet visualization."""
    # Determine scan directory
    scan_directory = None
    if st.session_state.get("custom_predictions_dir"):
        scan_directory = st.session_state["custom_predictions_dir"]
    elif predictions_dir:
        scan_directory = predictions_dir

    # Get settings from session state
    validation_level = st.session_state.get(
        "global_validation_level", ValidationLevel.STANDARD
    )
    max_triplets = st.session_state.get("global_max_triplets", 50)
    grid_columns = st.session_state.get("global_grid_columns", 3)
    enable_real_time = st.session_state.get("global_real_time", True)

    # Create and render gallery component
    gallery = ResultsGalleryComponent()

    gallery.render(
        scan_directory=scan_directory,
        validation_level=validation_level,
        max_triplets=max_triplets,
        grid_columns=grid_columns,
        show_validation_panel=True,
        show_export_panel=True,
        enable_real_time_scanning=enable_real_time,
    )

    gallery_state = gallery.ui_state

    # Display gallery statistics in sidebar
    if gallery_state["total_triplets"] > 0:
        with st.sidebar:
            st.markdown("### **Gallery Stats**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", gallery_state["total_triplets"])
                st.metric("Valid", gallery_state["valid_triplets"])

            with col2:
                st.metric("Selected", len(gallery_state["selected_triplets"]))
                cache_stats = gallery_state["cache_stats"]
                st.metric(
                    "Cache Hit", f"{cache_stats.get('hit_rate', 0):.1f}%"
                )

            # Quick actions
            st.markdown("### **Quick Actions**")

            if st.button("Refresh Gallery", use_container_width=True):
                # Clear cache and rerun
                st.cache_data.clear()
                st.rerun()

            if gallery_state["total_triplets"] > 0:
                if st.button("View Summary", use_container_width=True):
                    show_gallery_summary(gallery_state)
