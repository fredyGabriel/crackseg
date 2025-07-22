"""
Results page for the CrackSeg application. This module contains the
results visualization page that allows users to: - View TensorBoard
training metrics - Browse prediction triplets (Image | Mask |
Prediction) - Analyze model performance metrics - Compare different
model performances - Export results in various formats The page is
structured with reactive gallery integration and proper error handling
for missing training data.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from gui.components.header_component import render_header
from gui.utils.session_state import SessionStateManager

from .comparison_section import render_comparison_tab
from .config_section import render_config_panel
from .gallery_section import render_gallery_tab
from .metrics_section import render_metrics_tab
from .setup_section import render_setup_guide
from .tensorboard_section import render_tensorboard_tab


def page_results() -> None:
    """Professional results visualization page with reactive gallery."""
    state = SessionStateManager.get()

    # Page header
    render_header("Prediction Results & Analysis")

    # Check readiness
    if not state.is_ready_for_results():
        render_setup_guide()
        return

    # Main results interface
    _render_results_interface(state)


def _render_results_interface(state: Any) -> None:
    """Render the main results interface with tabs."""
    # Get run directory for scans
    run_dir = getattr(state, "run_dir", None)
    predictions_dir = None

    if run_dir:
        # Look for common prediction directories
        run_path = Path(run_dir)
        possible_pred_dirs = [
            run_path / "predictions",
            run_path / "outputs" / "predictions",
            run_path / "results",
            run_path / "inference",
        ]

        for pred_dir in possible_pred_dirs:
            if pred_dir.exists():
                predictions_dir = pred_dir
                break

    # Configuration panel
    with st.expander("Configuration", expanded=False):
        render_config_panel(predictions_dir)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Results Gallery",
            "TensorBoard",
            "Metrics Analysis",
            "Model Comparison",
        ]
    )

    with tab1:
        render_gallery_tab(predictions_dir)

    with tab2:
        render_tensorboard_tab(state)

    with tab3:
        render_metrics_tab(state)

    with tab4:
        render_comparison_tab(state)
