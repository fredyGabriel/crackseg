"""
Training page for the CrackSeg application.

This module contains the training dashboard content for launching
and monitoring model training.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from scripts.gui.components.tensorboard_component import TensorBoardComponent
from scripts.gui.utils.session_state import SessionStateManager


def page_train() -> None:
    """Training page content."""
    state = SessionStateManager.get()

    if not state.is_ready_for_training():
        st.warning("Please complete configuration setup before training.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "▶️ Start Training", type="primary", disabled=state.training_active
        ):
            state.set_training_active(True)
            state.add_notification("Training started")
            st.success("Training started!")

    with col2:
        if st.button("⏸️ Pause Training", disabled=not state.training_active):
            state.add_notification("Training paused")
            st.info(
                "Pause functionality will be implemented in future updates"
            )

    with col3:
        if st.button("⏹️ Stop Training", disabled=not state.training_active):
            state.set_training_active(False)
            state.add_notification("Training stopped")
            st.info("Training stopped")

    # Training progress
    st.markdown("---")
    st.subheader("Training Progress")

    if state.training_active:
        st.progress(state.training_progress)
        st.info(f"Training in progress: {state.training_progress:.1%}")

        # Mock progress update for demo
        if st.button("Simulate Progress", key="sim_progress"):
            new_progress = min(1.0, state.training_progress + 0.1)
            state.update_training_progress(
                new_progress, {"loss": 0.5 - new_progress * 0.3}
            )
    else:
        st.info("Real-time training metrics will be displayed here")

    # Training metrics
    if state.training_metrics:
        st.markdown("### Current Metrics")
        for metric, value in state.training_metrics.items():
            st.metric(metric.capitalize(), f"{value:.4f}")

    # TensorBoard integration during training
    if state.training_active:
        st.markdown("---")
        _render_training_tensorboard(state)


def _render_training_tensorboard(state: Any) -> None:
    """Render compact TensorBoard integration during training."""
    run_dir = getattr(state, "run_dir", None)

    if run_dir is None:
        return

    log_dir = Path(run_dir) / "logs" / "tensorboard"

    # Compact TensorBoard component for training page
    with st.expander("📊 TensorBoard Live Monitoring", expanded=False):
        tb_component = TensorBoardComponent(
            default_height=500,  # Smaller for training page
            auto_startup=True,  # Auto-start when available
            show_controls=False,  # Minimal controls during training
            show_status=True,  # Show status
        )

        tb_component.render(
            log_dir=log_dir, title="Live Training Metrics", show_refresh=True
        )
