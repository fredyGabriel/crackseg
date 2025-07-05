"""
Training page for the CrackSeg application.

This module contains the training dashboard content for launching
and monitoring model training.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from scripts.gui.components.loading_spinner import LoadingSpinner
from scripts.gui.components.progress_bar import (
    ProgressBar,
    create_step_progress,
)
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
        start_training_button = st.button(
            "▶️ Start Training",
            key="launch_training",
            type="primary",
            disabled=state.training_active,
        )

        if start_training_button:
            _start_training_process(state)

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
        _render_training_progress(state)
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


def _start_training_process(state: Any) -> None:
    """Start the training process with progress tracking."""
    # Use LoadingSpinner for quick initialization
    message, subtext, spinner_type = LoadingSpinner.get_contextual_message(
        "training"
    )

    with LoadingSpinner.spinner(
        message=message,
        subtext=subtext,
        spinner_type=spinner_type,
        timeout_seconds=10,
    ):
        # Quick initialization
        state.set_training_active(True)
        state.add_notification("Training initialization started")

    # Use ProgressBar for the actual training process
    training_steps = [
        "Loading model architecture",
        "Initializing data loaders",
        "Setting up optimizer",
        "Configuring loss functions",
        "Starting training loop",
    ]

    # Create step-based progress for training setup
    with create_step_progress(
        title="Training Setup",
        steps=training_steps,
        operation_id="training_setup",
    ) as progress:
        for step in training_steps:
            progress.next_step(f"Completing: {step}")
            # Simulate setup time
            import time

            time.sleep(1)

    st.success("Training started!")
    st.rerun()


def _render_training_progress(state: Any) -> None:
    """Render training progress with both spinner and progress bar."""
    # Show enhanced progress with spinner for active updates
    LoadingSpinner.show_progress_with_spinner(
        message="Training in progress",
        progress=state.training_progress,
        subtext=f"Epoch progress: {state.training_progress:.1%} complete",
        spinner_type="crack_pattern",
    )

    # Show detailed progress bar for training epochs
    if hasattr(state, "training_epoch_progress"):
        training_progress = ProgressBar("training_progress")
        training_progress.start(
            title="Training Progress",
            total_steps=getattr(state, "total_epochs", 100),
            description="Deep learning model training in progress",
        )

        training_progress.update(
            progress=state.training_progress,
            current_step=getattr(state, "current_epoch", 0),
            step_description=(
                f"Epoch {getattr(state, 'current_epoch', 0)}/"
                f"{getattr(state, 'total_epochs', 100)}"
            ),
        )

    # Mock progress update for demo
    if st.button("Simulate Progress", key="sim_progress"):
        new_progress = min(1.0, state.training_progress + 0.1)
        state.update_training_progress(
            new_progress, {"loss": 0.5 - new_progress * 0.3}
        )


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
