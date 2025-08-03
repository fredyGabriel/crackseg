"""
Training page for the CrackSeg application. This module contains the
training dashboard content for launching and monitoring model
training.
"""

import time
from pathlib import Path
from typing import Any

import streamlit as st

from gui.components.header_component import render_header
from gui.components.loading_spinner import LoadingSpinner
from gui.components.log_viewer import LogViewerComponent
from gui.components.progress_bar import (
    ProgressBar,
)
from gui.components.tensorboard_component import TensorBoardComponent
from gui.utils.gui_config import PAGE_CONFIG
from gui.utils.log_parser import (
    initialize_metrics_df,
)
from gui.utils.process_manager import ProcessManager
from gui.utils.session_state import SessionStateManager


def get_command(state: Any) -> list[str]:
    """Constructs the training command."""
    # Note: Assumes conda environment is active where streamlit is run
    command = [
        "python",
        "-m",
        "src.train",
        f"data.data_dir={Path('data').absolute()}",
        f"hydra.run.dir={state.run_directory}",
    ]
    # Add overrides from the loaded config
    # This is a simplified example. A real implementation would parse the
    # config.
    command.append(f"--config-name={Path(state.config_path).stem}")
    command.append(
        f"--config-path={str(Path(state.config_path).parent.absolute())}"
    )
    return command


def _render_training_controls(process_manager: Any) -> None:
    """Render training control buttons for start/stop/pause operations."""
    st.subheader("Training Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_training = st.button(
            "▶️ Start Training",
            type="primary",
            disabled=process_manager.is_running,
            help="Start the training process with current configuration",
            key="start_training_btn",
        )

    with col2:
        pause_training = st.button(
            "⏸️ Pause Training",
            disabled=not process_manager.is_running,
            help="Pause the current training session",
            key="pause_training_btn",
        )

    with col3:
        stop_training = st.button(
            "⏹️ Stop Training",
            disabled=not process_manager.is_running,
            help="Stop the training process completely",
            key="stop_training_btn",
        )

    # Handle training actions
    if start_training:
        try:
            state = SessionStateManager.get()
            command = get_command(state)
            success = process_manager.start_training(command)

            if success:
                st.success("Training started successfully!")
                st.session_state.process_manager = process_manager
                st.rerun()
            else:
                st.error("Failed to start training. Check configuration.")

        except Exception as e:
            st.error(f"Error starting training: {str(e)}")

    if pause_training:
        try:
            process_manager.pause()
            st.warning("Training paused.")
            st.rerun()
        except Exception as e:
            st.error(f"Error pausing training: {str(e)}")

    if stop_training:
        try:
            process_manager.stop()
            st.info("Training stopped.")
            st.session_state.process_manager = None
            st.rerun()
        except Exception as e:
            st.error(f"Error stopping training: {str(e)}")


def page_train() -> None:
    """
    Renders the training page, handling the lifecycle of a training
    process. This page facilitates starting, stopping, and monitoring the
    training process. It displays real-time logs, metrics, and manages the
    training lifecycle.
    """
    st.title(PAGE_CONFIG["Train"]["title"])
    state = SessionStateManager.get()
    # Create ProcessManager with proper initialization
    process_manager = ProcessManager([])
    render_header("Model Training")

    # Initialize process manager and metrics dataframe in session state
    if "process_manager" not in st.session_state:
        st.session_state.process_manager = None
    if "metrics_df" not in st.session_state:
        st.session_state.metrics_df = initialize_metrics_df()

    # Check if the system is ready for training
    ready = state.is_ready_for_training()
    if not ready:
        st.error(
            "System is not ready for training. Please resolve the following "
            "issues:"
        )
        st.warning("- Check configuration and directory settings")
        st.stop()

    # --- Training Control ---
    _render_training_controls(process_manager)

    # --- Log and Metrics Display ---
    if process_manager.is_running:
        st.markdown("---")
        log_col, metrics_col = st.columns(2)

        with log_col:
            st.subheader("Live Training Log")
            # Create log viewer with basic implementation
            log_viewer = LogViewerComponent()
            log_viewer.render()

        with metrics_col:
            st.subheader("Live Metrics")
            if not st.session_state.metrics_df.empty:
                # Plot main metrics (loss)
                loss_cols: list[str] = [
                    str(c)  # type: ignore
                    for c in st.session_state.metrics_df.columns  # type: ignore
                    if "loss" in str(c)  # type: ignore
                ]
                if loss_cols:
                    st.line_chart(st.session_state.metrics_df[loss_cols])  # type: ignore

                # Plot other validation metrics
                other_val_cols: list[str] = [
                    str(c)  # type: ignore
                    for c in st.session_state.metrics_df.columns  # type: ignore
                    if "val_" in str(c) and "loss" not in str(c)  # type: ignore
                ]
                if other_val_cols:
                    st.line_chart(st.session_state.metrics_df[other_val_cols])  # type: ignore
            else:
                st.info(
                    "Metrics will be plotted here as they become available."
                )

        # Auto-refresh loop to update logs and charts
        time.sleep(2)  # Increased sleep time to allow chart rendering
        process_manager.check_status()  # Update is_running flag
        if process_manager.is_running:
            st.rerun()
        else:
            # Final status update when process finishes
            st.success("Training process finished.")
            st.session_state.process_manager = None
            st.rerun()  # One final rerun to clear the page

    else:
        # Default view when no process is running
        st.header("Start a New Training Run")
        st.info(
            "Configure your experiment on the 'Config' page, then click "
            "'Start Training' above."
        )

    # Training progress
    st.markdown("---")
    st.subheader("Training Progress")

    if process_manager.is_running:
        _render_training_progress(state)
    else:
        st.info("Real-time training metrics will be displayed here")

    # Training metrics
    if state.training_metrics:
        st.markdown("### Current Metrics")
        for metric, value in state.training_metrics.items():
            st.metric(metric.capitalize(), f"{value:.4f}")

    # TensorBoard integration during training
    if process_manager.is_running:
        st.markdown("---")
        _render_training_tensorboard(state)


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
    with st.expander("TensorBoard Live Monitoring", expanded=False):
        tb_component = TensorBoardComponent(
            default_height=500,  # Smaller for training page
            auto_startup=True,  # Auto-start when available
            show_controls=False,  # Minimal controls during training
            show_status=True,  # Show status
        )

        tb_component.render(
            log_dir=log_dir, title="Live Training Metrics", show_refresh=True
        )
