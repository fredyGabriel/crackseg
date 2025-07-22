"""
TensorBoard section for the results page. This module handles the
TensorBoard tab with integrated iframe embedding for training
visualization and monitoring.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from gui.components.tensorboard_component import TensorBoardComponent


def render_tensorboard_tab(state: Any) -> None:
    """Render TensorBoard tab with integrated iframe embedding."""
    run_dir = getattr(state, "run_dir", None)

    if run_dir is None:
        st.info(
            "No active run directory. Please configure a run directory first."
        )
        return

    # Construct log directory path
    log_dir = Path(run_dir) / "logs" / "tensorboard"

    # Create and render TensorBoard component
    tb_component = TensorBoardComponent(
        default_height=700,
        auto_startup=True,
        show_controls=True,
        show_status=True,
    )

    # Render with custom title
    tb_component.render(
        log_dir=log_dir,
        title="TensorBoard Training Visualization",
        show_refresh=True,
    )
