"""Startup progress rendering for TensorBoard component."""

from typing import Any

import streamlit as st


def render_startup_progress(
    container: Any, attempts: int, max_attempts: int
) -> None:
    """Render startup progress indicator.

    Args:
        container: Streamlit container for progress display.
        attempts: Current attempt number.
        max_attempts: Maximum attempts allowed.
    """
    with container:
        st.info(
            f"🚀 Starting TensorBoard... (attempt {attempts}/{max_attempts})"
        )
        progress_bar = st.progress(0.1)
        progress_bar.progress(0.5)
