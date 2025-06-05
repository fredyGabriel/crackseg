"""
Architecture page for the CrackSeg application.

This module contains the architecture visualization page content
for viewing and understanding model structure.
"""

import streamlit as st

from scripts.gui.utils.session_state import SessionStateManager


def page_architecture() -> None:
    """Architecture visualization page content."""
    state = SessionStateManager.get()

    # Model status
    if state.model_loaded:
        st.success(f"✅ Model loaded: {state.model_architecture}")
    else:
        st.info("No model currently loaded")

    # Placeholder content
    st.info("Architecture visualization will be implemented in future updates")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Components")
        st.markdown(
            """
        - **Encoder**: Feature extraction backbone
        - **Bottleneck**: Feature processing
        - **Decoder**: Segmentation head
        """
        )

    with col2:
        st.subheader("Model Statistics")
        if state.model_parameters:
            st.json(state.model_parameters)
        else:
            st.markdown(
                """
            - **Parameters**: TBD
            - **FLOPs**: TBD
            - **Input Size**: TBD
            """
            )
