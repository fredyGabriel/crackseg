"""
Model information section for the architecture page. This module
handles the display of model information, statistics, and detailed
component breakdown.
"""

import streamlit as st

from gui.utils.architecture_viewer import format_model_summary
from gui.utils.session_state import SessionStateManager


def render_model_information_section() -> None:
    """Render the model information and statistics section."""
    st.subheader("Model Information")

    state = SessionStateManager.get()

    if not state.model_summary:
        st.info("No model summary available")
        return

    # Create tabs for different information sections
    tab1, tab2, tab3 = st.tabs(["Summary", "Components", "Details"])

    with tab1:
        # Model summary
        summary_text = format_model_summary(state.model_summary)
        st.markdown(summary_text)

        # Parameter breakdown
        if "total_params" in state.model_summary:
            total = state.model_summary["total_params"]
            trainable = state.model_summary.get("trainable_params", total)
            frozen = total - trainable

            st.markdown("### Parameter Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", f"{total:,}")
            with col2:
                st.metric("Trainable", f"{trainable:,}")
            with col3:
                st.metric("Frozen", f"{frozen:,}")

    with tab2:
        # Architecture components
        components = {}
        for key in [
            "encoder_type",
            "bottleneck_type",
            "decoder_type",
            "final_activation_type",
        ]:
            if key in state.model_summary:
                component_name = (
                    key.replace("_type", "").replace("_", " ").title()
                )
                components[component_name] = state.model_summary[key]

        if components:
            for component, type_name in components.items():  # type: ignore[assignment]
                st.markdown(f"**{component}:** {type_name}")
        else:
            st.info("Component information not available")

    with tab3:
        # Detailed information
        st.json(state.model_summary)
