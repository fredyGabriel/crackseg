"""
Metrics analysis section for the results page.

This module handles the metrics analysis tab with performance summaries,
detailed metrics display, and visualization placeholders.
"""

from typing import Any

import streamlit as st


def render_metrics_tab(state: Any) -> None:
    """Render metrics analysis tab."""
    st.subheader("Training Metrics Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### **Performance Summary**")

        if state.last_evaluation:
            # Display evaluation metrics
            eval_data = state.last_evaluation

            # Create metrics display
            metrics_container = st.container()
            with metrics_container:
                metric_cols = st.columns(3)

                # Extract common metrics
                if "iou" in eval_data:
                    with metric_cols[0]:
                        st.metric("IoU Score", f"{eval_data['iou']:.4f}")

                if "dice" in eval_data:
                    with metric_cols[1]:
                        st.metric("Dice Score", f"{eval_data['dice']:.4f}")

                if "loss" in eval_data:
                    with metric_cols[2]:
                        st.metric(
                            "Validation Loss", f"{eval_data['loss']:.4f}"
                        )

            # Display full evaluation data
            st.markdown("### **Detailed Metrics**")
            st.json(eval_data)

        else:
            st.info(
                "No evaluation metrics available. "
                "Complete training to see results."
            )

    with col2:
        st.markdown("### **Metrics Visualization**")
        st.info("Advanced metrics visualization coming soon!")

        st.markdown(
            """
        **Planned Features:**
        - Loss curves over time
        - Metric comparisons across epochs
        - Performance distribution plots
        - Learning rate scheduling visualization
        """
        )
