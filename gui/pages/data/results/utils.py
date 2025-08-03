"""
Utility functions for the results page module. This module contains
helper functions and utilities used across the results page
components.
"""

from typing import Any

import streamlit as st


def show_gallery_summary(gallery_state: dict[str, Any]) -> None:
    """Show gallery summary in expandable dialog."""
    with st.expander("Gallery Summary", expanded=True):
        st.markdown("### **Scan Results**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Triplets", gallery_state["total_triplets"])

        with col2:
            st.metric("Valid Triplets", gallery_state["valid_triplets"])

        with col3:
            success_rate = (
                gallery_state["valid_triplets"]
                / gallery_state["total_triplets"]
                * 100.0
                if gallery_state["total_triplets"] > 0
                else 0.0
            )
            st.metric("Success Rate", f"{success_rate:.1f}%")

        st.markdown("### **Selection Summary**")
        selected_count = len(gallery_state["selected_triplets"])
        st.write(f"Selected: **{selected_count}** triplets")

        if selected_count > 0:
            st.markdown("**Selected Items:**")
            for i, triplet in enumerate(
                gallery_state["selected_triplets"][:10]
            ):
                st.write(f"{i + 1}. {triplet.id} ({triplet.dataset_name})")

            if selected_count > 10:
                st.write(f"... and {selected_count - 10} more")

        # Cache performance
        cache_stats = gallery_state["cache_stats"]
        st.markdown("### **Cache Performance**")

        cache_col1, cache_col2 = st.columns(2)
        with cache_col1:
            st.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
        with cache_col2:
            st.metric("Cached Items", cache_stats.get("size", 0))
