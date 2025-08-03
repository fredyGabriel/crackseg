"""
TensorBoard iframe rendering for Streamlit. This module handles the
embedding of TensorBoard URLs in Streamlit using secure iframe
components with proper error handling.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def render_tensorboard_iframe(
    url: str | None,
    log_dir: Path,
    height: int,
    width: int | None,
) -> bool:
    """
    Render TensorBoard iframe in Streamlit. Args: url: TensorBoard URL to
    embed. log_dir: Log directory being displayed. height: Iframe height
    in pixels. width: Iframe width in pixels (None for responsive).
    Returns: True if iframe was rendered successfully, False otherwise.
    """
    if not url:
        st.error("🔴 TensorBoard is running but URL is not available")
        if st.button("🔄 Refresh URL"):
            st.rerun()
        return False

    # Show loading state
    st.markdown("---")

    try:
        # Use Streamlit's iframe component for secure embedding
        with st.spinner("Loading TensorBoard interface..."):
            components.iframe(
                src=url,
                height=height,
                width=width,
                scrolling=True,
            )

        # Add helpful information
        _render_iframe_footer(url, log_dir)
        return True

    except Exception as e:
        st.error(f"❌ Failed to embed TensorBoard iframe: {e}")
        _render_iframe_fallback(url)
        return False


def _render_iframe_footer(url: str, log_dir: Path) -> None:
    """Render footer information below the iframe."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.caption(f"📊 Embedded TensorBoard from: `{log_dir}`")
        st.caption(f"🔗 Direct access: [{url}]({url})")

    with col2:
        if st.button(
            "🔗 Open in New Tab",
            help="Open TensorBoard in new browser tab",
        ):
            st.markdown(
                f'<a href="{url}" target="_blank">🔗 Open TensorBoard</a>',
                unsafe_allow_html=True,
            )


def _render_iframe_fallback(url: str) -> None:
    """Render fallback options when iframe fails."""
    st.info(f"💡 You can access TensorBoard directly at: [{url}]({url})")

    # Offer troubleshooting options
    with st.expander("🔧 Troubleshooting Options", expanded=False):
        st.markdown(
            f"""
**If the iframe doesn't load:** 1. Click the direct link above to open
in a new tab 2. Check if your browser blocks iframes 3. Try refreshing
this page 4. Restart TensorBoard using the controls above **Direct
URL:** `{url}`
"""
        )
