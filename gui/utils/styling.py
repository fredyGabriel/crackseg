"""
GUI Styling Utilities This module provides functions for loading and
applying custom CSS styles to the Streamlit application.
"""

from pathlib import Path

import streamlit as st


def load_custom_css() -> None:
    """
    Finds and injects the custom CSS stylesheet into the Streamlit app. It
    constructs a path to 'main.css' relative to this file's location.
    """
    # Path to the CSS file, assuming it's in ../styles/main.css from this file
    css_path = Path(__file__).parent.parent / "styles" / "main.css"

    if not css_path.is_file():
        st.warning(
            f"Custom CSS file not found at {css_path}. Please ensure it "
            "exists."
        )
        return

    with open(css_path) as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def apply_custom_css(css_content: str) -> None:
    """
    Apply custom CSS content directly to the Streamlit app. Args:
    css_content: CSS content to apply
    """
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
