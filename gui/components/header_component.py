"""
Header Component for the CrackSeg application.

This module provides a reusable header component to be displayed
at the top of each page.
"""

from pathlib import Path
from typing import Literal

import streamlit as st


def render_header(
    page_title: str, anchor: str | Literal[False] | None = False
) -> None:
    """
    Renders a consistent page header with a logo and title.

    This component is designed to be the first visual element on each page,
    providing a consistent branding and navigation anchor.

    Args:
        page_title: The title to display for the current page.
        anchor: The anchor for the title link. Set to False to disable.
    """
    logo_path = Path("docs/designs/logo.png")

    columns = st.columns([1, 4], gap="medium")
    col1, col2 = columns[0], columns[1]

    with col1:
        if logo_path.is_file():
            st.image(str(logo_path), width=120)
        else:
            st.warning("Logo not found.")

    with col2:
        st.title(page_title, anchor=anchor)

    st.markdown("---")
