"""
UI Renderer for the Results Gallery component.

This module contains the GalleryRenderer class, which is responsible for
drawing all the Streamlit UI elements for the gallery. It is designed
to be stateless, receiving all necessary data and handlers from the
main component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from pathlib import Path

    import gui.components.gallery.state_manager as sm
    from gui.components.gallery.actions import GalleryActions


class GalleryRenderer:
    """Renders the UI for the Results Gallery."""

    def __init__(
        self, state: sm.GalleryStateManager, actions: GalleryActions
    ) -> None:
        """Initialize the renderer.

        Args:
            state: The state manager instance.
            actions: The actions handler instance.
        """
        self.state = state
        self.actions = actions

    def render_all(self, scan_directory: str | Path | None) -> None:
        """Render the complete results gallery interface."""
        key_prefix = self.state.key_prefix
        self._render_header(scan_directory, key_prefix)

        if scan_directory:
            self._render_scanning_controls(scan_directory, key_prefix)

        self._render_validation_panel(key_prefix)
        self._render_filter_panel(key_prefix)
        self._render_gallery_grid(key_prefix)
        self._render_export_panel(key_prefix)

    def _render_header(
        self, scan_directory: str | Path | None, key: str
    ) -> None:
        """Render the gallery header section."""
        st.header("📊 Prediction Results Gallery")
        # ... implementation from original file ...

    def _render_scanning_controls(
        self, scan_directory: str | Path, key: str
    ) -> None:
        """Render scanning controls and progress."""
        # ... implementation from original file, calling self.actions ...

    def _render_validation_panel(self, key: str) -> None:
        """Render validation statistics panel."""
        # ... implementation from original file ...

    def _render_filter_panel(self, key: str) -> None:
        """Render filtering and advanced selection panel."""
        # ... implementation from original file ...

    def _render_gallery_grid(self, key: str) -> None:
        """Render the main gallery grid."""
        # ... implementation from original file ...

    def _render_export_panel(self, key: str) -> None:
        """Render the export panel."""
        # ... implementation from original file ...
