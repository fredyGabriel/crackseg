"""
Configuration editor component for YAML editing with live validation.

This component provides a sophisticated YAML editor using Ace Editor
with syntax highlighting, live validation, and save functionality.
"""

import logging

import streamlit as st

from scripts.gui.components.config_editor import (
    ConfigEditorCore,
    FileBrowserIntegration,
    ValidationPanel,
)

logger = logging.getLogger(__name__)


class ConfigEditorComponent:
    """Advanced YAML configuration editor with live validation."""

    def __init__(self) -> None:
        """Initialize the configuration editor component."""
        self.editor_core = ConfigEditorCore()
        self.validation_panel = ValidationPanel()
        self.file_browser = FileBrowserIntegration()

    def render_editor(
        self,
        initial_content: str = "",
        key: str = "config_editor",
        height: int = 400,
    ) -> str:
        """Render the Ace editor with YAML configuration.

        Args:
            initial_content: Initial YAML content for the editor
            key: Unique key for the editor component
            height: Editor height in pixels

        Returns:
            Current content of the editor
        """
        return self.editor_core.render_editor(initial_content, key, height)

    def render_editor_with_advanced_validation(
        self,
        initial_content: str = "",
        key: str = "config_editor",
        height: int = 400,
    ) -> str:
        """Render editor with advanced validation panel.

        Args:
            initial_content: Initial YAML content for the editor
            key: Unique key for the editor component
            height: Editor height in pixels

        Returns:
            Current content of the editor
        """
        col_editor, col_validation = st.columns([2, 1])

        with col_editor:
            content = self.editor_core.render_editor(
                initial_content, key, height
            )

        with col_validation:
            st.subheader("✅ Validación en Vivo")
            self.validation_panel.render_advanced_validation(content, key)

        return content

    def render_file_browser_integration(
        self, key: str = "config_browser"
    ) -> None:
        """Render file browser integration for configuration files.

        Args:
            key: Unique key for the file browser component
        """
        self.file_browser.render_file_browser(key)

    def render_advanced_load_dialog(self, key: str = "config_editor") -> None:
        """Render advanced file loading dialog.

        Args:
            key: Base key for the editor component
        """
        self.file_browser.render_advanced_load_dialog(key)
