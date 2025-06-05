"""
Components package for the CrackSeg GUI application.

This package contains reusable UI components including logo,
sidebar, file browser, configuration editor, and routing functionality.
"""

from scripts.gui.components.config_editor_component import (
    ConfigEditorComponent,
)
from scripts.gui.components.file_browser_component import (
    FileBrowserComponent,
    render_file_browser,
)
from scripts.gui.components.logo_component import LogoComponent
from scripts.gui.components.page_router import PageRouter
from scripts.gui.components.sidebar_component import render_sidebar
from scripts.gui.components.theme_component import ThemeComponent

__all__ = [
    "ConfigEditorComponent",
    "FileBrowserComponent",
    "LogoComponent",
    "PageRouter",
    "render_file_browser",
    "render_sidebar",
    "ThemeComponent",
]
