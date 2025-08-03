"""
UI utilities and helpers for the CrackSeg application.

This module contains theming, dialog components, and error handling utilities
for consistent user interface across the application.
"""

from .dialogs.confirmation import ConfirmationDialog
from .dialogs.renderer import ConfirmationRenderer
from .dialogs.utils import ConfirmationUtils
from .error.auto_save import AutoSaveManager
from .error.console import ErrorConsole
from .error.log_viewer import LogViewer
from .theme.header import HeaderComponent
from .theme.logo import LogoComponent
from .theme.main import ThemeComponent

__all__ = [
    "ThemeComponent",
    "LogoComponent",
    "HeaderComponent",
    "ConfirmationDialog",
    "ConfirmationRenderer",
    "ConfirmationUtils",
    "ErrorConsole",
    "AutoSaveManager",
    "LogViewer",
]
