"""
UI utilities for the CrackSeg application.

This module contains utilities for theming, styling, and dialog management
for the user interface.
"""

from .dialogs.save import SaveDialog
from .styling.css import CSSGenerator
from .theme.manager import ColorScheme, ThemeConfig, ThemeManager
from .theme.optimizer import PerformanceOptimizer

__all__ = [
    "ThemeManager",
    "ThemeConfig",
    "ColorScheme",
    "PerformanceOptimizer",
    "CSSGenerator",
    "SaveDialog",
]
