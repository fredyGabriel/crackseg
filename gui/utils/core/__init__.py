"""
Core utilities for the CrackSeg application.

This module contains fundamental utilities including session state management,
configuration handling, and data validation.
"""

from .config.gui import GUIConfig
from .config.io import ConfigIO
from .session.auto_save import AutoSaveManager
from .session.state import SessionState, SessionStateManager
from .session.sync import SessionSyncManager
from .validation.error_state import ErrorMessageFactory, ErrorState

__all__ = [
    "SessionState",
    "SessionStateManager",
    "AutoSaveManager",
    "SessionSyncManager",
    "GUIConfig",
    "ConfigIO",
    "ErrorState",
    "ErrorMessageFactory",
]
