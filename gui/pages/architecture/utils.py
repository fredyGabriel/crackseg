"""
Utility functions for the architecture page module.

This module contains common utilities and helper functions used across
the architecture page components.
"""

import logging

from scripts.gui.utils.architecture_viewer import get_architecture_viewer
from scripts.gui.utils.session_state import SessionStateManager

logger = logging.getLogger(__name__)


def clear_model_state() -> None:
    """Clear all model-related state."""
    state = SessionStateManager.get()

    # Clear model state
    state.model_loaded = False
    state.current_model = None
    state.model_summary = {}
    state.model_device = None
    state.model_architecture = None

    # Clear diagram state
    state.architecture_diagram_path = None

    # Cleanup temporary files
    viewer = get_architecture_viewer()
    viewer.cleanup_temp_files()

    SessionStateManager.notify_change("model_cleared")
    logger.info("Model state cleared")
