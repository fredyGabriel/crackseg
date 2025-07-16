"""Session state synchronization and management API.

This module provides functions for managing session state synchronization
between training processes, log streaming, and GUI components to maintain
consistent state across the application.
"""

from typing import Any

from ..session_sync import (
    get_session_sync_coordinator,
    initialize_session_sync,
)
from .orchestrator import get_process_manager

# Global state tracking
_session_sync_initialized: bool = False


def initialize_session_state_sync() -> bool:
    """Initialize session state synchronization with process and log managers.

    This function sets up the SessionSyncCoordinator to automatically
    update session state when process lifecycle events or log streaming
    events occur. Should be called when starting the GUI application.

    Returns:
        True if initialization successful, False otherwise
    """
    global _session_sync_initialized

    if _session_sync_initialized:
        return True

    try:
        # Get manager instances
        process_manager = get_process_manager()

        # Get the log stream manager from the process manager
        log_stream_manager = None
        if hasattr(process_manager, "stream_manager"):
            log_stream_manager = process_manager.stream_manager

        if log_stream_manager is None:
            # Create a temporary log stream manager for sync
            from ..streaming import LogStreamManager

            log_stream_manager = LogStreamManager()

        # Initialize session synchronization
        coordinator = initialize_session_sync(
            process_manager, log_stream_manager
        )

        # Verify coordinator is active
        if not coordinator.get_sync_status()["active"]:
            coordinator.start()

        _session_sync_initialized = True
        return True

    except Exception as e:
        print(f"Failed to initialize session state sync: {e}")
        return False


def get_session_state_status() -> dict[str, Any]:
    """Get current session state synchronization status.

    Returns:
        Dictionary with synchronization status information
    """
    if not _session_sync_initialized:
        return {"initialized": False, "error": "Session sync not initialized"}

    try:
        coordinator = get_session_sync_coordinator()
        sync_status = coordinator.get_sync_status()
        sync_status["initialized"] = True
        return sync_status

    except Exception as e:
        return {"initialized": True, "error": str(e)}


def force_session_state_sync() -> bool:
    """Force immediate synchronization of all session state.

    Useful for debugging or when manual refresh is needed.

    Returns:
        True if sync successful, False otherwise
    """
    if not _session_sync_initialized:
        return False

    try:
        coordinator = get_session_sync_coordinator()
        coordinator.force_sync_all()
        return True

    except Exception as e:
        print(f"Failed to force session state sync: {e}")
        return False
