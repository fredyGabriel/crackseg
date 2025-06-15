"""Integration layer for status updates with existing run_manager components.

This module provides integration functions that connect the StatusUpdateManager
with ProcessManager, LogStreamManager, and session state components to ensure
automatic status broadcasting for all relevant events.
"""

from __future__ import annotations

from typing import Any

from ..streaming.core import StreamedLog
from .orchestrator import get_process_manager
from .status_updates import (
    StatusUpdate,
    StatusUpdateManager,
    StatusUpdateType,
    get_status_update_manager,
)


class StatusIntegrationCoordinator:
    """Coordinates status updates between all run_manager components.

    This class sets up automatic status broadcasting by registering
    callbacks with ProcessManager, LogStreamManager, and other components
    to ensure UI components receive real-time updates.
    """

    def __init__(
        self, status_manager: StatusUpdateManager | None = None
    ) -> None:
        """Initialize the integration coordinator.

        Args:
            status_manager: Optional status manager instance
                (uses global if None)
        """
        self.status_manager = status_manager or get_status_update_manager()
        self._is_integrated = False
        self._registered_callbacks: list[tuple[str, Any]] = []

    def integrate_all_components(self) -> bool:
        """Integrate status updates with all run_manager components.

        Returns:
            True if integration successful, False otherwise
        """
        if self._is_integrated:
            return True

        try:
            # Start the status manager
            self.status_manager.start()

            # Integrate with process manager
            self._integrate_process_manager()

            # Integrate with log streaming
            self._integrate_log_streaming()

            # Integrate with session state
            self._integrate_session_state()

            self._is_integrated = True
            return True

        except Exception as e:
            print(f"Failed to integrate status updates: {e}")
            return False

    def cleanup_integration(self) -> None:
        """Clean up all integrations and stop status manager."""
        if not self._is_integrated:
            return

        try:
            # Remove all registered callbacks
            self._cleanup_callbacks()

            # Stop status manager
            self.status_manager.stop()

            self._is_integrated = False

        except Exception as e:
            print(f"Error during status integration cleanup: {e}")

    def _integrate_process_manager(self) -> None:
        """Integrate with ProcessManager for process lifecycle events."""
        process_manager = get_process_manager()

        # Create callback for process state changes (for future use)
        def process_state_callback(old_state: str, new_state: str) -> None:
            """Handle process state changes."""
            event_type_map = {
                "running": StatusUpdateType.PROCESS_STARTED,
                "stopped": StatusUpdateType.PROCESS_STOPPED,
                "error": StatusUpdateType.PROCESS_ERROR,
            }

            event_type = event_type_map.get(
                new_state, StatusUpdateType.SESSION_STATE_CHANGED
            )

            self.status_manager.broadcast_process_event(
                event_type,
                {
                    "old_state": old_state,
                    "new_state": new_state,
                    "process_info": {
                        "pid": process_manager.process_info.pid,
                        "start_time": (
                            str(process_manager.process_info.start_time)
                            if process_manager.process_info.start_time
                            else None
                        ),
                        "return_code": (
                            process_manager.process_info.return_code
                        ),
                        "error_message": (
                            process_manager.process_info.error_message
                        ),
                    },
                },
            )

        del process_state_callback  # Suppress unused function warning

        # Register callback if process manager supports it
        # Note: ProcessManager doesn't currently have add_state_change_callback
        # This would be added in a future enhancement
        # if hasattr(process_manager, "add_state_change_callback"):
        #     process_manager.add_state_change_callback(process_state_callback)
        #     self._registered_callbacks.append(
        #         ("process_state", process_state_callback)
        #     )

    def _integrate_log_streaming(self) -> None:
        """Integrate with LogStreamManager for log events."""
        process_manager = get_process_manager()

        # Create log callback for status updates
        def log_status_callback(log: StreamedLog) -> None:
            """Handle log streaming events for status updates."""
            self.status_manager.broadcast_log_event(log)

        # Register with log stream manager
        if hasattr(process_manager, "stream_manager"):
            process_manager.stream_manager.add_callback(log_status_callback)
            self._registered_callbacks.append(
                ("log_stream", log_status_callback)
            )

    def _integrate_session_state(self) -> None:
        """Integrate with session state management."""
        from .session_api import get_session_state_status

        # Create periodic session state monitor (for future use)
        def check_session_state() -> None:
            """Check for session state changes."""
            try:
                session_status = get_session_state_status()

                # Broadcast session state update
                self.status_manager.broadcast_process_event(
                    StatusUpdateType.SESSION_STATE_CHANGED,
                    {"session_status": session_status},
                )

            except Exception as e:
                print(f"Error checking session state: {e}")

        del check_session_state  # Suppress unused function warning

        # Note: Session state integration would be enhanced with
        # actual session state change callbacks if available

    def _cleanup_callbacks(self) -> None:
        """Remove all registered callbacks."""
        process_manager = get_process_manager()

        for callback_type, callback in self._registered_callbacks:
            try:
                if callback_type == "process_state":
                    # Note: ProcessManager doesn't have remove_state_change_callback  # noqa: E501
                    # This would be implemented in a future enhancement
                    # if hasattr(process_manager, "remove_state_change_callback"):  # noqa: E501
                    #     process_manager.remove_state_change_callback(callback)  # noqa: E501
                    pass

                elif callback_type == "log_stream":
                    if hasattr(process_manager, "stream_manager"):
                        process_manager.stream_manager.remove_callback(
                            callback
                        )

            except Exception as e:
                print(f"Error removing {callback_type} callback: {e}")

        self._registered_callbacks.clear()


# Global integration coordinator
_global_integration_coordinator: StatusIntegrationCoordinator | None = None


def get_status_integration_coordinator() -> StatusIntegrationCoordinator:
    """Get or create the global status integration coordinator.

    Returns:
        Global StatusIntegrationCoordinator instance
    """
    global _global_integration_coordinator
    if _global_integration_coordinator is None:
        _global_integration_coordinator = StatusIntegrationCoordinator()
    return _global_integration_coordinator


def initialize_status_integration() -> bool:
    """Initialize status update integration with all components.

    This is the main function to call when setting up the GUI to enable
    automatic status updates across all components.

    Returns:
        True if initialization successful, False otherwise
    """
    coordinator = get_status_integration_coordinator()
    return coordinator.integrate_all_components()


def cleanup_status_integration() -> None:
    """Clean up status update integration.

    Should be called when shutting down the GUI or when a clean
    reset is needed.
    """
    global _global_integration_coordinator
    if _global_integration_coordinator is not None:
        _global_integration_coordinator.cleanup_integration()
        _global_integration_coordinator = None


def broadcast_manual_status_update(
    update_type: StatusUpdateType,
    data: dict[str, Any] | None = None,
    source: str = "Manual",
    priority: int = 2,
) -> None:
    """Manually broadcast a status update.

    Useful for GUI components that need to broadcast custom status updates.

    Args:
        update_type: Type of status update
        data: Additional data for the update
        source: Source component name
        priority: Update priority (1=low, 2=medium, 3=high, 4=critical)
    """
    status_manager = get_status_update_manager()

    from datetime import datetime

    update = StatusUpdate(
        update_type=update_type,
        timestamp=datetime.now(),
        data=data or {},
        source=source,
        priority=priority,
    )

    status_manager.broadcast_update(update)


def get_comprehensive_status() -> dict[str, Any]:
    """Get comprehensive status from all integrated components.

    Returns:
        Dictionary with complete system status information
    """
    status_manager = get_status_update_manager()
    return status_manager.get_current_status_summary()


def add_status_update_callback(callback: Any) -> None:
    """Add a callback to receive all status updates.

    Args:
        callback: Function that accepts StatusUpdate instances
    """
    status_manager = get_status_update_manager()
    status_manager.add_callback(callback)


def remove_status_update_callback(callback: Any) -> bool:
    """Remove a status update callback.

    Args:
        callback: Previously registered callback

    Returns:
        True if callback was found and removed
    """
    status_manager = get_status_update_manager()
    return status_manager.remove_callback(callback)
