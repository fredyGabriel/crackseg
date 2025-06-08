"""Session state synchronization coordinator for CrackSeg GUI.

This module provides the coordination layer between ProcessManager,
LogStreamManager, and SessionState to ensure real-time synchronization
of subprocess lifecycle events, log streaming status, and training metrics.

Implements the Event-Driven Synchronization pattern with callbacks
to provide real-time updates to the Streamlit session state.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from .session_state import SessionStateManager
from .streaming import StreamedLog

logger = logging.getLogger(__name__)


class SessionSyncCoordinator:
    """Coordinates session state updates across multiple components.

    This class acts as the central hub for synchronizing session state
    with subprocess lifecycle events, log streaming updates, and
    training progress. Implements thread-safe event handling with
    callback management.

    Features:
    - Real-time process lifecycle synchronization
    - Log streaming state management
    - Training metrics extraction from logs
    - Thread-safe callback management
    - Automatic cleanup and resource management

    Example:
        >>> coordinator = SessionSyncCoordinator()
        >>> coordinator.register_with_process_manager(process_manager)
        >>> coordinator.register_with_log_stream_manager(log_manager)
    """

    def __init__(self) -> None:
        """Initialize the session sync coordinator."""
        self._active = False
        self._lock = threading.Lock()
        self._callbacks: dict[str, list[Callable[..., None]]] = {
            "process_update": [],
            "log_update": [],
            "metrics_update": [],
        }

        # Tracking state for efficient updates
        self._last_process_update = 0.0
        self._last_log_update = 0.0
        self._update_frequency = 1.0  # Update frequency in seconds

        logger.info("SessionSyncCoordinator initialized")

    def start(self) -> None:
        """Start the session synchronization coordinator."""
        with self._lock:
            if self._active:
                logger.warning("SessionSyncCoordinator already active")
                return

            self._active = True
            logger.info("SessionSyncCoordinator started")

    def stop(self) -> None:
        """Stop the session synchronization coordinator."""
        with self._lock:
            if not self._active:
                return

            self._active = False
            logger.info("SessionSyncCoordinator stopped")

    def register_with_process_manager(self, process_manager: Any) -> None:
        """Register callbacks with ProcessManager for lifecycle events.

        Args:
            process_manager: ProcessManager instance to register with
        """
        try:
            # Create a callback wrapper for process info updates
            def process_callback() -> None:
                if not self._active:
                    return

                current_time = time.time()
                if (
                    current_time - self._last_process_update
                    < self._update_frequency
                ):
                    return

                self._last_process_update = current_time

                try:
                    # Get current process info
                    process_info = process_manager.process_info

                    # Update session state from process info
                    SessionStateManager.update_from_process_info(process_info)

                    # Update memory usage if available
                    if hasattr(process_manager, "get_memory_usage"):
                        memory_usage = process_manager.get_memory_usage()
                        if memory_usage:
                            state = SessionStateManager.get()
                            state.update_process_state(
                                state=state.process_state,  # Keep current
                                memory_usage=memory_usage,
                            )

                    # Trigger custom callbacks
                    self._trigger_callbacks("process_update", process_info)

                    logger.debug(
                        f"Process state updated: {process_info.state}"
                    )

                except Exception as e:
                    logger.error(f"Error in process callback: {e}")

            # Register the callback with the process manager's monitoring
            # thread. This depends on the ProcessManager implementation
            if hasattr(process_manager, "add_state_callback"):
                process_manager.add_state_callback(process_callback)
            else:
                logger.warning(
                    "ProcessManager doesn't support state callbacks, "
                    "using polling fallback"
                )

            logger.info("Registered with ProcessManager")

        except Exception as e:
            logger.error(f"Failed to register with ProcessManager: {e}")

    def register_with_log_stream_manager(
        self, log_stream_manager: Any
    ) -> None:
        """Register callbacks with LogStreamManager for log updates.

        Args:
            log_stream_manager: LogStreamManager instance to register with
        """
        try:
            # Create a callback wrapper for log updates
            def log_callback(log_entry: StreamedLog) -> None:
                if not self._active:
                    return

                current_time = time.time()
                if (
                    current_time - self._last_log_update < 0.5
                ):  # More frequent for logs
                    return

                self._last_log_update = current_time

                try:
                    # Get recent logs for session state
                    recent_logs = []
                    if hasattr(log_stream_manager, "get_recent_logs"):
                        recent_logs_raw = log_stream_manager.get_recent_logs(
                            50
                        )
                        recent_logs = [
                            {
                                "message": getattr(log, "message", ""),
                                "level": self._extract_log_level(log),
                                "timestamp": getattr(
                                    log, "timestamp", time.time()
                                ),
                                "source": getattr(log, "source", "unknown"),
                            }
                            for log in recent_logs_raw
                        ]

                    # Get buffer size
                    buffer_size = 0
                    if hasattr(log_stream_manager, "buffer_size"):
                        buffer_size = log_stream_manager.buffer_size

                    # Get Hydra run directory
                    hydra_run_dir = None
                    if (
                        hasattr(log_stream_manager, "hydra_watcher")
                        and log_stream_manager.hydra_watcher
                    ):
                        if hasattr(
                            log_stream_manager.hydra_watcher, "run_dir"
                        ):
                            hydra_run_dir = str(
                                log_stream_manager.hydra_watcher.run_dir
                            )

                    # Update session state with log streaming info
                    SessionStateManager.update_from_log_stream_info(
                        active=getattr(log_stream_manager, "is_active", False),
                        buffer_size=buffer_size,
                        recent_logs=recent_logs,
                        hydra_run_dir=hydra_run_dir,
                    )

                    # Extract training statistics from logs
                    if recent_logs:
                        SessionStateManager.extract_training_stats_from_logs(
                            recent_logs
                        )

                    # Trigger custom callbacks
                    self._trigger_callbacks("log_update", log_entry)

                    logger.debug("Log state updated")

                except Exception as e:
                    logger.error(f"Error in log callback: {e}")

            # Register with log stream manager
            if hasattr(log_stream_manager, "add_callback"):
                log_stream_manager.add_callback(log_callback)
                logger.info("Registered with LogStreamManager")
            else:
                logger.warning("LogStreamManager doesn't support callbacks")

        except Exception as e:
            logger.error(f"Failed to register with LogStreamManager: {e}")

    def register_callback(
        self, event_type: str, callback: Callable[..., None]
    ) -> None:
        """Register a custom callback for specific events.

        Args:
            event_type: Type of event ('process_update', 'log_update',
                'metrics_update')
            callback: Callback function to register
        """
        with self._lock:
            if event_type in self._callbacks:
                self._callbacks[event_type].append(callback)
                logger.debug(f"Registered callback for {event_type}")
            else:
                logger.warning(f"Unknown event type: {event_type}")

    def unregister_callback(
        self, event_type: str, callback: Callable[..., None]
    ) -> bool:
        """Unregister a custom callback.

        Args:
            event_type: Type of event
            callback: Callback function to unregister

        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._lock:
            if event_type in self._callbacks:
                try:
                    self._callbacks[event_type].remove(callback)
                    logger.debug(f"Unregistered callback for {event_type}")
                    return True
                except ValueError:
                    logger.warning(f"Callback not found for {event_type}")
                    return False
            else:
                logger.warning(f"Unknown event type: {event_type}")
                return False

    def _trigger_callbacks(self, event_type: str, *args: Any) -> None:
        """Trigger all callbacks for a specific event type.

        Args:
            event_type: Type of event to trigger
            *args: Arguments to pass to callbacks
        """
        with self._lock:
            callbacks = self._callbacks.get(event_type, [])

        for callback in callbacks:
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")

    def force_sync_all(self) -> None:
        """Force synchronization of all state immediately.

        This method can be called to ensure session state is
        up-to-date with all components, useful for debugging
        or manual refresh scenarios.
        """
        try:
            # Reset update timers to force updates
            self._last_process_update = 0.0
            self._last_log_update = 0.0

            # Trigger a state notification
            SessionStateManager.notify_change("manual_sync")

            logger.info("Forced synchronization of all state")

        except Exception as e:
            logger.error(f"Error in force_sync_all: {e}")

    def get_sync_status(self) -> dict[str, Any]:
        """Get current synchronization status information.

        Returns:
            Dictionary with sync status details
        """
        with self._lock:
            return {
                "active": self._active,
                "last_process_update": self._last_process_update,
                "last_log_update": self._last_log_update,
                "update_frequency": self._update_frequency,
                "callback_counts": {
                    event_type: len(callbacks)
                    for event_type, callbacks in self._callbacks.items()
                },
            }

    def set_update_frequency(self, frequency: float) -> None:
        """Set the minimum update frequency in seconds.

        Args:
            frequency: Minimum time between updates in seconds
        """
        with self._lock:
            self._update_frequency = max(0.1, min(10.0, frequency))
            logger.info(f"Update frequency set to {self._update_frequency}s")

    def __del__(self) -> None:
        """Cleanup when coordinator is destroyed."""
        try:
            self.stop()
        except Exception:
            pass

    def _extract_log_level(self, log: StreamedLog) -> str:
        """Extract log level from a StreamedLog object."""
        level = getattr(log, "level", None)
        if level and hasattr(level, "value"):
            return level.value
        elif level:
            return str(level)
        else:
            return "unknown"


# Global coordinator instance for singleton pattern
_global_coordinator: SessionSyncCoordinator | None = None


def get_session_sync_coordinator() -> SessionSyncCoordinator:
    """Get or create the global session sync coordinator.

    Returns:
        Global SessionSyncCoordinator instance
    """
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = SessionSyncCoordinator()
    return _global_coordinator


def cleanup_session_sync_coordinator() -> None:
    """Clean up and reset the global session sync coordinator."""
    global _global_coordinator
    if _global_coordinator is not None:
        _global_coordinator.stop()
        _global_coordinator = None


def initialize_session_sync(
    process_manager: Any, log_stream_manager: Any
) -> SessionSyncCoordinator:
    """Initialize session synchronization with managers.

    Args:
        process_manager: ProcessManager instance
        log_stream_manager: LogStreamManager instance

    Returns:
        Configured SessionSyncCoordinator
    """
    coordinator = get_session_sync_coordinator()

    # Register with managers
    coordinator.register_with_process_manager(process_manager)
    coordinator.register_with_log_stream_manager(log_stream_manager)

    # Start coordination
    coordinator.start()

    logger.info("Session synchronization initialized")
    return coordinator
