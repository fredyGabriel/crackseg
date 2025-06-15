"""Real-time status updates system for training process monitoring.

This module provides a comprehensive status update system that combines
reactive callbacks for critical events with periodic polling for general
status information. Designed to keep UI components synchronized with
training process state, log streaming, and error conditions.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..streaming.core import LogLevel, StreamedLog


class StatusUpdateType(Enum):
    """Types of status updates that can be broadcast."""

    PROCESS_STARTED = "process_started"
    PROCESS_STOPPED = "process_stopped"
    PROCESS_ERROR = "process_error"
    LOG_RECEIVED = "log_received"
    METRICS_UPDATED = "metrics_updated"
    PROGRESS_UPDATED = "progress_updated"
    SESSION_STATE_CHANGED = "session_state_changed"
    ABORT_INITIATED = "abort_initiated"
    ABORT_COMPLETED = "abort_completed"


@dataclass
class StatusUpdate:
    """Container for status update information."""

    update_type: StatusUpdateType
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.update_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "priority": self.priority,
        }


# Type alias for status update callbacks
StatusUpdateCallback = Callable[[StatusUpdate], None]


class StatusUpdateManager:
    """Manages real-time status updates with hybrid reactive/polling approach.

    Combines immediate callbacks for critical events with periodic polling
    for general status information. Provides thread-safe broadcasting
    to multiple UI components.
    """

    def __init__(
        self,
        polling_interval: float = 2.0,
        max_update_history: int = 1000,
        enable_polling: bool = True,
    ) -> None:
        """Initialize the status update manager.

        Args:
            polling_interval: Seconds between polling updates
            max_update_history: Maximum number of updates to keep in history
            enable_polling: Whether to enable periodic polling
        """
        self.polling_interval = polling_interval
        self.max_update_history = max_update_history
        self.enable_polling = enable_polling

        # Thread-safe storage
        self._callbacks: list[StatusUpdateCallback] = []
        self._update_history: list[StatusUpdate] = []
        self._lock = threading.RLock()

        # Polling control
        self._polling_thread: threading.Thread | None = None
        self._stop_polling = threading.Event()
        self._is_active = False

        # Cached status for efficient polling
        self._last_status_cache: dict[str, Any] = {}
        self._last_poll_time: datetime | None = None

    def start(self) -> None:
        """Start the status update manager and polling thread."""
        with self._lock:
            if self._is_active:
                return

            self._is_active = True
            self._stop_polling.clear()

            if self.enable_polling:
                self._polling_thread = threading.Thread(
                    target=self._polling_loop,
                    daemon=True,
                    name="StatusUpdatePoller",
                )
                self._polling_thread.start()

            self._broadcast_update(
                StatusUpdate(
                    update_type=StatusUpdateType.SESSION_STATE_CHANGED,
                    timestamp=datetime.now(),
                    data={"status": "manager_started"},
                    source="StatusUpdateManager",
                    priority=2,
                )
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the status update manager and polling thread.

        Args:
            timeout: Maximum time to wait for polling thread to stop
        """
        with self._lock:
            if not self._is_active:
                return

            self._is_active = False
            self._stop_polling.set()

            if self._polling_thread and self._polling_thread.is_alive():
                self._polling_thread.join(timeout=timeout)

            self._broadcast_update(
                StatusUpdate(
                    update_type=StatusUpdateType.SESSION_STATE_CHANGED,
                    timestamp=datetime.now(),
                    data={"status": "manager_stopped"},
                    source="StatusUpdateManager",
                    priority=2,
                )
            )

    def add_callback(self, callback: StatusUpdateCallback) -> None:
        """Add a callback to receive status updates.

        Args:
            callback: Function that accepts StatusUpdate instances
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: StatusUpdateCallback) -> bool:
        """Remove a status update callback.

        Args:
            callback: Previously registered callback

        Returns:
            True if callback was found and removed
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def broadcast_process_event(
        self, event_type: StatusUpdateType, data: dict[str, Any] | None = None
    ) -> None:
        """Broadcast a process-related event immediately.

        Args:
            event_type: Type of process event
            data: Additional event data
        """
        update = StatusUpdate(
            update_type=event_type,
            timestamp=datetime.now(),
            data=data or {},
            source="ProcessManager",
            priority=3 if "error" in event_type.value else 2,
        )
        self._broadcast_update(update)

    def broadcast_log_event(self, log: StreamedLog) -> None:
        """Broadcast a log streaming event.

        Args:
            log: Streamed log entry
        """
        # Determine priority based on log level
        priority_map = {
            LogLevel.ERROR: 4,
            LogLevel.WARNING: 3,
            LogLevel.INFO: 1,
            LogLevel.DEBUG: 1,
        }

        update = StatusUpdate(
            update_type=StatusUpdateType.LOG_RECEIVED,
            timestamp=log.timestamp,
            data={
                "level": log.level.value,
                "content": log.content,
                "source_type": log.source,
                "line_number": log.line_number,
            },
            source="LogStreamManager",
            priority=priority_map.get(log.level, 1),
        )
        self._broadcast_update(update)

    def broadcast_metrics_update(self, metrics: dict[str, float]) -> None:
        """Broadcast training metrics update.

        Args:
            metrics: Dictionary of metric names to values
        """
        update = StatusUpdate(
            update_type=StatusUpdateType.METRICS_UPDATED,
            timestamp=datetime.now(),
            data={"metrics": metrics},
            source="MetricsTracker",
            priority=2,
        )
        self._broadcast_update(update)

    def broadcast_progress_update(self, progress_data: dict[str, Any]) -> None:
        """Broadcast training progress update.

        Args:
            progress_data: Progress information (epoch, step, percentage, etc.)
        """
        update = StatusUpdate(
            update_type=StatusUpdateType.PROGRESS_UPDATED,
            timestamp=datetime.now(),
            data=progress_data,
            source="ProgressTracker",
            priority=2,
        )
        self._broadcast_update(update)

    def get_recent_updates(
        self, count: int | None = None, since: datetime | None = None
    ) -> list[StatusUpdate]:
        """Get recent status updates.

        Args:
            count: Maximum number of updates to return
            since: Only return updates after this timestamp

        Returns:
            List of recent status updates
        """
        with self._lock:
            updates = self._update_history.copy()

            if since:
                updates = [u for u in updates if u.timestamp > since]

            if count:
                updates = updates[-count:]

            return updates

    def get_current_status_summary(self) -> dict[str, Any]:
        """Get comprehensive current status summary.

        Returns:
            Dictionary with current system status
        """
        from .orchestrator import get_training_status
        from .session_api import get_session_state_status
        from .streaming_api import get_streaming_status

        try:
            # Get status from all subsystems
            process_status = get_training_status()
            streaming_status = get_streaming_status()
            session_status = get_session_state_status()

            # Get recent critical updates
            recent_errors = [
                u
                for u in self.get_recent_updates(count=10)
                if u.priority >= 3 and "error" in u.update_type.value
            ]

            return {
                "timestamp": datetime.now().isoformat(),
                "manager_active": self._is_active,
                "process": process_status,
                "streaming": streaming_status,
                "session": session_status,
                "recent_errors": [u.to_dict() for u in recent_errors],
                "update_history_size": len(self._update_history),
                "active_callbacks": len(self._callbacks),
                "last_poll_time": (
                    self._last_poll_time.isoformat()
                    if self._last_poll_time
                    else None
                ),
            }

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "manager_active": self._is_active,
                "error": f"Failed to get status: {e}",
                "update_history_size": len(self._update_history),
                "active_callbacks": len(self._callbacks),
            }

    def clear_update_history(self) -> None:
        """Clear the update history."""
        with self._lock:
            self._update_history.clear()

    def broadcast_update(self, update: StatusUpdate) -> None:
        """Public method to broadcast a status update.

        Args:
            update: Status update to broadcast
        """
        self._broadcast_update(update)

    def _broadcast_update(self, update: StatusUpdate) -> None:
        """Broadcast an update to all registered callbacks.

        Args:
            update: Status update to broadcast
        """
        with self._lock:
            # Add to history
            self._update_history.append(update)

            # Trim history if needed
            if len(self._update_history) > self.max_update_history:
                self._update_history = self._update_history[
                    -self.max_update_history :
                ]

            # Broadcast to callbacks (copy list to avoid modification)
            callbacks = self._callbacks.copy()

        # Call callbacks outside of lock to prevent deadlocks
        for callback in callbacks:
            try:
                callback(update)
            except Exception as e:
                # Log callback errors but don't let them break the system
                print(f"Error in status update callback: {e}")

    def _polling_loop(self) -> None:
        """Main polling loop that runs in background thread."""
        while not self._stop_polling.is_set():
            try:
                self._perform_polling_update()
                self._last_poll_time = datetime.now()

            except Exception as e:
                # Log polling errors but continue
                print(f"Error in status polling: {e}")

            # Wait for next poll or stop signal
            self._stop_polling.wait(self.polling_interval)

    def _perform_polling_update(self) -> None:
        """Perform a single polling update cycle."""
        current_status = self.get_current_status_summary()

        # Check for significant changes since last poll
        if self._has_significant_changes(current_status):
            update = StatusUpdate(
                update_type=StatusUpdateType.SESSION_STATE_CHANGED,
                timestamp=datetime.now(),
                data={"status_summary": current_status},
                source="StatusPoller",
                priority=1,
            )
            self._broadcast_update(update)

        # Cache current status for next comparison
        self._last_status_cache = current_status

    def _has_significant_changes(self, current_status: dict[str, Any]) -> bool:
        """Check if current status has significant changes from cached status.

        Args:
            current_status: Current status dictionary

        Returns:
            True if significant changes detected
        """
        if not self._last_status_cache:
            return True

        # Check for process state changes
        old_process = self._last_status_cache.get("process", {})
        new_process = current_status.get("process", {})

        if old_process.get("is_running") != new_process.get("is_running"):
            return True

        if old_process.get("state") != new_process.get("state"):
            return True

        # Check for new errors
        old_errors = len(old_process.get("recent_errors", []))
        new_errors = len(new_process.get("recent_errors", []))

        if new_errors > old_errors:
            return True

        # Check for streaming status changes
        old_streaming = self._last_status_cache.get("streaming", {})
        new_streaming = current_status.get("streaming", {})

        if old_streaming.get("is_streaming") != new_streaming.get(
            "is_streaming"
        ):
            return True

        return False


# Global instance for singleton pattern
_global_status_manager: StatusUpdateManager | None = None


def get_status_update_manager() -> StatusUpdateManager:
    """Get or create the global status update manager instance.

    Returns:
        Global StatusUpdateManager instance
    """
    global _global_status_manager
    if _global_status_manager is None:
        _global_status_manager = StatusUpdateManager()
    return _global_status_manager


def cleanup_status_update_manager() -> None:
    """Clean up and reset the global status update manager."""
    global _global_status_manager
    if _global_status_manager is not None:
        _global_status_manager.stop()
        _global_status_manager = None
