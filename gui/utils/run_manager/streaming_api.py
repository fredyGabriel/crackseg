"""Log streaming API functions for real-time training monitoring.

This module provides high-level functions for managing log streaming,
callbacks, and buffer operations, acting as the interface between
GUI components and the underlying streaming system.
"""

from typing import Any

from ..streaming import StreamedLog
from ..streaming.core import LogCallback
from .orchestrator import get_process_manager


def add_log_callback(callback: LogCallback) -> None:
    """Add a callback function to receive real-time log entries.

    The callback will be called for each log line streamed from the
    training process (both stdout and Hydra log files).

    Args:
        callback: Function that accepts StreamedLog instances

    Example:
        >>> def log_handler(log: StreamedLog) -> None:
        ...     print(f"[{log.level.value}] {log.content}")
        >>> add_log_callback(log_handler)
    """
    manager = get_process_manager()
    manager.stream_manager.add_callback(callback)


def remove_log_callback(callback: LogCallback) -> bool:
    """Remove a previously added log callback.

    Args:
        callback: Callback function to remove

    Returns:
        True if callback was found and removed, False otherwise
    """
    manager = get_process_manager()
    return manager.stream_manager.remove_callback(callback)


def get_recent_logs(count: int | None = None) -> list[StreamedLog]:
    """Get recent log entries from the streaming buffer.

    Args:
        count: Number of recent logs to return (None for all)

    Returns:
        List of StreamedLog instances ordered chronologically

    Example:
        >>> recent_logs = get_recent_logs(50)  # Last 50 log entries
        >>> for log in recent_logs:
        ...     print(f"{log.timestamp}: {log.content}")
    """
    manager = get_process_manager()
    return manager.stream_manager.get_recent_logs(count)


def clear_log_buffer() -> None:
    """Clear the streaming log buffer.

    Useful for starting fresh when beginning a new training session.
    """
    manager = get_process_manager()
    manager.stream_manager.clear_buffer()


def get_streaming_status() -> dict[str, Any]:
    """Get comprehensive status of the log streaming system.

    Returns:
        Dictionary with streaming status, buffer info, and source details

    Example:
        >>> status = get_streaming_status()
        >>> print(f"Streaming: {status['is_streaming']}")
        >>> print(f"Logs processed: {status['total_logs_processed']}")
    """
    manager = get_process_manager()
    base_status = manager.stream_manager.get_status()

    # Add process-specific streaming info
    streaming_info = {
        "stdout_reader_active": manager.get_stdout_reader_status(),
        "hydra_watcher_active": manager.get_hydra_watcher_status(),
        "tracked_log_files": manager.get_tracked_log_files(),
    }

    return {**base_status, **streaming_info}
