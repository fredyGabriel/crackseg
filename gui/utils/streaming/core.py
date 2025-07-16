"""Core log streaming components for real-time training monitoring.

This module provides the main StreamedLog data structure and LogStreamManager
for coordinating multiple log sources with thread-safe operations.
"""

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .exceptions import LogStreamingError


class LogLevel(Enum):
    """Log level enumeration for consistent categorization."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class StreamedLog:
    """Immutable log entry from streaming process.

    Represents a single log line captured from crackseg.training process
    with metadata for GUI display and filtering.

    Attributes:
        timestamp: When the log was captured
        level: Log level (INFO, ERROR, etc.)
        source: Source identifier (stdout, file, etc.)
        content: Raw log line content
        line_number: Line number in source (if applicable)

    Example:
        >>> log = StreamedLog(
        ...     timestamp=datetime.now(),
        ...     level=LogLevel.INFO,
        ...     source="stdout",
        ...     content="Epoch 1/100 - Loss: 0.45",
        ...     line_number=142
        ... )
    """

    timestamp: datetime
    level: LogLevel
    source: str
    content: str
    line_number: int | None = None


type LogCallback = Callable[[StreamedLog], None]


class LogStreamManager:
    """Thread-safe manager for coordinating multiple log streams.

    Handles real-time log streaming from multiple sources (stdout, files)
    with buffering, filtering, and callback support for GUI integration.

    Features:
    - Thread-safe operations with proper locking
    - Configurable buffer size to prevent memory issues
    - Multiple log source coordination
    - Callback system for real-time GUI updates
    - Graceful shutdown and resource cleanup

    Example:
        >>> manager = LogStreamManager()
        >>> manager.add_callback(lambda log: print(f"[{log.level.value}]"))
        >>> manager.start_streaming()
        >>> # Add log sources...
        >>> manager.stop_streaming()
    """

    def __init__(self, max_buffer_size: int = 1000) -> None:
        """Initialize the log stream manager.

        Args:
            max_buffer_size: Maximum number of logs to keep in memory
        """
        self._max_buffer_size = max_buffer_size
        self._buffer: list[StreamedLog] = []
        self._callbacks: list[LogCallback] = []
        self._sources: dict[str, Any] = {}

        # Threading controls
        self._log_queue: queue.Queue[StreamedLog | None] = queue.Queue()
        self._processor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Stream state
        self._is_streaming = False
        self._total_logs_processed = 0

    @property
    def is_streaming(self) -> bool:
        """Check if manager is currently streaming logs."""
        with self._lock:
            return self._is_streaming

    @property
    def total_logs_processed(self) -> int:
        """Get total number of logs processed since start."""
        with self._lock:
            return self._total_logs_processed

    @property
    def buffer_size(self) -> int:
        """Get current number of logs in buffer."""
        with self._lock:
            return len(self._buffer)

    def add_callback(self, callback: LogCallback) -> None:
        """Add a callback function for new log entries.

        Args:
            callback: Function to call with new StreamedLog instances

        Example:
            >>> def log_handler(log: StreamedLog) -> None:
            ...     print(f"{log.timestamp}: {log.content}")
            >>> manager.add_callback(log_handler)
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: LogCallback) -> bool:
        """Remove a callback function.

        Args:
            callback: Callback function to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def start_streaming(self) -> None:
        """Start the log streaming and processing system.

        Raises:
            LogStreamingError: If streaming is already active
        """
        with self._lock:
            if self._is_streaming:
                raise LogStreamingError("Log streaming is already active")

            self._is_streaming = True
            self._stop_event.clear()

        # Start log processor thread
        self._processor_thread = threading.Thread(
            target=self._process_logs, name="LogStreamProcessor", daemon=True
        )
        self._processor_thread.start()

    def stop_streaming(self, timeout: float = 5.0) -> None:
        """Stop log streaming and clean up resources.

        Args:
            timeout: Maximum time to wait for thread shutdown
        """
        with self._lock:
            if not self._is_streaming:
                return

            self._is_streaming = False

        # Signal stop and wait for processor thread
        self._stop_event.set()
        self._log_queue.put(None)  # Sentinel to wake up processor

        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=timeout)

        # Stop all log sources
        for source in self._sources.values():
            if hasattr(source, "stop"):
                try:
                    source.stop()
                except Exception:
                    pass  # Best effort cleanup

    def add_log(self, log: StreamedLog) -> None:
        """Add a log entry to the stream (thread-safe).

        Args:
            log: StreamedLog instance to add
        """
        if not self.is_streaming:
            return

        self._log_queue.put(log)

    def get_recent_logs(self, count: int | None = None) -> list[StreamedLog]:
        """Get recent log entries from buffer.

        Args:
            count: Number of recent logs to return (None for all)

        Returns:
            List of recent StreamedLog instances
        """
        with self._lock:
            if count is None:
                return self._buffer.copy()
            return self._buffer[-count:] if self._buffer else []

    def clear_buffer(self) -> None:
        """Clear the log buffer (thread-safe)."""
        with self._lock:
            self._buffer.clear()

    def register_source(self, name: str, source: Any) -> None:
        """Register a log source for management.

        Args:
            name: Unique identifier for the source
            source: Log source object (should have start/stop methods)
        """
        with self._lock:
            self._sources[name] = source

    def unregister_source(self, name: str) -> bool:
        """Unregister a log source.

        Args:
            name: Source identifier to remove

        Returns:
            True if source was found and removed
        """
        with self._lock:
            return self._sources.pop(name, None) is not None

    def _process_logs(self) -> None:
        """Main log processing loop (runs in separate thread)."""
        while not self._stop_event.is_set():
            try:
                # Get log from queue with timeout
                log = self._log_queue.get(timeout=1.0)

                if log is None:  # Sentinel value for shutdown
                    break

                # Process the log entry
                self._handle_log_entry(log)

            except queue.Empty:
                continue
            except Exception as e:
                # Log processing should be robust
                print(f"Error processing log: {e}")
                continue

    def _handle_log_entry(self, log: StreamedLog) -> None:
        """Handle a single log entry (add to buffer and notify callbacks)."""
        with self._lock:
            # Add to buffer with size management
            self._buffer.append(log)
            if len(self._buffer) > self._max_buffer_size:
                self._buffer.pop(0)  # Remove oldest

            self._total_logs_processed += 1

            # Get callbacks copy for safe iteration
            callbacks = self._callbacks.copy()

        # Execute callbacks outside of lock
        for callback in callbacks:
            try:
                callback(log)
            except Exception:
                # Callbacks should not break the streaming system
                pass

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive streaming status information.

        Returns:
            Dictionary with streaming status details
        """
        with self._lock:
            return {
                "is_streaming": self._is_streaming,
                "buffer_size": len(self._buffer),
                "max_buffer_size": self._max_buffer_size,
                "total_logs_processed": self._total_logs_processed,
                "active_callbacks": len(self._callbacks),
                "registered_sources": list(self._sources.keys()),
                "processor_thread_alive": (
                    self._processor_thread.is_alive()
                    if self._processor_thread
                    else False
                ),
            }

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        if hasattr(self, "_is_streaming") and self._is_streaming:
            self.stop_streaming(timeout=1.0)
