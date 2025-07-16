"""TensorBoard core types, states, and exceptions.

This module defines the fundamental data structures and exceptions used
throughout the TensorBoard management system. Provides type safety and
consistent error handling across all TensorBoard components.

Key Components:
- TensorBoardState: Process execution state enumeration
- TensorBoardError: Custom exception for TensorBoard failures
- TensorBoardInfo: Comprehensive process information container
- Utility functions for state validation and error handling

Example:
    >>> info = TensorBoardInfo(
    ...     pid=1234,
    ...     port=6006,
    ...     state=TensorBoardState.RUNNING
    ... )
    >>> if info.is_healthy():
    ...     print(f"TensorBoard running on port {info.port}")
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .port_management import PortAllocation


class TensorBoardState(Enum):
    """TensorBoard process execution states.

    Represents the lifecycle states of a TensorBoard process from
    initial idle state through startup, running, and shutdown phases.
    Includes error states for proper failure handling.

    Values:
        IDLE: Process not started or completely stopped
        STARTING: Process is in startup phase
        RUNNING: Process is active and serving
        STOPPING: Process is shutting down
        STOPPED: Process has stopped (may restart)
        FAILED: Process failed to start or crashed
    """

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

    def is_active(self) -> bool:
        """Check if state represents an active process."""
        return self in (self.STARTING, self.RUNNING)

    def is_transitioning(self) -> bool:
        """Check if state represents a transitional phase."""
        return self in (self.STARTING, self.STOPPING)

    def can_start(self) -> bool:
        """Check if process can be started from this state."""
        return self in (self.IDLE, self.STOPPED, self.FAILED)

    def can_stop(self) -> bool:
        """Check if process can be stopped from this state."""
        return self in (self.STARTING, self.RUNNING)

    def can_restart(self) -> bool:
        """Check if process can be restarted from this state."""
        return self != self.STARTING


class TensorBoardError(Exception):
    """Custom exception for TensorBoard process errors.

    Raised when TensorBoard subprocess management fails due to:
    - Port allocation conflicts
    - Log directory not found
    - Process startup failures
    - Network interface issues
    - Lifecycle management errors

    Attributes:
        message: Error description
        error_code: Optional error code for categorization
        port: Port number involved in error (if applicable)
        log_dir: Log directory path involved in error (if applicable)

    Examples:
        >>> raise TensorBoardError("Port 6006 is already in use", port=6006)
        >>> raise TensorBoardError(
        ...     "Log directory not found", log_dir=Path("/logs")
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        port: int | None = None,
        log_dir: Path | None = None,
    ) -> None:
        """Initialize TensorBoard error with context information."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.port = port
        self.log_dir = log_dir

    def __str__(self) -> str:
        """Return formatted error message with context."""
        parts = [self.message]

        if self.port:
            parts.append(f"(port: {self.port})")
        if self.log_dir:
            parts.append(f"(log_dir: {self.log_dir})")
        if self.error_code:
            parts.append(f"[{self.error_code}]")

        return " ".join(parts)


class PortConflictError(TensorBoardError):
    """Specific error for port allocation conflicts."""

    def __init__(self, port: int, message: str | None = None) -> None:
        """Initialize port conflict error."""
        msg = message or f"Port {port} is already in use or allocated"
        super().__init__(msg, error_code="PORT_CONFLICT", port=port)


class LogDirectoryError(TensorBoardError):
    """Specific error for log directory issues."""

    def __init__(self, log_dir: Path, message: str | None = None) -> None:
        """Initialize log directory error."""
        msg = message or f"Log directory not accessible: {log_dir}"
        super().__init__(msg, error_code="LOG_DIR_ERROR", log_dir=log_dir)


class ProcessStartupError(TensorBoardError):
    """Specific error for process startup failures."""

    def __init__(self, message: str, port: int | None = None) -> None:
        """Initialize process startup error."""
        super().__init__(message, error_code="STARTUP_ERROR", port=port)


@dataclass
class TensorBoardInfo:
    """Comprehensive information about a TensorBoard process.

    Contains all metadata and state information for a running or configured
    TensorBoard instance. Used for monitoring, health checks, and lifecycle
    management.

    Attributes:
        pid: Process ID of the TensorBoard subprocess
        port: Port number TensorBoard is bound to
        log_dir: Directory containing TensorBoard logs
        url: Full URL for accessing TensorBoard interface
        start_time: Timestamp when process was started
        state: Current process state
        error_message: Last error message if any
        startup_attempts: Number of startup attempts made
        last_health_check: Timestamp of last health check
        health_status: Result of last health check
        port_allocation: Port allocation information
    """

    pid: int | None = None
    port: int | None = None
    log_dir: Path | None = None
    url: str | None = None
    start_time: float | None = None
    state: TensorBoardState = TensorBoardState.IDLE
    error_message: str | None = None
    startup_attempts: int = 0
    last_health_check: float | None = None
    health_status: bool = False
    port_allocation: "PortAllocation | None" = None

    def is_running(self) -> bool:
        """Check if TensorBoard is currently running."""
        return self.state == TensorBoardState.RUNNING

    def is_healthy(self) -> bool:
        """Check if TensorBoard is running and healthy."""
        return self.is_running() and self.health_status

    def has_error(self) -> bool:
        """Check if there's an active error."""
        return (
            self.error_message is not None
            or self.state == TensorBoardState.FAILED
        )

    def get_uptime(self) -> float | None:
        """Get uptime in seconds since start."""
        if self.start_time and self.is_running():
            return time.time() - self.start_time
        return None

    def get_health_age(self) -> float | None:
        """Get age of last health check in seconds."""
        if self.last_health_check:
            return time.time() - self.last_health_check
        return None

    def is_health_check_stale(self, max_age: float = 30.0) -> bool:
        """Check if health check is stale."""
        health_age = self.get_health_age()
        return health_age is None or health_age > max_age

    def update_state(
        self,
        state: TensorBoardState,
        error_message: str | None = None,
        clear_error: bool = True,
    ) -> None:
        """Update the state and optionally clear/set error message."""
        self.state = state

        if clear_error and state != TensorBoardState.FAILED:
            self.error_message = None
        elif error_message:
            self.error_message = error_message

    def update_health(self, healthy: bool) -> None:
        """Update health status and timestamp."""
        self.health_status = healthy
        self.last_health_check = time.time()

    def record_startup_attempt(self) -> None:
        """Record a startup attempt."""
        self.startup_attempts += 1

    def reset_startup_attempts(self) -> None:
        """Reset startup attempt counter."""
        self.startup_attempts = 0

    def get_status_summary(self) -> dict[str, str | int | bool | float | None]:
        """Get a summary dict of current status."""
        return {
            "state": self.state.value,
            "pid": self.pid,
            "port": self.port,
            "healthy": self.is_healthy(),
            "uptime": self.get_uptime(),
            "error": self.error_message,
            "startup_attempts": self.startup_attempts,
        }

    def copy(self) -> "TensorBoardInfo":
        """Create a copy of this info object."""
        return TensorBoardInfo(
            pid=self.pid,
            port=self.port,
            log_dir=self.log_dir,
            url=self.url,
            start_time=self.start_time,
            state=self.state,
            error_message=self.error_message,
            startup_attempts=self.startup_attempts,
            last_health_check=self.last_health_check,
            health_status=self.health_status,
            port_allocation=self.port_allocation,
        )


def validate_log_directory(log_dir: Path) -> None:
    """Validate that a log directory is accessible.

    Args:
        log_dir: Path to validate

    Raises:
        LogDirectoryError: If directory is not accessible
    """
    if not log_dir.exists():
        raise LogDirectoryError(log_dir, "Directory does not exist")

    if not log_dir.is_dir():
        raise LogDirectoryError(log_dir, "Path is not a directory")

    # Check if directory is readable
    try:
        list(log_dir.iterdir())
    except PermissionError as e:
        raise LogDirectoryError(log_dir, "Directory is not readable") from e


def validate_port_number(port: int) -> None:
    """Validate that a port number is in valid range.

    Args:
        port: Port number to validate

    Raises:
        ValueError: If port is outside valid range
    """
    if not (1024 <= port <= 65535):
        raise ValueError(f"Port {port} is outside valid range (1024-65535)")


def create_tensorboard_url(host: str, port: int) -> str:
    """Create a TensorBoard URL from host and port.

    Args:
        host: Hostname or IP address
        port: Port number

    Returns:
        Complete TensorBoard URL

    Example:
        >>> create_tensorboard_url("localhost", 6006)
        'http://localhost:6006'
    """
    return f"http://{host}:{port}"


def format_uptime(uptime_seconds: float) -> str:
    """Format uptime in a human-readable string.

    Args:
        uptime_seconds: Uptime in seconds

    Returns:
        Formatted uptime string

    Example:
        >>> format_uptime(3665)
        '1h 1m 5s'
    """
    if uptime_seconds < 0:
        return "0s"

    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)
