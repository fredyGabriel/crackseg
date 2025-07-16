"""TensorBoard process management and control.

This module provides the core TensorBoardManager class for managing
TensorBoard subprocess lifecycle, including startup, shutdown, health
monitoring, and port management integration.

Key Features:
- Automated port discovery and allocation
- Process lifecycle management with health monitoring
- Thread-safe status callbacks and state management
- Robust error handling and recovery mechanisms
- Integration with port registry for conflict resolution

Example:
    >>> from pathlib import Path
    >>> manager = TensorBoardManager()
    >>> success = manager.start_tensorboard(
    ...     log_dir=Path("outputs/logs/tensorboard"),
    ...     preferred_port=6006
    ... )
    >>> if success:
    ...     print(f"TensorBoard running at {manager.get_url()}")
"""

import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from .core import (
    ProcessStartupError,
    TensorBoardError,
    TensorBoardInfo,
    TensorBoardState,
    create_tensorboard_url,
    validate_log_directory,
    validate_port_number,
)
from .port_management import (
    PortRange,
    PortRegistry,
    discover_available_port,
    is_port_available,
)


class TensorBoardManager:
    """Comprehensive TensorBoard process manager.

    Manages TensorBoard subprocess with automatic port discovery, lifecycle
    control, and health monitoring. Integrates seamlessly with the existing
    port registry and provides thread-safe operations.

    This manager handles the complete lifecycle from process startup through
    health monitoring to graceful shutdown. It supports dynamic port
    allocation,
    automatic recovery, and comprehensive error handling.

    Attributes:
        port_range: Configuration for port discovery
        host: Hostname for TensorBoard binding
        startup_timeout: Maximum time to wait for startup
        health_check_interval: Interval between health checks
        manager_id: Unique identifier for this manager instance
    """

    def __init__(
        self,
        port_range: PortRange | None = None,
        host: str = "localhost",
        startup_timeout: float = 30.0,
        health_check_interval: float = 10.0,
    ) -> None:
        """Initialize TensorBoard manager.

        Args:
            port_range: Port range configuration for discovery
            host: Host interface for TensorBoard to bind to
            startup_timeout: Maximum seconds to wait for startup
            health_check_interval: Seconds between health checks
        """
        self.port_range = port_range or PortRange()
        self.host = host
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.manager_id = str(uuid.uuid4())

        # Process and state management
        self._process: subprocess.Popen[bytes] | None = None
        self._info = TensorBoardInfo()
        self._lock = threading.Lock()

        # Health monitoring
        self._health_thread: threading.Thread | None = None
        self._health_stop_event = threading.Event()

        # Status callbacks
        self._status_callbacks: list[Callable[[TensorBoardInfo], None]] = []

    @property
    def is_running(self) -> bool:
        """Check if TensorBoard is currently running."""
        with self._lock:
            return self._info.is_running()

    @property
    def info(self) -> TensorBoardInfo:
        """Get current TensorBoard information.

        Returns a copy of the current info to prevent external modification.

        Returns:
            Copy of current TensorBoard process information
        """
        with self._lock:
            return self._info.copy()

    def get_url(self) -> str | None:
        """Get TensorBoard URL if running.

        Returns:
            Complete URL string or None if not running
        """
        with self._lock:
            return self._info.url

    def get_port(self) -> int | None:
        """Get allocated port number.

        Returns:
            Port number or None if not allocated
        """
        with self._lock:
            return self._info.port

    def get_allocated_ports(self) -> list[int]:
        """Get list of all allocated ports from registry."""
        return PortRegistry.get_allocated_ports()

    def force_release_port(self, port: int) -> bool:
        """Force release a specific port from registry.

        Args:
            port: Port number to force release

        Returns:
            True if port was released, False if not found
        """
        return PortRegistry.force_release_port(port)

    def get_port_info(self, port: int) -> object | None:
        """Get allocation information for a specific port.

        Args:
            port: Port number to query

        Returns:
            PortAllocation information or None if not allocated
        """
        return PortRegistry.get_allocation_info(port)

    def is_port_allocated_by_this_manager(self, port: int) -> bool:
        """Check if port is allocated by this manager instance.

        Args:
            port: Port number to check

        Returns:
            True if allocated by this manager, False otherwise
        """
        allocation = PortRegistry.get_allocation_info(port)
        return (
            allocation is not None and allocation.manager_id == self.manager_id
        )

    def get_available_ports_in_range(self) -> list[int]:
        """Get available ports within the configured range.

        Returns:
            List of port numbers that are available for allocation
        """
        available = []
        for port in range(self.port_range.start, self.port_range.end + 1):
            if is_port_available(port):
                available.append(port)
        return available

    def start_tensorboard(
        self,
        log_dir: Path,
        preferred_port: int | None = None,
        force_restart: bool = False,
    ) -> bool:
        """Start TensorBoard process with specified configuration.

        Args:
            log_dir: Directory containing TensorBoard logs
            preferred_port: Preferred port (will try range if unavailable)
            force_restart: Whether to force restart if already running

        Returns:
            True if startup successful, False otherwise

        Raises:
            TensorBoardError: If startup fails due to configuration issues
        """
        with self._lock:
            # Check if already running
            if self.is_running and not force_restart:
                return True

            if self.is_running and force_restart:
                if not self._stop_process():
                    return False

            try:
                # Validate configuration
                validate_log_directory(log_dir)
                if preferred_port:
                    validate_port_number(preferred_port)

                # Update state
                self._update_state(TensorBoardState.STARTING)
                self._info.record_startup_attempt()

                # Discover and allocate port
                port = self._discover_and_reserve_port(preferred_port)
                if port is None:
                    self._update_state(
                        TensorBoardState.FAILED,
                        "No available ports in configured range",
                    )
                    return False

                # Build command and start process
                command = self._build_command(log_dir, port)

                try:
                    self._process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=False,
                        bufsize=0,
                    )
                except Exception as e:
                    self._cleanup_failed_start()
                    raise ProcessStartupError(
                        f"Failed to start process: {e}", port
                    ) from e

                # Update info with process details
                self._info.pid = self._process.pid
                self._info.port = port
                self._info.log_dir = log_dir
                self._info.url = create_tensorboard_url(self.host, port)
                self._info.start_time = time.time()

                # Update port allocation with process ID
                PortRegistry.update_process_id(
                    port, self._process.pid, self.manager_id
                )

                # Wait for startup completion
                if self._wait_for_startup():
                    self._update_state(TensorBoardState.RUNNING)
                    self._start_health_monitoring()
                    self._info.reset_startup_attempts()
                    return True
                else:
                    self._cleanup_failed_start()
                    return False

            except Exception as e:
                self._cleanup_failed_start()
                error_msg = f"TensorBoard startup failed: {e}"
                self._update_state(TensorBoardState.FAILED, error_msg)
                raise TensorBoardError(error_msg) from e

    def stop_tensorboard(self, timeout: float = 10.0) -> bool:
        """Stop TensorBoard process gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if not self.is_running:
                return True

            self._update_state(TensorBoardState.STOPPING)
            return self._stop_process(timeout)

    def restart_tensorboard(
        self,
        log_dir: Path | None = None,
        preferred_port: int | None = None,
    ) -> bool:
        """Restart TensorBoard process.

        Args:
            log_dir: New log directory (uses current if None)
            preferred_port: New preferred port (uses current if None)

        Returns:
            True if restart successful, False otherwise
        """
        with self._lock:
            current_log_dir = log_dir or self._info.log_dir
            current_port = preferred_port or self._info.port

            if current_log_dir is None:
                self._update_state(
                    TensorBoardState.FAILED,
                    "No log directory specified for restart",
                )
                return False

        # Stop and start (releases lock between operations)
        if not self.stop_tensorboard():
            return False

        return self.start_tensorboard(current_log_dir, current_port)

    def is_healthy(self) -> bool:
        """Check if TensorBoard process is healthy."""
        with self._lock:
            return self._info.is_healthy()

    def add_status_callback(
        self, callback: Callable[[TensorBoardInfo], None]
    ) -> None:
        """Add callback for status change notifications."""
        with self._lock:
            if callback not in self._status_callbacks:
                self._status_callbacks.append(callback)

    def remove_status_callback(
        self, callback: Callable[[TensorBoardInfo], None]
    ) -> None:
        """Remove status change callback."""
        with self._lock:
            if callback in self._status_callbacks:
                self._status_callbacks.remove(callback)

    def _discover_port(self, preferred: int | None = None) -> int | None:
        """Discover an available port using registry and socket testing.

        Args:
            preferred: Optional preferred port to try first

        Returns:
            Available port number or None if none available
        """
        return discover_available_port(self.port_range, preferred)

    def _discover_and_reserve_port(
        self, preferred: int | None = None
    ) -> int | None:
        """Discover and reserve a port for this manager.

        Tries multiple strategies to find and allocate an available port:
        1. Try preferred port if specified
        2. Try default preferred port
        3. Sequential search through range

        Args:
            preferred: Optional preferred port to try first

        Returns:
            Reserved port number or None if no ports available
        """
        # Strategy 1: Try preferred port
        if preferred and self._try_reserve_preferred_port(preferred):
            return preferred

        # Strategy 2: Try default preferred port
        if (
            preferred != self.port_range.preferred
            and self._try_reserve_default_port()
        ):
            return self.port_range.preferred

        # Strategy 3: Sequential search
        return self._try_reserve_sequential_port()

    def _try_reserve_preferred_port(
        self, preferred: int | None = None
    ) -> bool:
        """Try to reserve the preferred port."""
        if not preferred:
            return False

        if self._is_port_fully_available(preferred):
            return PortRegistry.allocate_port(preferred, self.manager_id)
        return False

    def _try_reserve_default_port(self) -> bool:
        """Try to reserve the default preferred port."""
        port = self.port_range.preferred
        if self._is_port_fully_available(port):
            return PortRegistry.allocate_port(port, self.manager_id)
        return False

    def _try_reserve_sequential_port(self) -> int | None:
        """Try to reserve ports sequentially through the range."""
        for port in range(self.port_range.start, self.port_range.end + 1):
            if (
                port != self.port_range.preferred
                and self._is_port_fully_available(port)
                and PortRegistry.allocate_port(port, self.manager_id)
            ):
                return port
        return None

    def _is_port_fully_available(self, port: int) -> bool:
        """Check if port is available both in registry and for socket
        binding."""
        return not PortRegistry.is_port_allocated(port) and is_port_available(
            port
        )

    def _release_allocated_port(self) -> bool:
        """Release the currently allocated port."""
        if self._info.port:
            success = PortRegistry.release_port(
                self._info.port, self.manager_id
            )
            if success:
                self._info.port = None
                self._info.url = None
            return success
        return True

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                sock.bind((self.host, port))
                return True
        except OSError:
            return False

    def _build_command(self, log_dir: Path, port: int) -> list[str]:
        """Build TensorBoard command line.

        Args:
            log_dir: Directory containing logs
            port: Port number for TensorBoard

        Returns:
            Command list ready for subprocess execution
        """
        # Resolve log directory to absolute path
        resolved_log_dir = log_dir.resolve()

        command = [
            "tensorboard",
            "--logdir",
            str(resolved_log_dir),
            "--port",
            str(port),
            "--host",
            self.host,
            "--load_fast",
            "false",  # Ensure complete data loading
            "--reload_interval",
            "30",  # Reload every 30 seconds
        ]

        # Add binding configuration for security
        if self.host == "localhost":
            command.extend(["--bind_all", "false"])
        else:
            command.extend(["--bind_all", "true"])

        return command

    def _wait_for_startup(self) -> bool:
        """Wait for TensorBoard to start up and become responsive."""
        if not self._process or not self._info.port:
            return False

        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            # Check if process is still running
            if self._process.poll() is not None:
                return False

            # Check if TensorBoard is responding
            if self._check_tensorboard_health():
                return True

            time.sleep(1.0)

        return False

    def _check_tensorboard_health(self) -> bool:
        """Check TensorBoard health by attempting connection."""
        if not self._info.port:
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                result = sock.connect_ex((self.host, self._info.port))
                healthy = result == 0
                self._info.update_health(healthy)
                return healthy
        except Exception:
            self._info.update_health(False)
            return False

    def _start_health_monitoring(self) -> None:
        """Start background health monitoring thread."""
        self._stop_health_monitoring()  # Stop any existing monitoring

        self._health_stop_event.clear()
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            name=f"TensorBoard-Health-{self.manager_id[:8]}",
            daemon=True,
        )
        self._health_thread.start()

    def _stop_health_monitoring(self) -> None:
        """Stop health monitoring thread."""
        if self._health_thread and self._health_thread.is_alive():
            self._health_stop_event.set()
            self._health_thread.join(timeout=5.0)
        self._health_thread = None

    def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while not self._health_stop_event.wait(self.health_check_interval):
            with self._lock:
                if not self.is_running:
                    break

                # Check process status
                if self._process and self._process.poll() is not None:
                    self._update_state(
                        TensorBoardState.FAILED, "Process terminated"
                    )
                    break

                # Check network health
                self._check_tensorboard_health()

    def _update_state(
        self, state: TensorBoardState, error: str | None = None
    ) -> None:
        """Update TensorBoard state and notify callbacks."""
        self._info.update_state(state, error)
        self._notify_status_change()

    def _notify_status_change(self) -> None:
        """Notify all registered callbacks of status change."""
        info_copy = self._info.copy()

        # Notify callbacks outside the lock to avoid deadlocks
        callbacks = self._status_callbacks.copy()

        def notify_async() -> None:
            for callback in callbacks:
                try:
                    callback(info_copy)
                except Exception:
                    # Ignore callback errors to prevent cascading failures
                    pass

        # Run notifications in background thread
        thread = threading.Thread(target=notify_async, daemon=True)
        thread.start()

    def _cleanup_failed_start(self) -> None:
        """Clean up after failed startup attempt."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            finally:
                self._process = None

        self._release_allocated_port()
        self._stop_health_monitoring()

    def _stop_process(self, timeout: float = 10.0) -> bool:
        """Stop the TensorBoard process."""
        if not self._process:
            self._update_state(TensorBoardState.STOPPED)
            return True

        try:
            # First try graceful termination
            self._process.terminate()

            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                self._process.kill()
                self._process.wait(timeout=5.0)

        except Exception as e:
            self._update_state(
                TensorBoardState.FAILED, f"Error stopping process: {e}"
            )
            return False
        finally:
            self._process = None
            self._stop_health_monitoring()
            self._release_allocated_port()
            self._cleanup()

        self._update_state(TensorBoardState.STOPPED)
        return True

    def _cleanup(self) -> None:
        """Perform final cleanup of resources."""
        # Reset info state
        self._info.pid = None
        self._info.start_time = None
        self._info.last_health_check = None
        self._info.health_status = False

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            if self.is_running:
                self.stop_tensorboard()
        except Exception:
            # Ignore errors during cleanup
            pass


def create_tensorboard_manager(
    port_range: tuple[int, int] | None = None,
    preferred_port: int = 6006,
    host: str = "localhost",
) -> TensorBoardManager:
    """Factory function to create a TensorBoard manager.

    Args:
        port_range: Tuple of (start, end) ports for discovery
        preferred_port: Preferred port to try first
        host: Host interface for binding

    Returns:
        Configured TensorBoardManager instance

    Example:
        >>> manager = create_tensorboard_manager(
        ...     port_range=(6006, 6020),
        ...     preferred_port=6006
        ... )
    """
    if port_range:
        start, end = port_range
        port_config = PortRange(start=start, end=end, preferred=preferred_port)
    else:
        port_config = PortRange(preferred=preferred_port)

    return TensorBoardManager(port_range=port_config, host=host)


# Global default manager instance
_default_manager: TensorBoardManager | None = None


def get_default_tensorboard_manager() -> TensorBoardManager:
    """Get or create the default global TensorBoard manager.

    Returns:
        Singleton TensorBoardManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = create_tensorboard_manager()
    return _default_manager
