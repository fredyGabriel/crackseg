"""Core subprocess management for training execution.

This module provides the main ProcessManager class with core functionality
for secure subprocess execution, process control, and resource cleanup.
Designed for long-running training processes.
"""

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from ..parsing import AdvancedOverrideParser
from ..streaming import LogStreamManager, OutputStreamReader
from .states import ProcessInfo, ProcessState, TrainingProcessError


class ProcessManager:
    """Core subprocess manager for training execution.

    Handles secure subprocess execution with proper resource cleanup,
    process control, and error handling. Designed for long-running
    training processes with real-time monitoring capabilities.

    Features:
    - Secure subprocess execution (never uses shell=True)
    - Cross-platform process group management
    - Thread-safe operations with proper locking
    - Integration with monitoring and log streaming

    Example:
        >>> manager = ProcessManager()
        >>> success = manager.start_training(
        ...     config_path=Path("configs"),
        ...     config_name="train_baseline",
        ...     overrides=["trainer.max_epochs=50"]
        ... )
    """

    def __init__(self) -> None:
        """Initialize the process manager."""
        self._process: subprocess.Popen[str] | None = None
        self._process_info = ProcessInfo()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._override_parser = AdvancedOverrideParser()

        # Log streaming components
        self._stream_manager = LogStreamManager(max_buffer_size=2000)
        self._stdout_reader: OutputStreamReader | None = None

    @property
    def process_info(self) -> ProcessInfo:
        """Get current process information (thread-safe)."""
        with self._lock:
            return ProcessInfo(
                pid=self._process_info.pid,
                command=self._process_info.command.copy(),
                start_time=self._process_info.start_time,
                state=self._process_info.state,
                return_code=self._process_info.return_code,
                error_message=self._process_info.error_message,
                working_directory=self._process_info.working_directory,
            )

    @property
    def is_running(self) -> bool:
        """Check if process is currently running."""
        return self.process_info.state in {
            ProcessState.STARTING,
            ProcessState.RUNNING,
        }

    @property
    def stream_manager(self) -> LogStreamManager:
        """Get the log stream manager for callback registration."""
        return self._stream_manager

    @property
    def subprocess_handle(self) -> subprocess.Popen[str] | None:
        """Get the subprocess handle (for monitoring access)."""
        return self._process

    @property
    def stop_event(self) -> threading.Event:
        """Get the stop event (for monitoring access)."""
        return self._stop_event

    @property
    def lock(self) -> threading.Lock:
        """Get the thread lock (for monitoring access)."""
        return self._lock

    def start_training(
        self,
        config_path: Path,
        config_name: str,
        overrides: list[str] | None = None,
        working_dir: Path | None = None,
    ) -> bool:
        """Start a training process with the specified configuration.

        Args:
            config_path: Path to the configuration directory
            config_name: Name of the configuration file (without .yaml)
            overrides: List of Hydra overrides (e.g.,
                ['trainer.max_epochs=50'])
            working_dir: Working directory for the process (defaults to
                project root)

        Returns:
            True if process started successfully, False otherwise

        Raises:
            TrainingProcessError: If process is already running or command
                invalid
        """
        if self.is_running:
            raise TrainingProcessError("Training process is already running")

        # Build command safely (no shell=True)
        command = self._build_command(config_path, config_name, overrides)

        # Validate working directory
        work_dir = working_dir or Path.cwd()
        if not work_dir.exists():
            raise TrainingProcessError(
                f"Working directory does not exist: {work_dir}"
            )

        try:
            with self._lock:
                self._process_info = ProcessInfo(
                    command=command,
                    working_directory=work_dir,
                    state=ProcessState.STARTING,
                    start_time=time.time(),
                )

            # Start subprocess with secure configuration
            self._process = subprocess.Popen(
                command,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                # Security: Never use shell=True with user input
                shell=False,
                # Prevent inheriting file descriptors
                close_fds=True,
                # Set process group for clean termination
                preexec_fn=os.setsid if os.name != "nt" else None,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if os.name == "nt"
                    else 0
                ),
            )

            with self._lock:
                self._process_info.pid = self._process.pid
                self._process_info.state = ProcessState.RUNNING

            return True

        except (subprocess.SubprocessError, OSError, PermissionError) as e:
            with self._lock:
                self._process_info.state = ProcessState.FAILED
                self._process_info.error_message = (
                    f"Failed to start process: {e}"
                )
            self._cleanup()
            return False

    def stop_training(self, timeout: float = 30.0) -> bool:
        """Stop the running training process gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown

        Returns:
            True if process stopped successfully, False otherwise
        """
        if not self.is_running:
            return True

        with self._lock:
            self._process_info.state = ProcessState.STOPPING

        try:
            if self._process is None:
                return True

            # Try graceful termination first
            self._terminate_gracefully(timeout)

            if self._process.poll() is None:  # Force kill if necessary
                self._force_kill()

            # Wait for monitoring thread to finish
            self._stop_event.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)

            with self._lock:
                self._process_info.state = ProcessState.ABORTED
                self._process_info.return_code = self._process.returncode

            self._cleanup()
            return True

        except Exception as e:
            with self._lock:
                self._process_info.error_message = (
                    f"Error stopping process: {e}"
                )
            return False

    def update_process_info(self, **kwargs: Any) -> None:
        """Update process info (thread-safe interface for monitoring)."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._process_info, key):
                    setattr(self._process_info, key, value)

    def terminate_gracefully(self, timeout: float) -> None:
        """Public interface for graceful termination."""
        self._terminate_gracefully(timeout)

    def force_kill(self) -> None:
        """Public interface for force kill."""
        self._force_kill()

    def cleanup(self) -> None:
        """Public interface for cleanup."""
        self._cleanup()

    def stop_monitoring_thread(self, timeout: float = 5.0) -> None:
        """Stop monitoring thread safely."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)

    def is_monitoring_thread_alive(self) -> bool:
        """Check if monitoring thread is alive."""
        return bool(self._monitor_thread and self._monitor_thread.is_alive())

    def set_stdout_reader(self, reader: OutputStreamReader | None) -> None:
        """Set stdout reader reference."""
        self._stdout_reader = reader

    def get_stdout_reader(self) -> OutputStreamReader | None:
        """Get stdout reader reference."""
        return self._stdout_reader

    def clear_stdout_reader(self) -> None:
        """Clear stdout reader reference."""
        self._stdout_reader = None

    def set_monitor_thread(self, thread: threading.Thread | None) -> None:
        """Set monitor thread reference."""
        self._monitor_thread = thread

    def _build_command(
        self,
        config_path: Path,
        config_name: str,
        overrides: list[str] | None = None,
    ) -> list[str]:
        """Build the training command safely.

        Args:
            config_path: Path to configuration directory
            config_name: Configuration file name
            overrides: List of Hydra overrides

        Returns:
            Command as list of strings (safe for subprocess)
        """
        # Validate inputs
        if not config_path.exists():
            raise TrainingProcessError(
                f"Config path does not exist: {config_path}"
            )

        if not config_name:
            raise TrainingProcessError("Config name cannot be empty")

        # Build base command
        command = [
            "python",
            "run.py",
            f"--config-path={config_path.resolve()}",
            f"--config-name={config_name}",
        ]

        # Add overrides safely using advanced parser
        if overrides:
            validated_overrides = self._validate_overrides(overrides)
            command.extend(validated_overrides)

        return command

    def _validate_overrides(self, overrides: list[str]) -> list[str]:
        """Validate a list of override strings.

        Args:
            overrides: List of override strings

        Returns:
            List of validated override strings

        Raises:
            TrainingProcessError: If any override is invalid
        """
        validated = []
        errors = []

        for override in overrides:
            is_valid, error = self._override_parser.validate_override_string(
                override
            )
            if is_valid:
                validated.append(override)
            else:
                errors.append(f"{override}: {error}")

        if errors:
            raise TrainingProcessError(
                "Invalid overrides detected:\n" + "\n".join(errors)
            )

        return validated

    def _terminate_gracefully(self, timeout: float) -> None:
        """Attempt graceful process termination.

        Args:
            timeout: Maximum time to wait for termination
        """
        if self._process is None:
            return

        try:
            if os.name == "nt":
                # Windows: Send CTRL_BREAK_EVENT to process group
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Unix: Send SIGTERM to process group
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)

            # Wait for graceful shutdown
            self._process.wait(timeout=timeout)

        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            # Process didn't terminate gracefully, will be force killed
            pass

    def _force_kill(self) -> None:
        """Force kill the process and its children."""
        if self._process is None:
            return

        try:
            if os.name == "nt":
                # Windows: Terminate process tree
                self._process.terminate()
            else:
                # Unix: Kill entire process group
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)

            # Final wait
            self._process.wait(timeout=5.0)

        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            # Process may already be dead
            pass

    def _cleanup(self) -> None:
        """Clean up resources after process completion."""
        if self._process:
            try:
                # Ensure pipes are closed
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()
                if self._process.stdin:
                    self._process.stdin.close()
            except Exception:
                pass  # Ignore cleanup errors

            self._process = None

    def __del__(self) -> None:
        """Ensure proper cleanup on object deletion."""
        if self.is_running:
            self.stop_training(timeout=5.0)
