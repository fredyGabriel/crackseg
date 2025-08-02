"""Core process management for training execution.

This module provides the main ProcessManager class for secure subprocess
execution with proper resource cleanup and process control.
"""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .log_streamer import LogStreamer
from .override_handler import OverrideHandler
from .process_cleanup import ProcessCleanup
from .process_monitor import ProcessMonitor
from .states import (
    AbortCallback,
    AbortLevel,
    AbortProgress,
    AbortResult,
    ProcessInfo,
    ProcessState,
)


class TrainingProcessError(Exception):
    """Custom exception for training process errors.

    Raised when training subprocess management fails due to:
    - Process already running when starting new training
    - Invalid command construction
    - Working directory doesn't exist
    - Process termination failures
    - Override validation errors

    Examples:
        >>> raise TrainingProcessError("Training process is already running")
        >>> raise TrainingProcessError("Invalid overrides detected: key=value")
    """

    pass


class ProcessManager:
    """Robust subprocess manager for training execution.

    Handles secure subprocess execution with proper resource cleanup,
    process control, and error handling. Designed for long-running
    training processes with real-time monitoring capabilities.

    Features:
    - Secure subprocess execution (never uses shell=True)
    - Cross-platform process group management
    - Real-time memory and CPU monitoring
    - Thread-safe operations with proper locking
    - Advanced override parsing and validation
    - Graceful and force termination support

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
        self._lock = threading.Lock()

        # Initialize specialized components
        self._monitor = ProcessMonitor()
        self._override_handler = OverrideHandler()
        self._log_streamer = LogStreamer()
        self._cleanup = ProcessCleanup()

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
    def stream_manager(self) -> Any:
        """Get the log stream manager."""
        return self._log_streamer.stream_manager

    def get_stdout_reader_status(self) -> bool:
        """Check if stdout reader is active."""
        return self._log_streamer.get_stdout_reader_status()

    def get_hydra_watcher_status(self) -> bool:
        """Check if Hydra log watcher is active."""
        return self._log_streamer.get_hydra_watcher_status()

    def get_tracked_log_files(self) -> list[str]:
        """Get list of currently tracked log files."""
        return self._log_streamer.get_tracked_log_files()

    def start_training(
        self,
        config_path: Path,
        config_name: str,
        overrides: list[str] | None = None,
        working_dir: Path | None = None,
    ) -> bool:
        """Start a new training process.

        Args:
            config_path: Path to Hydra configuration directory
            config_name: Name of the configuration to use
            overrides: List of configuration overrides
            working_dir: Working directory for the process

        Returns:
            True if process started successfully, False otherwise

        Raises:
            TrainingProcessError: If process is already running or invalid config
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise TrainingProcessError(
                    "Training process is already running"
                )

            # Validate and parse overrides
            if overrides:
                overrides = self._override_handler.validate_overrides(
                    overrides
                )

            # Build command
            command = self._override_handler.build_command(
                config_path, config_name, overrides
            )

            # Set working directory
            if working_dir is None:
                working_dir = Path.cwd()
            elif not working_dir.exists():
                raise TrainingProcessError(
                    f"Working directory doesn't exist: {working_dir}"
                )

            # Start process
            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=working_dir,
                    preexec_fn=os.setsid if os.name != "nt" else None,
                )

                # Update process info
                self._process_info.pid = self._process.pid
                self._process_info.command = command
                self._process_info.start_time = time.time()
                self._process_info.state = ProcessState.RUNNING
                self._process_info.working_directory = working_dir

                # Start monitoring and logging
                self._monitor.start_monitoring(
                    self._process, self._process_info
                )
                self._log_streamer.start_log_streaming(working_dir)

                return True

            except Exception as e:
                self._process_info.error_message = str(e)
                self._process_info.state = ProcessState.ERROR
                raise TrainingProcessError(
                    f"Failed to start training: {e}"
                ) from e

    def stop_training(self, timeout: float = 30.0) -> bool:
        """Stop the training process gracefully.

        Args:
            timeout: Maximum time to wait for graceful termination

        Returns:
            True if process stopped successfully, False otherwise
        """
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return True

            try:
                # Stop monitoring and logging
                self._monitor.stop_monitoring()
                self._log_streamer.stop_log_streaming()

                # Terminate process
                success = self._cleanup.terminate_gracefully(
                    self._process, timeout
                )

                if success:
                    self._process_info.state = ProcessState.STOPPED
                else:
                    # Force kill if graceful termination fails
                    self._cleanup.force_kill(self._process)
                    self._process_info.state = ProcessState.KILLED

                return success

            except Exception as e:
                self._process_info.error_message = str(e)
                self._process_info.state = ProcessState.ERROR
                return False

    def abort_training(
        self,
        level: AbortLevel = AbortLevel.GRACEFUL,
        timeout: float = 30.0,
        callback: AbortCallback | None = None,
    ) -> AbortResult:
        """Abort training with specified level of force.

        Args:
            level: Level of force to use for abortion
            timeout: Maximum time to wait for termination
            callback: Optional callback for progress reporting

        Returns:
            AbortResult with success status and details
        """
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return AbortResult(
                    success=True,
                    abort_level_used=level,
                    process_killed=False,
                    children_killed=0,
                    zombies_cleaned=0,
                    total_time=0.0,
                )

            def report_progress(
                stage: str, message: str, percent: float
            ) -> None:
                if callback:
                    progress = AbortProgress(
                        stage=stage,
                        message=message,
                        progress_percent=percent,
                        elapsed_time=0.0,
                    )
                    callback(progress)

            try:
                report_progress("INITIATING", "Starting abort procedure", 0.0)

                # Stop monitoring and logging
                self._monitor.stop_monitoring()
                self._log_streamer.stop_log_streaming()
                report_progress("CLEANUP", "Stopped monitoring", 25.0)

                success = False
                killed_count = 0

                if level == AbortLevel.GRACEFUL:
                    report_progress(
                        "TERMINATING", "Attempting graceful termination", 50.0
                    )
                    success = self._cleanup.terminate_gracefully(
                        self._process, timeout
                    )
                elif level == AbortLevel.FORCE:
                    report_progress(
                        "FORCE_KILL", "Force killing process", 75.0
                    )
                    self._cleanup.force_kill(self._process)
                    success = True
                elif level == AbortLevel.NUCLEAR:
                    report_progress(
                        "NUCLEAR", "Nuclear cleanup in progress", 90.0
                    )
                    killed_count = self._cleanup.kill_process_tree_nuclear()
                    success = killed_count > 0

                if success:
                    self._process_info.state = ProcessState.ABORTED
                    report_progress(
                        "COMPLETED", "Abort completed successfully", 100.0
                    )
                    return AbortResult(
                        success=True,
                        abort_level_used=level,
                        process_killed=True,
                        children_killed=killed_count,
                        zombies_cleaned=0,
                        total_time=timeout,
                    )
                else:
                    report_progress("FAILED", "Abort failed", 100.0)
                    return AbortResult(
                        success=False,
                        abort_level_used=level,
                        process_killed=False,
                        children_killed=0,
                        zombies_cleaned=0,
                        total_time=timeout,
                    )

            except Exception as e:
                self._process_info.error_message = str(e)
                self._process_info.state = ProcessState.ERROR
                report_progress("ERROR", f"Abort error: {e}", 100.0)
                return AbortResult(
                    success=False,
                    abort_level_used=level,
                    process_killed=False,
                    children_killed=0,
                    zombies_cleaned=0,
                    total_time=timeout,
                )

    def get_memory_usage(self) -> dict[str, float] | None:
        """Get current memory usage of the training process.

        Returns:
            Dictionary with memory usage information or None if no active process
        """
        return self._monitor.get_memory_usage(self._process)

    def get_process_tree_info(self) -> dict[str, Any]:
        """Get detailed information about the process tree.

        Returns:
            Dictionary with process tree information
        """
        return self._cleanup.get_process_tree_info(self._process)

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            if self._process is not None and self._process.poll() is None:
                self.stop_training()
        except Exception:
            pass
