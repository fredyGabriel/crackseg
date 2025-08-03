"""Secure subprocess management for training execution.

This module provides robust subprocess management for executing training
runs with process control, monitoring, and resource cleanup. Designed
for long-running training processes with real-time monitoring capabilities.
"""

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from ..parsing import AdvancedOverrideParser, OverrideParsingError
from ..streaming import HydraLogWatcher, LogStreamManager, OutputStreamReader
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
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._override_parser = AdvancedOverrideParser()

        # Log streaming components
        self._stream_manager = LogStreamManager(max_buffer_size=2000)
        self._stdout_reader: OutputStreamReader | None = None
        self._hydra_watcher: HydraLogWatcher | None = None

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

    def get_stdout_reader_status(self) -> bool:
        """Get stdout reader status (public interface)."""
        return self._stdout_reader.is_running if self._stdout_reader else False

    def get_hydra_watcher_status(self) -> bool:
        """Get Hydra watcher status (public interface)."""
        return self._hydra_watcher.is_running if self._hydra_watcher else False

    def get_tracked_log_files(self) -> list[str]:
        """Get list of tracked log files (public interface)."""
        return self._hydra_watcher.tracked_files if self._hydra_watcher else []

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

            # Start monitoring thread
            self._start_monitoring()

            # Start log streaming
            self._start_log_streaming(work_dir)

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

            # Force kill if necessary
            if self._process.poll() is None:
                self._force_kill()

            # Stop log streaming
            self._stop_log_streaming()

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

    def abort_training(
        self,
        level: AbortLevel = AbortLevel.GRACEFUL,
        timeout: float = 30.0,
        callback: AbortCallback | None = None,
    ) -> AbortResult:
        """Abort the running training process with enhanced control.

        Provides enhanced feedback and multiple abort levels.

        Provides multiple abort levels with comprehensive progress tracking
        and detailed result reporting. Handles process trees, zombie cleanup,
        and UI feedback through progress callbacks.

        Args:
            level: Abort intensity level (GRACEFUL, FORCE, or NUCLEAR)
            timeout: Maximum time to wait for graceful shutdown
            callback: Optional callback for progress updates

        Returns:
            Detailed AbortResult with operation metrics and status

        Example:
            >>> def progress_callback(progress: AbortProgress) -> None:
            ...     print(f"{progress.stage}: {progress.message}")
            >>> result = manager.abort_training(
            ...     level=AbortLevel.GRACEFUL,
            ...     timeout=15.0,
            ...     callback=progress_callback
            ... )
            >>> print(f"Success: {result.success}")
        """
        start_time = time.time()
        warnings: list[str] = []
        children_killed = 0
        zombies_cleaned = 0

        # Helper to send progress updates
        def report_progress(stage: str, message: str, percent: float) -> None:
            if callback:
                elapsed = time.time() - start_time
                progress = AbortProgress(
                    stage=stage,
                    message=message,
                    progress_percent=percent,
                    elapsed_time=elapsed,
                    estimated_remaining=(
                        max(0, timeout - elapsed) if percent < 100 else 0
                    ),
                )
                callback(progress)

        # Check if process is running
        if not self.is_running:
            report_progress(
                "complete", "No process running - abort not needed", 100.0
            )
            return AbortResult(
                success=True,
                abort_level_used=level,
                process_killed=False,
                children_killed=0,
                zombies_cleaned=0,
                total_time=time.time() - start_time,
            )

        try:
            report_progress("initializing", "Starting abort operation", 10.0)

            with self._lock:
                self._process_info.state = ProcessState.STOPPING

            if self._process is None:
                report_progress(
                    "complete", "Process already cleaned up", 100.0
                )
                return AbortResult(
                    success=True,
                    abort_level_used=level,
                    process_killed=False,
                    children_killed=0,
                    zombies_cleaned=0,
                    total_time=time.time() - start_time,
                )

            # Phase 1: Stop log streaming first to prevent corruption
            report_progress("cleanup", "Stopping log streams", 20.0)
            self._stop_log_streaming()

            # Phase 2: Handle process termination based on level
            process_killed = False

            if level == AbortLevel.NUCLEAR:
                report_progress(
                    "terminating", "NUCLEAR: Force killing process tree", 40.0
                )
                children_killed = self._kill_process_tree_nuclear()
                self._force_kill()
                process_killed = True

            elif level == AbortLevel.FORCE:
                report_progress(
                    "terminating", "FORCE: Killing process immediately", 40.0
                )
                self._force_kill()
                process_killed = True

            else:  # GRACEFUL
                report_progress(
                    "terminating", "GRACEFUL: Requesting clean shutdown", 40.0
                )
                self._terminate_gracefully(timeout)

                # Check if graceful termination worked
                if self._process.poll() is None:
                    report_progress(
                        "escalating",
                        "Graceful failed, escalating to FORCE",
                        60.0,
                    )
                    self._force_kill()
                    level = AbortLevel.FORCE  # Update actual level used
                    warnings.append(
                        "Graceful termination timed out, escalated to FORCE"
                    )

                process_killed = True

            # Phase 3: Wait for monitoring thread
            report_progress("cleanup", "Waiting for monitoring thread", 70.0)
            self._stop_event.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
                if self._monitor_thread.is_alive():
                    warnings.append("Monitoring thread did not stop cleanly")

            # Phase 4: Clean up zombie processes if needed
            if level == AbortLevel.NUCLEAR:
                report_progress("cleanup", "Cleaning zombie processes", 80.0)
                zombies_cleaned = self._cleanup_zombies()

            # Phase 5: Final cleanup
            report_progress("finalizing", "Performing final cleanup", 90.0)
            with self._lock:
                self._process_info.state = ProcessState.ABORTED
                self._process_info.return_code = (
                    self._process.returncode if self._process else None
                )

            self._cleanup()

            report_progress(
                "complete", "Abort operation completed successfully", 100.0
            )

            return AbortResult(
                success=True,
                abort_level_used=level,
                process_killed=process_killed,
                children_killed=children_killed,
                zombies_cleaned=zombies_cleaned,
                total_time=time.time() - start_time,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = f"Abort operation failed: {e}"

            report_progress("error", error_msg, 100.0)

            with self._lock:
                self._process_info.error_message = error_msg

            return AbortResult(
                success=False,
                abort_level_used=level,
                process_killed=False,
                children_killed=children_killed,
                zombies_cleaned=zombies_cleaned,
                total_time=time.time() - start_time,
                error_message=error_msg,
                warnings=warnings,
            )

    def get_memory_usage(self) -> dict[str, float] | None:
        """Get memory usage of the training process.

        Returns:
            Dictionary with memory info or None if process not running
        """
        process_info = self.process_info
        if not process_info.pid:
            return None

        try:
            process = psutil.Process(process_info.pid)
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def parse_overrides_text(
        self, overrides_text: str, validate_types: bool = True
    ) -> tuple[list[str], list[str]]:
        """Parse override text and return valid overrides and errors.

        Args:
            overrides_text: Raw text containing Hydra overrides
            validate_types: Whether to perform type validation

        Returns:
            Tuple of (valid_overrides, error_messages)
        """
        try:
            self._override_parser.parse_overrides(
                overrides_text, validate_types
            )
            valid_overrides = self._override_parser.get_valid_overrides()
            errors = self._override_parser.get_parsing_errors()
            return valid_overrides, errors
        except OverrideParsingError as e:
            return [], [str(e)]

    def validate_single_override(
        self, override: str
    ) -> tuple[bool, str | None]:
        """Validate a single override string.

        Args:
            override: Single override string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self._override_parser.validate_override_string(override)

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
            is_valid, error = self.validate_single_override(override)
            if is_valid:
                validated.append(override)
            else:
                errors.append(f"{override}: {error}")

        if errors:
            raise TrainingProcessError(
                "Invalid overrides detected:\n" + "\n".join(errors)
            )

        return validated

    def _start_monitoring(self) -> None:
        """Start the process monitoring thread."""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_process,
            name="TrainingProcessMonitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_process(self) -> None:
        """Monitor the subprocess for completion or errors."""
        if self._process is None:
            return

        try:
            # Wait for process completion
            return_code = self._process.wait()

            with self._lock:
                self._process_info.return_code = return_code
                if return_code == 0:
                    self._process_info.state = ProcessState.COMPLETED
                else:
                    self._process_info.state = ProcessState.FAILED
                    self._process_info.error_message = (
                        f"Process exited with code {return_code}"
                    )

        except Exception as e:
            with self._lock:
                self._process_info.state = ProcessState.FAILED
                self._process_info.error_message = (
                    f"Process monitoring error: {e}"
                )

        finally:
            self._cleanup()

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

    def _start_log_streaming(self, working_dir: Path) -> None:
        """Start log streaming components.

        Args:
            working_dir: Working directory where Hydra creates outputs
        """
        try:
            # Start the stream manager
            self._stream_manager.start_streaming()

            # Start stdout streaming if process has stdout pipe
            if self._process and self._process.stdout:
                self._stdout_reader = OutputStreamReader(
                    stream=self._process.stdout,
                    stream_manager=self._stream_manager,
                    source_name="stdout",
                )
                self._stream_manager.register_source(
                    "stdout", self._stdout_reader
                )
                self._stdout_reader.start()

            # Look for Hydra output directory to watch
            hydra_output_dir = self._find_hydra_output_dir(working_dir)
            if hydra_output_dir:
                self._hydra_watcher = HydraLogWatcher(
                    watch_dir=hydra_output_dir,
                    stream_manager=self._stream_manager,
                    poll_interval=0.5,  # Check every 500ms
                )
                self._stream_manager.register_source(
                    "hydra_logs", self._hydra_watcher
                )
                self._hydra_watcher.start()

        except Exception as e:
            # Log streaming is non-critical, don't fail the entire process
            print(f"Warning: Failed to start log streaming: {e}")

    def _stop_log_streaming(self) -> None:
        """Stop all log streaming components."""
        try:
            # Stop stdout reader
            if self._stdout_reader:
                self._stdout_reader.stop()
                self._stdout_reader = None

            # Stop Hydra watcher
            if self._hydra_watcher:
                self._hydra_watcher.stop()
                self._hydra_watcher = None

            # Stop stream manager
            self._stream_manager.stop_streaming()

        except Exception:
            # Best effort cleanup
            pass

    def _find_hydra_output_dir(self, working_dir: Path) -> Path | None:
        """Find the most recent Hydra output directory.

        Args:
            working_dir: Working directory where training is executed

        Returns:
            Path to Hydra output directory or None if not found
        """
        try:
            outputs_dir = working_dir / "outputs"
            if not outputs_dir.exists():
                return None

            # Find most recent timestamped directory
            date_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
            if not date_dirs:
                return None

            # Get most recent date directory
            most_recent_date = max(date_dirs, key=lambda x: x.stat().st_mtime)

            # Find most recent time directory within it
            time_dirs = [d for d in most_recent_date.iterdir() if d.is_dir()]
            if not time_dirs:
                return None

            most_recent_time = max(time_dirs, key=lambda x: x.stat().st_mtime)
            return most_recent_time

        except Exception:
            return None

    def _kill_process_tree_nuclear(self) -> int:
        """Kill entire process tree aggressively (NUCLEAR mode).

        This method identifies and terminates all child processes,
        including PyTorch workers and CUDA contexts.

        Returns:
            Number of child processes terminated
        """
        children_killed = 0

        if not self._process or not self._process.pid:
            return children_killed

        try:
            # Get the main process
            main_process = psutil.Process(self._process.pid)

            # Get all children recursively
            children = main_process.children(recursive=True)

            # First, try to terminate children gracefully
            for child in children:
                try:
                    child.terminate()
                    children_killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Process may already be dead or no permission

            # Wait briefly for graceful termination
            gone, alive = psutil.wait_procs(children, timeout=3.0)

            # Force kill any remaining children
            for child in alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Special handling for PyTorch/CUDA processes
            children_killed += self._cleanup_pytorch_processes()

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Main process may already be dead
            pass
        except Exception as e:
            # Non-critical - log and continue
            print(f"Warning: Error during process tree cleanup: {e}")

        return children_killed

    def _cleanup_pytorch_processes(self) -> int:
        """Clean up PyTorch-specific processes (workers, CUDA contexts).

        Returns:
            Number of PyTorch processes cleaned
        """
        pytorch_processes_killed = 0

        try:
            # Look for Python processes that might be PyTorch workers
            # Common patterns: python train.py, torch.distributed.launch
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if proc.info["name"] in ["python", "python.exe"]:
                        cmdline = proc.info["cmdline"]
                        if cmdline and any(
                            "torch" in arg or "train" in arg for arg in cmdline
                        ):
                            # Check if it's related to our training
                            if self._process and proc.pid != self._process.pid:
                                proc.terminate()
                                pytorch_processes_killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            print(f"Warning: Error cleaning PyTorch processes: {e}")

        return pytorch_processes_killed

    def _cleanup_zombies(self) -> int:
        """Clean up zombie processes left behind after termination.

        Returns:
            Number of zombie processes cleaned
        """
        zombies_cleaned = 0

        try:
            # Find zombie processes
            for proc in psutil.process_iter(["pid", "status", "name"]):
                try:
                    if proc.info["status"] == psutil.STATUS_ZOMBIE:
                        # Check if it's related to our process
                        if self._is_related_process(proc):
                            # Try to reap the zombie
                            try:
                                proc.wait(timeout=1.0)
                                zombies_cleaned += 1
                            except (
                                psutil.TimeoutExpired,
                                psutil.NoSuchProcess,
                            ):
                                # Force cleanup on Unix systems
                                if os.name != "nt":
                                    try:
                                        os.waitpid(proc.pid, os.WNOHANG)
                                        zombies_cleaned += 1
                                    except (OSError, ChildProcessError):
                                        pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            print(f"Warning: Error during zombie cleanup: {e}")

        return zombies_cleaned

    def _is_related_process(self, proc: psutil.Process) -> bool:
        """Check if a process is related to our training process.

        Args:
            proc: Process to check

        Returns:
            True if process appears to be related to our training
        """
        if not self._process:
            return False

        try:
            # Check if it's a direct child
            if proc.ppid() == self._process.pid:
                return True

            # Check command line for training indicators
            cmdline = proc.cmdline()
            if cmdline:
                training_indicators = ["train", "run.py", "pytorch", "hydra"]
                return any(
                    indicator in " ".join(cmdline).lower()
                    for indicator in training_indicators
                )

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return False

    def get_process_tree_info(self) -> dict[str, Any]:
        """Get information about the current process tree.

        Useful for monitoring and debugging process hierarchies.

        Returns:
            Dictionary with process tree information
        """
        if not self._process or not self._process.pid:
            return {"main_process": None, "children": [], "total_processes": 0}

        try:
            main_process = psutil.Process(self._process.pid)
            children = main_process.children(recursive=True)

            children_info = []
            for child in children:
                try:
                    children_info.append(
                        {
                            "pid": child.pid,
                            "name": child.name(),
                            "status": child.status(),
                            "cpu_percent": child.cpu_percent(),
                            "memory_mb": child.memory_info().rss / 1024 / 1024,
                            "cmdline": " ".join(
                                child.cmdline()[:3]
                            ),  # First 3 args
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                "main_process": {
                    "pid": main_process.pid,
                    "name": main_process.name(),
                    "status": main_process.status(),
                    "cpu_percent": main_process.cpu_percent(),
                    "memory_mb": main_process.memory_info().rss / 1024 / 1024,
                },
                "children": children_info,
                "total_processes": 1 + len(children_info),
            }

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"main_process": None, "children": [], "total_processes": 0}

    def __del__(self) -> None:
        """Ensure proper cleanup on object deletion."""
        # Stop streaming first
        if hasattr(self, "_stream_manager"):
            try:
                self._stop_log_streaming()
            except Exception:
                pass

        # Then stop process
        if self.is_running:
            self.stop_training(timeout=5.0)
