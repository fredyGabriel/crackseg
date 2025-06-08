"""Advanced abort functionality with process tree management.

This module provides sophisticated process termination capabilities including
nuclear-level cleanup, PyTorch process management, and zombie cleanup.
"""

import os
import time

import psutil

from .core import ProcessManager
from .monitoring import ProcessMonitor
from .states import (
    AbortCallback,
    AbortLevel,
    AbortProgress,
    AbortResult,
    ProcessState,
)


class ProcessAbortManager:
    """Handles complex abort operations with multiple levels.

    Provides enhanced abort functionality with comprehensive progress tracking,
    process tree management, and detailed result reporting.

    Features:
    - Multiple abort levels (GRACEFUL, FORCE, NUCLEAR)
    - Process tree termination
    - PyTorch worker cleanup
    - Zombie process cleanup
    - Progress callback support

    Example:
        >>> abort_manager = ProcessAbortManager(process_manager, monitor)
        >>> result = abort_manager.abort_training(
        ...     level=AbortLevel.GRACEFUL,
        ...     callback=progress_callback
        ... )
    """

    def __init__(
        self, process_manager: ProcessManager, monitor: ProcessMonitor
    ) -> None:
        """Initialize the abort manager.

        Args:
            process_manager: ProcessManager instance
            monitor: ProcessMonitor instance
        """
        self._process_manager = process_manager
        self._monitor = monitor

    def abort_training(
        self,
        level: AbortLevel = AbortLevel.GRACEFUL,
        timeout: float = 30.0,
        callback: AbortCallback | None = None,
    ) -> AbortResult:
        """Abort the running training process with enhanced control.

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
            >>> result = abort_manager.abort_training(
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
        if not self._process_manager.is_running:
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

            self._process_manager.update_process_info(
                state=ProcessState.STOPPING
            )

            subprocess_handle = self._process_manager.subprocess_handle
            if subprocess_handle is None:
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

            # Phase 2: Handle process termination based on level
            process_killed = False

            if level == AbortLevel.NUCLEAR:
                report_progress(
                    "terminating", "NUCLEAR: Force killing process tree", 40.0
                )
                children_killed = self._kill_process_tree_nuclear()
                self._process_manager.force_kill()
                process_killed = True

            elif level == AbortLevel.FORCE:
                report_progress(
                    "terminating", "FORCE: Killing process immediately", 40.0
                )
                self._process_manager.force_kill()
                process_killed = True

            else:  # GRACEFUL
                report_progress(
                    "terminating", "GRACEFUL: Requesting clean shutdown", 40.0
                )
                self._process_manager.terminate_gracefully(timeout)

                # Check if graceful termination worked
                if subprocess_handle.poll() is None:
                    report_progress(
                        "escalating",
                        "Graceful failed, escalating to FORCE",
                        60.0,
                    )
                    self._process_manager.force_kill()
                    level = AbortLevel.FORCE  # Update actual level used
                    warnings.append(
                        "Graceful termination timed out, escalated to FORCE"
                    )

                process_killed = True

            # Phase 3: Wait for monitoring thread
            report_progress("cleanup", "Waiting for monitoring thread", 70.0)
            self._process_manager.stop_monitoring_thread(timeout=5.0)
            if self._process_manager.is_monitoring_thread_alive():
                warnings.append("Monitoring thread did not stop cleanly")

            # Phase 4: Clean up zombie processes if needed
            if level == AbortLevel.NUCLEAR:
                report_progress("cleanup", "Cleaning zombie processes", 80.0)
                zombies_cleaned = self._cleanup_zombies()

            # Phase 5: Final cleanup
            report_progress("finalizing", "Performing final cleanup", 90.0)
            self._process_manager.update_process_info(
                state=ProcessState.ABORTED,
                return_code=(
                    subprocess_handle.returncode if subprocess_handle else None
                ),
            )

            self._process_manager.cleanup()

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

            self._process_manager.update_process_info(error_message=error_msg)

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

    def _kill_process_tree_nuclear(self) -> int:
        """Kill entire process tree aggressively (NUCLEAR mode).

        This method identifies and terminates all child processes,
        including PyTorch workers and CUDA contexts.

        Returns:
            Number of child processes terminated
        """
        children_killed = 0
        subprocess_handle = self._process_manager.subprocess_handle

        if not subprocess_handle or not subprocess_handle.pid:
            return children_killed

        try:
            # Get the main process
            main_process = psutil.Process(subprocess_handle.pid)

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
        subprocess_handle = self._process_manager.subprocess_handle

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
                            if (
                                subprocess_handle
                                and proc.pid != subprocess_handle.pid
                            ):
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
                        if self._monitor.is_related_process(proc):
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
