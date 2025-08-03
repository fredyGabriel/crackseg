"""Enhanced abort functionality and process tree management.

This module provides comprehensive abort capabilities with progress tracking,
process tree management, and orphan cleanup for training processes.
"""

import os
import time
from typing import Any

import psutil

from ..process import AbortCallback, AbortLevel, AbortResult
from .orchestrator import get_process_manager


def abort_training_session(
    level: AbortLevel = AbortLevel.GRACEFUL,
    timeout: float = 30.0,
    callback: AbortCallback | None = None,
) -> AbortResult:
    """Abort the current training session with enhanced control.

    High-level wrapper around ProcessManager.abort_training that provides
    comprehensive abort capabilities with progress tracking.

    Args:
        level: Abort intensity (GRACEFUL, FORCE, or NUCLEAR)
        timeout: Maximum time to wait for graceful shutdown
        callback: Optional callback for progress updates

    Returns:
        AbortResult with detailed operation metrics

    Example:
        >>> def progress_handler(progress: AbortProgress) -> None:
        ...     print(f"{progress.stage}: {progress.progress_percent:.1f}%")
        >>> result = abort_training_session(
        ...     level=AbortLevel.FORCE,
        ...     callback=progress_handler
        ... )
        >>> if result.success:
        ...     print(f"Aborted in {result.total_time:.2f}s")
        >>> else:
        ...     print(f"Abort failed: {result.error_message}")
    """
    manager = get_process_manager()
    return manager.abort_training(
        level=level, timeout=timeout, callback=callback
    )


def get_process_tree_info() -> dict[str, Any]:
    """Get information about the current training process tree.

    Useful for monitoring process hierarchies and debugging
    issues with child processes.

    Returns:
        Dictionary with process tree information including
        main process, children, and resource usage

    Example:
        >>> tree_info = get_process_tree_info()
        >>> print(f"Total processes: {tree_info['total_processes']}")
        >>> for child in tree_info['children']:
        ...     print(f"Child PID {child['pid']}: {child['name']}")
    """
    manager = get_process_manager()
    return manager.get_process_tree_info()


def force_cleanup_orphans() -> dict[str, Any]:
    """Force cleanup of orphaned processes from previous training runs.

    This function performs a comprehensive cleanup of orphaned training
    processes that may have been left behind due to improper shutdowns.
    Use with caution as it may affect other Python processes.

    Returns:
        Dictionary with cleanup statistics and warnings

    Example:
        >>> cleanup_result = force_cleanup_orphans()
        >>> print(f"Cleaned {cleanup_result['processes_killed']} orphans")
        >>> for warning in cleanup_result['warnings']:
        ...     print(f"Warning: {warning}")
    """
    warnings: list[str] = []
    processes_killed = 0
    pytorch_processes_killed = 0
    zombies_cleaned = 0

    try:
        # Look for orphaned training processes
        for proc in psutil.process_iter(["pid", "name", "cmdline", "status"]):
            try:
                # Skip if not a Python process
                if proc.info["name"] not in ["python", "python.exe"]:
                    continue

                cmdline = proc.info["cmdline"]
                if not cmdline:
                    continue

                # Check for training indicators
                cmdline_str = " ".join(cmdline).lower()
                training_indicators = [
                    "run.py",
                    "train.py",
                    "hydra",
                    "pytorch",
                ]

                if any(
                    indicator in cmdline_str
                    for indicator in training_indicators
                ):
                    # Check if it's a zombie
                    if proc.info["status"] == psutil.STATUS_ZOMBIE:
                        try:
                            proc.wait(timeout=1.0)
                            zombies_cleaned += 1
                        except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                            if os.name != "nt":
                                try:
                                    os.waitpid(proc.pid, os.WNOHANG)
                                    zombies_cleaned += 1
                                except (OSError, ChildProcessError):
                                    pass
                    else:
                        # Check if it's truly orphaned
                        try:
                            # Heuristic: if running >10 minutes
                            # and shows low CPU usage, it might be stuck
                            create_time = proc.create_time()
                            current_time = time.time()
                            if current_time - create_time > 600:  # 10 minutes
                                cpu_percent = proc.cpu_percent(interval=1.0)
                                if cpu_percent < 1.0:  # Very low CPU usage
                                    proc.terminate()
                                    processes_killed += 1
                                    warnings.append(
                                        f"Terminated PID {proc.pid}"
                                    )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

                # Special handling for torch.distributed processes
                if (
                    "torch.distributed" in cmdline_str
                    or "torchrun" in cmdline_str
                ):
                    try:
                        proc.terminate()
                        pytorch_processes_killed += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    except Exception as e:
        warnings.append(f"Error during orphan cleanup: {e}")

    return {
        "success": True,
        "processes_killed": processes_killed,
        "pytorch_processes_killed": pytorch_processes_killed,
        "zombies_cleaned": zombies_cleaned,
        "total_cleaned": processes_killed
        + pytorch_processes_killed
        + zombies_cleaned,
        "warnings": warnings,
    }
