"""
Process monitoring and metrics collection. This module provides
comprehensive process monitoring capabilities including memory usage
tracking, CPU monitoring, and process lifecycle management.
"""

import threading
from typing import Any

import psutil

from .core import ProcessManager
from .states import ProcessState


class ProcessMonitor:
    """
    Monitors running processes and collects metrics. Provides
    comprehensive monitoring of subprocess execution including memory
    usage, CPU utilization, and process state tracking. Features: -
    Real-time memory and CPU monitoring - Process lifecycle tracking -
    Thread-safe metrics collection - Related process detection Example:
    >>> monitor = ProcessMonitor(process_manager) >>>
    monitor.start_monitoring() >>> memory_info =
    monitor.get_memory_usage()
    """

    def __init__(self, process_manager: ProcessManager) -> None:
        """
        Initialize the process monitor. Args: process_manager: ProcessManager
        instance to monitor
        """
        self._process_manager = process_manager

    def start_monitoring(self) -> None:
        """Start the process monitoring thread."""
        self._process_manager.stop_event.clear()
        monitor_thread = threading.Thread(
            target=self._monitor_process,
            name="TrainingProcessMonitor",
            daemon=True,
        )
        # Store reference in process manager
        self._process_manager.set_monitor_thread(monitor_thread)
        monitor_thread.start()

    def get_memory_usage(self) -> dict[str, float] | None:
        """
        Get memory usage of the training process. Returns: Dictionary with
        memory info or None if process not running
        """
        process_info = self._process_manager.process_info
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

    def get_process_tree_info(self) -> dict[str, Any]:
        """
        Get information about the current process tree. Useful for monitoring
        and debugging process hierarchies. Returns: Dictionary with process
        tree information
        """
        subprocess_handle = self._process_manager.subprocess_handle
        if not subprocess_handle or not subprocess_handle.pid:
            return {"main_process": None, "children": [], "total_processes": 0}

        try:
            main_process = psutil.Process(subprocess_handle.pid)
            children = main_process.children(recursive=True)

            children_info = []
            for child in children:
                try:
                    children_info.append(  # type: ignore[arg-type]
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
                "total_processes": 1 + len(children_info),  # type: ignore[arg-type]
            }

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"main_process": None, "children": [], "total_processes": 0}

    def is_related_process(self, proc: psutil.Process) -> bool:
        """
        Check if a process is related to our training process. Args: proc:
        Process to check Returns: True if process appears to be related to our
        training
        """
        subprocess_handle = self._process_manager.subprocess_handle
        if not subprocess_handle:
            return False

        try:
            # Check if it's a direct child
            if proc.ppid() == subprocess_handle.pid:
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

    def _monitor_process(self) -> None:
        """Monitor the subprocess for completion or errors."""
        subprocess_handle = self._process_manager.subprocess_handle
        if subprocess_handle is None:
            return

        try:
            # Wait for process completion
            return_code = subprocess_handle.wait()

            # Update process info through manager
            if return_code == 0:
                self._process_manager.update_process_info(
                    return_code=return_code, state=ProcessState.COMPLETED
                )
            else:
                self._process_manager.update_process_info(
                    return_code=return_code,
                    state=ProcessState.FAILED,
                    error_message=f"Process exited with code {return_code}",
                )

        except Exception as e:
            self._process_manager.update_process_info(
                state=ProcessState.FAILED,
                error_message=f"Process monitoring error: {e}",
            )

        finally:
            self._process_manager.cleanup()
