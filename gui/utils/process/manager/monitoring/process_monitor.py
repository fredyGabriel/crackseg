"""Process monitoring and resource tracking.

This module provides real-time monitoring of training processes including
memory usage, CPU utilization, and process state tracking.
"""

import threading
import time
from typing import Any

import psutil


class ProcessMonitor:
    """Monitor training process resources and state."""

    def __init__(self) -> None:
        """Initialize the process monitor."""
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start_monitoring(self, process: Any, process_info: Any) -> None:
        """Start monitoring the process.

        Args:
            process: The subprocess to monitor
            process_info: Process information object to update
        """
        if (
            self._monitor_thread is not None
            and self._monitor_thread.is_alive()
        ):
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_process,
            args=(process, process_info),
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring the process."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)

    def _monitor_process(self, process: Any, process_info: Any) -> None:
        """Monitor process in background thread."""
        while not self._stop_event.is_set():
            try:
                if process is None or process.poll() is not None:
                    break

                # Update process info
                if hasattr(process_info, "return_code"):
                    process_info.return_code = process.poll()

                time.sleep(1.0)

            except Exception:
                break

    def get_memory_usage(self, process: Any) -> dict[str, float] | None:
        """Get current memory usage of the process.

        Args:
            process: The subprocess to check

        Returns:
            Dictionary with memory usage information or None if no active process
        """
        if process is None or process.poll() is not None:
            return None

        try:
            psutil_process = psutil.Process(process.pid)
            memory_info = psutil_process.memory_info()

            return {
                "rss": memory_info.rss / 1024 / 1024,  # MB
                "vms": memory_info.vms / 1024 / 1024,  # MB
                "percent": psutil_process.memory_percent(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
