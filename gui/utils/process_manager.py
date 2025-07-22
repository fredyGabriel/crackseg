"""
Process Management Utilities This module provides a class to manage
long-running subprocesses, capture their output in real-time, and
handle their lifecycle.
"""

import subprocess
import threading
from collections import deque
from typing import IO


class ProcessManager:
    """
    Manages a subprocess for real-time output streaming in a separate
    thread. This class is designed to run a command (like a training
    script) as a non-blocking subprocess, continuously capturing its
    stdout. It is ideal for use in GUIs like Streamlit where the main
    thread cannot be blocked. Attributes: command (List[str]): The command
    to execute. process (subprocess.Popen | None): The subprocess
    instance. logs (Deque[str]): A deque holding the most recent log
    lines. is_running (bool): A flag indicating if the process is active.
    """

    def __init__(self, command: list[str]):
        self.command = command
        self.process: subprocess.Popen[str] | None = None
        self.logs: deque[str] = deque(maxlen=1000)  # Store last 1000 lines
        self.is_running = False
        self._thread: threading.Thread | None = None

    def _stream_output(self, pipe: IO[str]) -> None:
        """
        Reads from a pipe (e.g., stdout) and appends lines to the log deque.
        This method is intended to be run in a separate thread.
        """
        for line in iter(pipe.readline, ""):
            self.logs.append(line.strip())
        pipe.close()

    def start(self) -> None:
        """Starts the subprocess and the log streaming threads."""
        if self.is_running:
            return

        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # Line-buffered
        )
        self.is_running = True

        # Start a thread to read stdout to avoid blocking the main thread.
        self._thread = threading.Thread(
            target=self._stream_output, args=(self.process.stdout,)
        )
        self._thread.daemon = True
        self._thread.start()

        # It's also possible to start another thread for stderr if needed
        # For now, we'll focus on stdout

    def stop(self) -> None:
        """
        Stops the running subprocess gracefully. It first tries to terminate
        the process and waits for a timeout. If the process does not
        terminate, it is killed forcefully.
        """
        if self.process and self.is_running:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.is_running = False

    def get_logs(self) -> list[str]:
        """
        Returns a thread-safe copy of the current logs.
        """
        return list(self.logs)

    def check_status(self) -> None:
        """
        Updates the running status of the process by polling it. This should
        be called periodically to know when a process has finished.
        """
        if (
            self.is_running
            and self.process
            and self.process.poll() is not None
        ):
            self.is_running = False
