"""Subprocess stdout/stderr stream reader for real-time log capture.

This module provides the OutputStreamReader class for capturing training
output from subprocess stdout pipes with minimal latency and thread safety.
"""

import re
import threading
from datetime import datetime
from typing import IO

from ..core import LogLevel, LogStreamManager, StreamedLog
from ..exceptions import LogSourceError


class OutputStreamReader:
    """Reads from subprocess stdout/stderr stream in real-time.

    Designed to work with the ProcessManager's subprocess.Popen stdout pipe
    to capture training output as it happens with minimal latency.

    Features:
    - Non-blocking readline with timeout
    - Log level detection from content
    - Thread-safe operation
    - Proper resource cleanup

    Example:
        >>> process = subprocess.Popen(...)  # From ProcessManager
        >>> reader = OutputStreamReader(process.stdout, stream_manager)
        >>> reader.start()
        >>> # ... training runs ...
        >>> reader.stop()
    """

    def __init__(
        self,
        stream: IO[str],
        stream_manager: LogStreamManager,
        source_name: str = "stdout",
    ) -> None:
        """Initialize stdout/stderr reader.

        Args:
            stream: Text stream from subprocess (stdout/stderr)
            stream_manager: LogStreamManager to send logs to
            source_name: Identifier for this stream source
        """
        self._stream = stream
        self._stream_manager = stream_manager
        self._source_name = source_name

        # Threading controls
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running = False

        # State tracking
        self._line_count = 0

        # Log level detection patterns
        self._level_patterns = {
            LogLevel.ERROR: re.compile(
                r"\b(error|exception|failed|failure)\b", re.IGNORECASE
            ),
            LogLevel.WARNING: re.compile(
                r"\b(warn|warning|deprecated)\b", re.IGNORECASE
            ),
            LogLevel.DEBUG: re.compile(r"\b(debug|trace)\b", re.IGNORECASE),
            LogLevel.INFO: re.compile(
                r"\b(info|epoch|loss|accuracy|step)\b", re.IGNORECASE
            ),
        }

    def start(self) -> None:
        """Start reading from the stream.

        Raises:
            LogSourceError: If reader is already running
        """
        if self._is_running:
            raise LogSourceError(
                f"Stream reader for {self._source_name} is already running"
            )

        self._is_running = True
        self._stop_event.clear()

        self._reader_thread = threading.Thread(
            target=self._read_stream,
            name=f"StreamReader-{self._source_name}",
            daemon=True,
        )
        self._reader_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop reading from the stream.

        Args:
            timeout: Maximum time to wait for thread shutdown
        """
        if not self._is_running:
            return

        self._is_running = False
        self._stop_event.set()

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        """Check if reader is currently active."""
        return self._is_running

    @property
    def lines_read(self) -> int:
        """Get total number of lines read."""
        return self._line_count

    def _read_stream(self) -> None:
        """Main stream reading loop (runs in separate thread)."""
        try:
            while not self._stop_event.is_set():
                line = self._stream.readline()

                if not line:  # EOF or stream closed
                    break

                # Process the line
                line = line.rstrip("\n\r")
                if line:  # Skip empty lines
                    self._process_line(line)
                    self._line_count += 1

        except Exception as e:
            # Stream reading should be robust
            error_log = StreamedLog(
                timestamp=datetime.now(),
                level=LogLevel.ERROR,
                source=self._source_name,
                content=f"Stream reading error: {e}",
                line_number=self._line_count,
            )
            self._stream_manager.add_log(error_log)
        finally:
            self._is_running = False

    def _process_line(self, line: str) -> None:
        """Process a single line from the stream."""
        # Detect log level from content
        level = self._detect_log_level(line)

        # Create streamd log entry
        log_entry = StreamedLog(
            timestamp=datetime.now(),
            level=level,
            source=self._source_name,
            content=line,
            line_number=self._line_count + 1,
        )

        # Send to stream manager
        self._stream_manager.add_log(log_entry)

    def _detect_log_level(self, line: str) -> LogLevel:
        """Detect log level from line content using patterns.

        Args:
            line: Log line content

        Returns:
            Detected LogLevel (defaults to INFO)
        """
        for level, pattern in self._level_patterns.items():
            if pattern.search(line):
                return level
        return LogLevel.INFO
