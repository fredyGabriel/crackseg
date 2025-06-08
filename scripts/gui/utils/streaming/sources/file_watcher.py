"""Hydra log file watcher for real-time monitoring of training logs.

This module provides the HydraLogWatcher class for monitoring Hydra output
directories and streaming log file content with cross-platform compatibility.
"""

import re
import threading
import time
from datetime import datetime
from pathlib import Path

from ..core import LogLevel, LogStreamManager, StreamedLog
from ..exceptions import LogSourceError


class HydraLogWatcher:
    """Watches Hydra output directory for new log files and streams content.

    Monitors a directory for .log files created by Hydra during training
    and streams their content in real-time using a polling approach.

    Features:
    - Cross-platform file watching (polling-based)
    - Multiple log file support
    - New file detection
    - File-based log level parsing
    - Efficient incremental reading

    Example:
        >>> watcher = HydraLogWatcher(
        ...     watch_dir=Path("outputs/2024-01-01/12-00-00"),
        ...     stream_manager=manager
        ... )
        >>> watcher.start()
        >>> # ... logs are streamed as they appear ...
        >>> watcher.stop()
    """

    def __init__(
        self,
        watch_dir: Path,
        stream_manager: LogStreamManager,
        poll_interval: float = 1.0,
    ) -> None:
        """Initialize Hydra log directory watcher.

        Args:
            watch_dir: Directory to watch for .log files
            stream_manager: LogStreamManager to send logs to
            poll_interval: How often to check for changes (seconds)
        """
        self._watch_dir = Path(watch_dir)
        self._stream_manager = stream_manager
        self._poll_interval = poll_interval

        # Threading controls
        self._watcher_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running = False

        # File tracking
        self._tracked_files: dict[Path, int] = {}  # file -> last_read_position
        self._log_pattern = re.compile(r".*\.log$")

        # Hydra log parsing patterns
        self._hydra_level_pattern = re.compile(
            r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]\[(\w+)\]"
        )

    def start(self) -> None:
        """Start watching the directory for log files.

        Raises:
            LogSourceError: If watcher is already running or directory invalid
        """
        if self._is_running:
            raise LogSourceError("Hydra log watcher is already running")

        if not self._watch_dir.exists():
            raise LogSourceError(
                f"Watch directory does not exist: {self._watch_dir}"
            )

        self._is_running = True
        self._stop_event.clear()

        self._watcher_thread = threading.Thread(
            target=self._watch_directory, name="HydraLogWatcher", daemon=True
        )
        self._watcher_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop watching the directory.

        Args:
            timeout: Maximum time to wait for thread shutdown
        """
        if not self._is_running:
            return

        self._is_running = False
        self._stop_event.set()

        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        """Check if watcher is currently active."""
        return self._is_running

    @property
    def tracked_files(self) -> list[str]:
        """Get list of currently tracked log files."""
        return [str(f) for f in self._tracked_files.keys()]

    def _watch_directory(self) -> None:
        """Main directory watching loop (runs in separate thread)."""
        while not self._stop_event.is_set():
            try:
                # Scan for new or modified log files
                self._scan_for_log_files()

                # Read new content from tracked files
                self._read_file_updates()

                # Wait before next poll
                self._stop_event.wait(self._poll_interval)

            except Exception as e:
                # Watcher should be robust
                error_log = StreamedLog(
                    timestamp=datetime.now(),
                    level=LogLevel.ERROR,
                    source="hydra_watcher",
                    content=f"Directory watching error: {e}",
                )
                self._stream_manager.add_log(error_log)
                time.sleep(self._poll_interval)

        self._is_running = False

    def _scan_for_log_files(self) -> None:
        """Scan directory for new .log files."""
        try:
            for file_path in self._watch_dir.iterdir():
                if file_path.is_file() and self._log_pattern.match(
                    file_path.name
                ):
                    if file_path not in self._tracked_files:
                        # New log file detected
                        self._tracked_files[file_path] = 0

                        # Log discovery
                        discovery_log = StreamedLog(
                            timestamp=datetime.now(),
                            level=LogLevel.INFO,
                            source="hydra_watcher",
                            content=f"Tracking: {file_path.name}",
                        )
                        self._stream_manager.add_log(discovery_log)

        except OSError:
            # Directory might be temporarily unavailable
            pass

    def _read_file_updates(self) -> None:
        """Read new content from all tracked files."""
        for file_path, last_position in list(self._tracked_files.items()):
            try:
                if not file_path.exists():
                    # File was deleted
                    del self._tracked_files[file_path]
                    continue

                current_size = file_path.stat().st_size
                if current_size > last_position:
                    # File has new content
                    self._read_file_content(file_path, last_position)

            except OSError:
                # File might be temporarily unavailable
                continue

    def _read_file_content(self, file_path: Path, start_position: int) -> None:
        """Read new content from a specific file.

        Args:
            file_path: Path to the log file
            start_position: Byte position to start reading from
        """
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                f.seek(start_position)

                line_number = start_position  # Approximate line tracking
                for line in f:
                    line = line.rstrip("\n\r")
                    if line:  # Skip empty lines
                        self._process_hydra_line(
                            line, file_path.name, line_number
                        )
                    line_number += 1

                # Update position
                self._tracked_files[file_path] = f.tell()

        except (OSError, UnicodeDecodeError) as e:
            # Log file reading error
            error_log = StreamedLog(
                timestamp=datetime.now(),
                level=LogLevel.ERROR,
                source="hydra_watcher",
                content=f"Error reading {file_path.name}: {e}",
            )
            self._stream_manager.add_log(error_log)

    def _process_hydra_line(
        self, line: str, filename: str, line_number: int
    ) -> None:
        """Process a line from a Hydra log file.

        Args:
            line: Log line content
            filename: Name of the source file
            line_number: Line number in file
        """
        # Parse Hydra log format for level
        level = self._parse_hydra_log_level(line)

        # Create log entry
        log_entry = StreamedLog(
            timestamp=datetime.now(),
            level=level,
            source=f"hydra:{filename}",
            content=line,
            line_number=line_number,
        )

        # Send to stream manager
        self._stream_manager.add_log(log_entry)

    def _parse_hydra_log_level(self, line: str) -> LogLevel:
        """Parse log level from Hydra log line format.

        Args:
            line: Hydra log line

        Returns:
            Parsed LogLevel (defaults to INFO)
        """
        match = self._hydra_level_pattern.match(line)
        if match:
            level_str = match.group(2).upper()
            try:
                return LogLevel(level_str)
            except ValueError:
                pass

        # Fallback to content-based detection
        if "ERROR" in line.upper():
            return LogLevel.ERROR
        elif "WARNING" in line.upper() or "WARN" in line.upper():
            return LogLevel.WARNING
        elif "DEBUG" in line.upper():
            return LogLevel.DEBUG

        return LogLevel.INFO
