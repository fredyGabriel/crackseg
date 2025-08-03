"""Log streaming and monitoring for training processes.

This module provides real-time log streaming capabilities for training
processes, including stdout monitoring and Hydra log file watching.
"""

from pathlib import Path

from ..streaming import HydraLogWatcher, LogStreamManager, OutputStreamReader


class LogStreamer:
    """Handle log streaming for training processes."""

    def __init__(self) -> None:
        """Initialize the log streamer."""
        self._stream_manager = LogStreamManager(max_buffer_size=2000)
        self._stdout_reader: OutputStreamReader | None = None
        self._hydra_watcher: HydraLogWatcher | None = None

    @property
    def stream_manager(self) -> LogStreamManager:
        """Get the log stream manager."""
        return self._stream_manager

    def start_log_streaming(self, working_dir: Path) -> None:
        """Start log streaming for the training process.

        Args:
            working_dir: Working directory for the process
        """
        # Note: OutputStreamReader will be initialized when process starts
        # For now, just set up the stream manager

        # Start Hydra log watcher
        hydra_output_dir = self._find_hydra_output_dir(working_dir)
        if hydra_output_dir:
            self._hydra_watcher = HydraLogWatcher(
                watch_dir=hydra_output_dir,
                stream_manager=self._stream_manager,
            )

    def stop_log_streaming(self) -> None:
        """Stop log streaming."""
        if self._stdout_reader:
            self._stdout_reader.stop()
            self._stdout_reader = None

        if self._hydra_watcher:
            self._hydra_watcher.stop()
            self._hydra_watcher = None

    def get_stdout_reader_status(self) -> bool:
        """Check if stdout reader is active.

        Returns:
            True if stdout reader is active, False otherwise
        """
        return (
            self._stdout_reader is not None and self._stdout_reader.is_active()
        )

    def get_hydra_watcher_status(self) -> bool:
        """Check if Hydra log watcher is active.

        Returns:
            True if Hydra watcher is active, False otherwise
        """
        return (
            self._hydra_watcher is not None and self._hydra_watcher.is_active()
        )

    def get_tracked_log_files(self) -> list[str]:
        """Get list of currently tracked log files.

        Returns:
            List of tracked log file paths
        """
        if self._hydra_watcher:
            return self._hydra_watcher.get_tracked_files()
        return []

    def _find_hydra_output_dir(self, working_dir: Path) -> Path | None:
        """Find Hydra output directory.

        Args:
            working_dir: Working directory to search in

        Returns:
            Path to Hydra output directory or None if not found
        """
        # Look for Hydra output directories
        for item in working_dir.iterdir():
            if item.is_dir() and item.name.startswith("outputs"):
                return item
        return None
