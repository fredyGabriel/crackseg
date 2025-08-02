"""Log streaming integration for training processes.

This module provides integration between process management and log streaming
components, handling both stdout capture and Hydra log file monitoring.
"""

from pathlib import Path

from ..streaming import HydraLogWatcher, OutputStreamReader
from .core import ProcessManager


class LogStreamingIntegrator:
    """Integrates log streaming with process management.

    Manages the coordination between subprocess execution and real-time
    log streaming from both stdout and Hydra log files.

    Features:
    - Stdout stream capture
    - Hydra log file monitoring
    - Automatic output directory detection
    - Stream lifecycle management

    Example:
        >>> integrator = LogStreamingIntegrator(process_manager)
        >>> integrator.start_log_streaming(Path("./"))
        >>> files = integrator.get_tracked_log_files()
    """

    def __init__(self, process_manager: ProcessManager) -> None:
        """Initialize the log streaming integrator.

        Args:
            process_manager: ProcessManager instance to integrate with
        """
        self._process_manager = process_manager
        self._hydra_watcher: HydraLogWatcher | None = None

    def start_log_streaming(self, working_dir: Path) -> None:
        """Start log streaming components.

        Args:
            working_dir: Working directory where Hydra creates outputs
        """
        try:
            # Start the stream manager
            self._process_manager.stream_manager.start_streaming()

            # Start stdout streaming if process has stdout pipe
            subprocess_handle = self._process_manager.subprocess_handle
            if subprocess_handle and subprocess_handle.stdout:
                stdout_reader = OutputStreamReader(
                    stream=subprocess_handle.stdout,
                    stream_manager=self._process_manager.stream_manager,
                    source_name="stdout",
                )
                self._process_manager.stream_manager.register_source(
                    "stdout", stdout_reader
                )
                # Store reference in process manager
                self._process_manager.set_stdout_reader(stdout_reader)
                stdout_reader.start()

            # Look for Hydra output directory to watch
            hydra_output_dir = self._find_hydra_output_dir(working_dir)
            if hydra_output_dir:
                self._hydra_watcher = HydraLogWatcher(
                    watch_dir=hydra_output_dir,
                    stream_manager=self._process_manager.stream_manager,
                    poll_interval=0.5,  # Check every 500ms
                )
                self._process_manager.stream_manager.register_source(
                    "hydra_logs", self._hydra_watcher
                )
                self._hydra_watcher.start()

        except Exception as e:
            # Log streaming is non-critical, don't fail the entire process
            print(f"Warning: Failed to start log streaming: {e}")

    def stop_log_streaming(self) -> None:
        """Stop all log streaming components."""
        try:
            # Stop stdout reader
            stdout_reader = self._process_manager.get_stdout_reader()
            if stdout_reader:
                stdout_reader.stop()
                self._process_manager.clear_stdout_reader()

            # Stop Hydra watcher
            if self._hydra_watcher:
                self._hydra_watcher.stop()
                self._hydra_watcher = None

            # Stop stream manager
            self._process_manager.stream_manager.stop_streaming()

        except Exception:
            # Best effort cleanup
            pass

    def get_stdout_reader_status(self) -> bool:
        """Get stdout reader status."""
        stdout_reader = self._process_manager.get_stdout_reader()
        return bool(stdout_reader and stdout_reader.is_running)

    def get_hydra_watcher_status(self) -> bool:
        """Get Hydra watcher status."""
        return self._hydra_watcher.is_running if self._hydra_watcher else False

    def get_tracked_log_files(self) -> list[str]:
        """Get list of tracked log files."""
        return self._hydra_watcher.tracked_files if self._hydra_watcher else []

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
