"""Process management module for CrackSeg training execution.

This module provides comprehensive subprocess management capabilities for
secure training execution, including process monitoring, advanced abort
functionality, and real-time log streaming integration.

Refactored Architecture:
- core: Core ProcessManager functionality
- monitoring: Process monitoring and metrics
- abort_system: Advanced abort with multiple levels
- override_parser: Hydra override parsing and validation
- log_integration: Log streaming integration
- states: Shared state types and exceptions

Example:
    >>> from gui.utils.process import TrainingManager
    >>> manager = TrainingManager()
    >>> success = manager.start_training(
    ...     config_path=Path("configs"),
    ...     config_name="baseline"
    ... )
"""

from pathlib import Path

from .abort_system import ProcessAbortManager
from .core import ProcessManager as CoreProcessManager
from .log_integration import LogStreamingIntegrator
from .monitoring import ProcessMonitor
from .override_parser import HydraOverrideManager

# Backward compatibility imports
from .states import (
    AbortCallback,
    AbortLevel,
    AbortProgress,
    AbortResult,
    ProcessInfo,
    ProcessState,
    TrainingProcessError,
)


class TrainingManager:
    """Integrated training process manager with all capabilities.

    This class combines all the refactored components into a single
    high-level interface that provides full backward compatibility
    with the original ProcessManager while offering modular access
    to specialized functionality.

    Features:
    - Full ProcessManager compatibility
    - Modular component access
    - Advanced abort functionality
    - Real-time log streaming
    - Override parsing and validation
    - Process monitoring and metrics

    Example:
        >>> manager = TrainingManager()
        >>> # Core functionality
        >>> success = manager.start_training(Path("configs"), "baseline")
        >>>
        >>> # Advanced features
        >>> abort_result = manager.abort_training(level=AbortLevel.NUCLEAR)
        >>> memory_info = manager.get_memory_usage()
        >>> valid_overrides, errors = manager.parse_overrides_text(
        ...     "epochs=100"
        ... )
    """

    def __init__(self) -> None:
        """Initialize the integrated training manager."""
        # Core components
        self._core = CoreProcessManager()
        self._monitor = ProcessMonitor(self._core)
        self._abort_manager = ProcessAbortManager(self._core, self._monitor)
        self._override_manager = HydraOverrideManager()
        self._log_integrator = LogStreamingIntegrator(self._core)

    # === Core ProcessManager Interface (Backward Compatibility) ===

    @property
    def process_info(self) -> ProcessInfo:
        """Get current process information (thread-safe)."""
        return self._core.process_info

    @property
    def is_running(self) -> bool:
        """Check if process is currently running."""
        return self._core.is_running

    @property
    def stream_manager(self):
        """Get the log stream manager for callback registration."""
        return self._core.stream_manager

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
            overrides: List of Hydra overrides
            working_dir: Working directory for the process

        Returns:
            True if process started successfully, False otherwise
        """
        success = self._core.start_training(
            config_path, config_name, overrides, working_dir
        )

        if success:
            # Start monitoring
            self._monitor.start_monitoring()
            # Start log streaming
            work_dir = (
                working_dir or config_path.parent
                if hasattr(config_path, "parent")
                else None
            )
            if work_dir:
                self._log_integrator.start_log_streaming(work_dir)

        return success

    def stop_training(self, timeout: float = 30.0) -> bool:
        """Stop the running training process gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown

        Returns:
            True if process stopped successfully, False otherwise
        """
        # Stop log streaming first
        self._log_integrator.stop_log_streaming()
        return self._core.stop_training(timeout)

    # === Enhanced Features ===

    def abort_training(
        self,
        level: AbortLevel = AbortLevel.GRACEFUL,
        timeout: float = 30.0,
        callback: AbortCallback | None = None,
    ) -> AbortResult:
        """Abort the running training process with enhanced control.

        Args:
            level: Abort intensity level (GRACEFUL, FORCE, or NUCLEAR)
            timeout: Maximum time to wait for graceful shutdown
            callback: Optional callback for progress updates

        Returns:
            Detailed AbortResult with operation metrics and status
        """
        # Stop log streaming first
        self._log_integrator.stop_log_streaming()
        return self._abort_manager.abort_training(level, timeout, callback)

    def get_memory_usage(self):
        """Get memory usage of the training process."""
        return self._monitor.get_memory_usage()

    def get_process_tree_info(self):
        """Get information about the current process tree."""
        return self._monitor.get_process_tree_info()

    def parse_overrides_text(
        self, overrides_text: str, validate_types: bool = True
    ):
        """Parse override text and return valid overrides and errors."""
        return self._override_manager.parse_overrides_text(
            overrides_text, validate_types
        )

    def validate_single_override(self, override: str):
        """Validate a single override string."""
        return self._override_manager.validate_single_override(override)

    # === Log Streaming Interface ===

    def get_stdout_reader_status(self) -> bool:
        """Get stdout reader status."""
        return self._log_integrator.get_stdout_reader_status()

    def get_hydra_watcher_status(self) -> bool:
        """Get Hydra watcher status."""
        return self._log_integrator.get_hydra_watcher_status()

    def get_tracked_log_files(self) -> list[str]:
        """Get list of tracked log files."""
        return self._log_integrator.get_tracked_log_files()

    # === Component Access ===

    @property
    def core(self) -> CoreProcessManager:
        """Access to core process manager."""
        return self._core

    @property
    def monitor(self) -> ProcessMonitor:
        """Access to process monitor."""
        return self._monitor

    @property
    def abort_manager(self) -> ProcessAbortManager:
        """Access to abort manager."""
        return self._abort_manager

    @property
    def override_manager(self) -> HydraOverrideManager:
        """Access to override manager."""
        return self._override_manager

    @property
    def log_integrator(self) -> LogStreamingIntegrator:
        """Access to log integrator."""
        return self._log_integrator

    def __del__(self) -> None:
        """Ensure proper cleanup on object deletion."""
        if self.is_running:
            self.stop_training(timeout=5.0)


# Backward compatibility alias
ProcessManager = TrainingManager

# Public API exports
__all__ = [
    # Main interface
    "TrainingManager",
    "ProcessManager",  # Backward compatibility
    # Core components
    "CoreProcessManager",
    "ProcessMonitor",
    "ProcessAbortManager",
    "HydraOverrideManager",
    "LogStreamingIntegrator",
    # State types
    "ProcessInfo",
    "ProcessState",
    "AbortLevel",
    "AbortProgress",
    "AbortResult",
    "AbortCallback",
    # Exceptions
    "TrainingProcessError",
]
