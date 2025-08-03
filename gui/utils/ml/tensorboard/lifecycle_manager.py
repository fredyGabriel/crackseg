"""TensorBoard lifecycle management and automation.

This module provides automated lifecycle management for TensorBoard instances,
including integration with training processes, session management, and
automatic startup/shutdown based on training state changes.

Key Features:
- Automatic TensorBoard startup when training begins
- Automatic shutdown on training completion or app exit
- Integration with training session managers
- Configurable startup delays and automation settings
- Thread-safe state management and callbacks

Example:
    >>> from pathlib import Path
    >>> lifecycle_manager = TensorBoardLifecycleManager()
    >>> lifecycle_manager.handle_training_state_change(
    ...     "starting",
    ...     log_dir=Path("outputs/logs/tensorboard")
    ... )
"""

import atexit
import threading
import time
from pathlib import Path
from typing import Any

from .core import TensorBoardError, validate_log_directory
from .process_manager import TensorBoardManager, create_tensorboard_manager


class TensorBoardLifecycleManager:
    """Automated TensorBoard lifecycle management.

    Manages TensorBoard startup and shutdown in response to training lifecycle
    events. Provides integration points for training session managers and
    process managers to automatically control TensorBoard instances.

    This manager handles:
    - Automatic startup when training begins
    - Delayed startup to avoid conflicts with training initialization
    - Automatic shutdown on training completion or application exit
    - Thread-safe coordination with other system components

    Attributes:
        manager: TensorBoard process manager instance
        auto_start_on_training: Whether to auto-start TensorBoard on training
        auto_stop_on_training_complete: Whether to auto-stop on training end
        auto_stop_on_app_exit: Whether to auto-stop on application exit
        startup_delay: Delay before starting TensorBoard (seconds)
    """

    def __init__(
        self,
        manager: TensorBoardManager | None = None,
        auto_start_on_training: bool = True,
        auto_stop_on_training_complete: bool = True,
        auto_stop_on_app_exit: bool = True,
        startup_delay: float = 5.0,
    ) -> None:
        """Initialize TensorBoard lifecycle manager.

        Args:
            manager: TensorBoard manager instance (creates default if None)
            auto_start_on_training: Enable automatic startup on training
            auto_stop_on_training_complete: Enable automatic stop on training
                end
            auto_stop_on_app_exit: Enable automatic stop on application exit
            startup_delay: Seconds to wait before starting TensorBoard
        """
        self._manager = manager or create_tensorboard_manager()
        self._auto_start_on_training = auto_start_on_training
        self._auto_stop_on_training_complete = auto_stop_on_training_complete
        self._auto_stop_on_app_exit = auto_stop_on_app_exit
        self._startup_delay = startup_delay

        # State tracking
        self._lock = threading.Lock()
        self._startup_timer: threading.Timer | None = None
        self._training_active = False

        # Register cleanup on exit if enabled
        if self._auto_stop_on_app_exit:
            atexit.register(self._cleanup_on_exit)

    def register_with_session_manager(self, session_manager: object) -> None:
        """Register with training session manager for state updates.

        Args:
            session_manager: Session manager instance to register with

        Note:
            This is a placeholder for future integration. The session manager
            should call handle_training_state_change() on relevant events.
        """
        # Future implementation: register callbacks with session manager
        # session_manager.add_callback(self.handle_training_state_change)
        pass

    def register_with_process_manager(self, process_manager: object) -> None:
        """Register with process manager for training events.

        Args:
            process_manager: Process manager instance to register with

        Note:
            This is a placeholder for future integration. The process manager
            should call handle_training_state_change() and
            handle_log_directory_available() on relevant events.
        """
        # Future implementation: register callbacks with process manager
        # process_manager.add_training_callback(
        #     self.handle_training_state_change
        # )
        # process_manager.add_log_callback(self.handle_log_directory_available)
        pass

    def handle_training_state_change(
        self,
        training_state: str,
        log_dir: Path | None = None,
        force: bool = False,
    ) -> bool:
        """Handle training state changes with automatic TensorBoard management.

        Args:
            training_state: New training state ("starting", "running",
                "completed", etc.)
            log_dir: Log directory for TensorBoard (required for startup)
            force: Force action regardless of auto settings

        Returns:
            True if action was successful, False otherwise

        Example:
            >>> lifecycle.handle_training_state_change(
            ...     "starting",
            ...     log_dir=Path("outputs/experiment_1/logs")
            ... )
        """
        with self._lock:
            if training_state in ("starting", "running", "active"):
                self._training_active = True

                if (self._auto_start_on_training or force) and log_dir:
                    return self._schedule_auto_startup(log_dir)

            elif training_state in (
                "completed",
                "stopped",
                "failed",
                "finished",
            ):
                self._training_active = False

                if self._auto_stop_on_training_complete or force:
                    return self._perform_auto_shutdown()

            return True

    def handle_log_directory_available(self, log_dir: Path) -> bool:
        """Handle log directory becoming available during training.

        Args:
            log_dir: Path to the newly available log directory

        Returns:
            True if TensorBoard was started successfully, False otherwise
        """
        with self._lock:
            if (
                self._training_active
                and self._auto_start_on_training
                and not self._manager.is_running
            ):
                return self._schedule_auto_startup(log_dir)
            return True

    def force_startup(self, log_dir: Path) -> bool:
        """Force TensorBoard startup regardless of automation settings.

        Args:
            log_dir: Directory containing TensorBoard logs

        Returns:
            True if startup successful, False otherwise

        Raises:
            TensorBoardError: If startup fails due to configuration issues
        """
        try:
            validate_log_directory(log_dir)

            with self._lock:
                # Cancel any pending startup
                if self._startup_timer:
                    self._startup_timer.cancel()
                    self._startup_timer = None

            # Start immediately
            return self._manager.start_tensorboard(log_dir, force_restart=True)

        except Exception as e:
            raise TensorBoardError(f"Failed to force startup: {e}") from e

    def force_shutdown(self) -> bool:
        """Force TensorBoard shutdown regardless of automation settings.

        Returns:
            True if shutdown successful, False otherwise
        """
        with self._lock:
            # Cancel any pending startup
            if self._startup_timer:
                self._startup_timer.cancel()
                self._startup_timer = None

        return self._manager.stop_tensorboard()

    def get_manager(self) -> TensorBoardManager:
        """Get the underlying TensorBoard manager.

        Returns:
            TensorBoard manager instance
        """
        return self._manager

    def is_auto_management_active(self) -> bool:
        """Check if automatic management is currently active.

        Returns:
            True if auto-management is enabled and training is active
        """
        with self._lock:
            return self._training_active and (
                self._auto_start_on_training
                or self._auto_stop_on_training_complete
            )

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status information.

        Returns:
            Dictionary containing lifecycle and manager status
        """
        with self._lock:
            manager_info = self._manager.info

            return {
                "lifecycle": {
                    "training_active": self._training_active,
                    "auto_start_enabled": self._auto_start_on_training,
                    "auto_stop_enabled": self._auto_stop_on_training_complete,
                    "startup_delay": self._startup_delay,
                    "pending_startup": self._startup_timer is not None,
                },
                "tensorboard": manager_info.get_status_summary(),
            }

    def _schedule_auto_startup(self, log_dir: Path) -> bool:
        """Schedule automatic TensorBoard startup with delay.

        Args:
            log_dir: Directory containing TensorBoard logs

        Returns:
            True if startup was scheduled, False if already running or failed
        """
        if self._manager.is_running:
            return True

        try:
            validate_log_directory(log_dir)
        except Exception:
            return False

        # Cancel any existing startup timer
        if self._startup_timer:
            self._startup_timer.cancel()

        def delayed_startup() -> None:
            """Delayed startup function for timer."""
            try:
                with self._lock:
                    if not self._training_active:
                        return  # Training stopped while waiting

                    if self._manager.is_running:
                        return  # Already started by someone else

                success = self._manager.start_tensorboard(log_dir)
                if not success:
                    # Retry once after a short delay
                    time.sleep(2.0)
                    self._manager.start_tensorboard(log_dir)

            except Exception:
                # Ignore startup errors in background thread
                pass
            finally:
                with self._lock:
                    self._startup_timer = None

        # Schedule startup with delay
        self._startup_timer = threading.Timer(
            self._startup_delay, delayed_startup
        )
        self._startup_timer.daemon = True
        self._startup_timer.start()

        return True

    def _perform_auto_shutdown(self) -> bool:
        """Perform automatic TensorBoard shutdown.

        Returns:
            True if shutdown was successful, False otherwise
        """
        # Cancel any pending startup
        if self._startup_timer:
            self._startup_timer.cancel()
            self._startup_timer = None

        if not self._manager.is_running:
            return True

        return self._manager.stop_tensorboard()

    def _cleanup_on_exit(self) -> None:
        """Cleanup TensorBoard on application exit."""
        try:
            if self._manager.is_running:
                self._manager.stop_tensorboard(timeout=5.0)
        except Exception:
            # Ignore errors during exit cleanup
            pass

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            with self._lock:
                if self._startup_timer:
                    self._startup_timer.cancel()

            if self._manager.is_running:
                self._manager.stop_tensorboard()
        except Exception:
            # Ignore errors during cleanup
            pass


# Global lifecycle manager instance
_global_lifecycle_manager: TensorBoardLifecycleManager | None = None


def get_global_lifecycle_manager() -> TensorBoardLifecycleManager:
    """Get or create the global TensorBoard lifecycle manager.

    Returns:
        Singleton TensorBoardLifecycleManager instance
    """
    global _global_lifecycle_manager
    if _global_lifecycle_manager is None:
        _global_lifecycle_manager = TensorBoardLifecycleManager()
    return _global_lifecycle_manager


def initialize_tensorboard_lifecycle() -> TensorBoardLifecycleManager:
    """Initialize and configure TensorBoard lifecycle management.

    Returns:
        Configured TensorBoardLifecycleManager instance
    """
    return TensorBoardLifecycleManager(
        auto_start_on_training=True,
        auto_stop_on_training_complete=True,
        auto_stop_on_app_exit=True,
        startup_delay=5.0,
    )


def cleanup_tensorboard_lifecycle() -> None:
    """Clean up global TensorBoard lifecycle manager."""
    global _global_lifecycle_manager
    if _global_lifecycle_manager:
        _global_lifecycle_manager.force_shutdown()
        _global_lifecycle_manager = None
