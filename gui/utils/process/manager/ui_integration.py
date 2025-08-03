"""
UI responsive functions and threading coordination. This module
provides UI responsiveness functions that wrap training operations
with asynchronous execution, progress tracking, and cancellation
support to prevent GUI blocking.
"""

# pyright: reportInvalidTypeForm=false

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from ..threading import (
    BackgroundTaskResult,
    CancellationToken,
    ProgressCallback,
    TaskPriority,
    ThreadCoordinator,
    UIResponsiveWrapper,
)
from .orchestrator import start_training_session

# Global instances for singleton pattern
_global_ui_wrapper: UIResponsiveWrapper | None = None
_global_thread_coordinator: ThreadCoordinator | None = None


def get_ui_wrapper() -> UIResponsiveWrapper:
    """
    Get or create the global UI responsive wrapper instance. Provides
    singleton access to the UI wrapper to ensure consistent threading
    behavior across the GUI. Returns: Global UIResponsiveWrapper instance
    """
    global _global_ui_wrapper, _global_thread_coordinator

    if _global_ui_wrapper is None:
        # Create shared thread coordinator if needed
        if _global_thread_coordinator is None:
            _global_thread_coordinator = ThreadCoordinator(
                max_workers=4, max_io_workers=2, enable_monitoring=True
            )

        _global_ui_wrapper = UIResponsiveWrapper(
            coordinator=_global_thread_coordinator,  # type: ignore[arg-type]
            default_timeout=300.0,
        )

    return _global_ui_wrapper  # type: ignore[return-value]


def cleanup_ui_wrapper() -> None:
    """
    Clean up and reset the global UI wrapper and thread coordinator.
    Should be called when the GUI is shutting down or when a clean reset
    is needed. Cancels all active operations.
    """
    global _global_ui_wrapper, _global_thread_coordinator

    if _global_ui_wrapper is not None:
        _global_ui_wrapper.shutdown(timeout=30.0)  # type: ignore[misc]
        _global_ui_wrapper = None

    if _global_thread_coordinator is not None:
        _global_thread_coordinator.shutdown(timeout=30.0)  # type: ignore[misc]
        _global_thread_coordinator = None


def execute_training_async(
    config_path: Path,
    config_name: str,
    overrides_text: str = "",
    working_dir: Path | None = None,
    validate_overrides: bool = True,
    task_name: str = "TrainingExecution",
    priority: TaskPriority = TaskPriority.HIGH,
) -> tuple[Future[tuple[bool, list[str]]], CancellationToken]:
    """Execute training asynchronously without blocking the UI.

    High-level function that wraps start_training_session for
    asynchronous execution with cancellation support.

    Args:
        config_path: Path to configuration directory
        config_name: Configuration file name (without .yaml)
        overrides_text: Raw override text from GUI input
        working_dir: Working directory for execution
        validate_overrides: Whether to validate override types
        task_name: Name for task tracking
        priority: Task priority level

    Returns:
        Tuple of (Future for result, CancellationToken for cancellation)

    Example:
        >>> future, token = execute_training_async(
        ...     Path("configs"),
        ...     "train_baseline",
        ...     "trainer.max_epochs=50"
        ... )
        >>> # Later, to cancel:
        >>> token.cancel("User requested stop")
        >>> # To get result:
        >>> success, errors = future.result(timeout=30.0)
    """
    wrapper = get_ui_wrapper()  # type: ignore[misc]

    return wrapper.execute_async(  # type: ignore[misc]
        func=start_training_session,
        args=(
            config_path,
            config_name,
            overrides_text,
            working_dir,
            validate_overrides,
        ),
        task_name=task_name,
        priority=priority,
    )


def execute_with_progress(
    func: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    task_name: str = "BackgroundOperation",
    timeout: float | None = None,
    priority: TaskPriority = TaskPriority.NORMAL,
) -> BackgroundTaskResult[Any]:
    """Execute any function with progress tracking and UI responsiveness.

    Generic wrapper for executing long-running operations in the background
    while maintaining UI responsiveness through progress callbacks.

    Args:
        func: Function to execute
        args: Positional arguments for function
        kwargs: Keyword arguments for function
        progress_callback: Callback for progress updates
        task_name: Name for task tracking
        timeout: Maximum execution time
        priority: Task priority level

    Returns:
        BackgroundTaskResult with execution details

    Example:
        >>> def progress_handler(update: ProgressUpdate) -> None:
        ...     print(f"Progress: {update.percentage:.1f}% - {update.message}")
        >>>
        >>> result = execute_with_progress(
        ...     func=expensive_data_processing,
        ...     args=(large_dataset,),
        ...     progress_callback=progress_handler,
        ...     task_name="DataProcessing"
        ... )
        >>>
        >>> if result.is_successful:
        ...     print(f"Result: {result.result}")
        >>> else:
        ...     print(f"Error: {result.error}")
    """
    wrapper = get_ui_wrapper()  # type: ignore[misc]

    return wrapper.execute_with_progress(  # type: ignore[misc]
        func=func,
        args=args,
        kwargs=kwargs,
        progress_callback=progress_callback,
        task_name=task_name,
        timeout=timeout,
        priority=priority,
    )
