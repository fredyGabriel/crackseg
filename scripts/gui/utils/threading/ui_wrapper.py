"""UI responsive wrapper for background task execution.

This module provides the main UIResponsiveWrapper class for executing
long-running operations in background threads while maintaining UI
responsiveness.
"""

import inspect
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

from .cancellation import CancellationToken
from .coordinator import TaskPriority, ThreadCoordinator
from .progress_tracking import ProgressCallback, ProgressUpdate
from .task_results import BackgroundTaskResult
from .task_status import TaskStatus

T = TypeVar("T")


class UIResponsiveWrapper:
    """Wrapper for UI-responsive operations through background execution.

    Provides a high-level interface for executing long-running operations
    in background threads while maintaining UI responsiveness through
    progress callbacks and cancellation support.

    Features:
    - Automatic progress tracking and reporting
    - Cancellation support with cleanup
    - Error handling and recovery
    - Thread-safe operation
    - Integration with ThreadCoordinator

    Example:
        >>> wrapper = UIResponsiveWrapper()
        >>>
        >>> def progress_handler(update: ProgressUpdate) -> None:
        ...     print(f"Progress: {update.percentage:.1f}% - {update.message}")
        >>>
        >>> result = wrapper.execute_with_progress(
        ...     func=expensive_computation,
        ...     args=(large_dataset,),
        ...     progress_callback=progress_handler,
        ...     task_name="DataProcessing"
        ... )
    """

    def __init__(
        self,
        coordinator: ThreadCoordinator | None = None,
        default_timeout: float = 300.0,
    ) -> None:
        """Initialize the UI responsive wrapper.

        Args:
            coordinator: Thread coordinator to use (creates new if None)
            default_timeout: Default timeout for operations
        """
        self._coordinator = coordinator or ThreadCoordinator()
        self._default_timeout = default_timeout
        self._owns_coordinator = coordinator is None

        # Active operations tracking
        self._active_operations: dict[str, CancellationToken] = {}
        self._lock = threading.Lock()

    def execute_with_progress(
        self,
        func: Callable[..., T],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
        task_name: str = "BackgroundTask",
        timeout: float | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> BackgroundTaskResult[T]:
        """Execute a function in background with progress tracking.

        Args:
            func: Function to execute
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            progress_callback: Callback for progress updates
            cancellation_token: Token for cancellation support
            task_name: Name for task tracking
            timeout: Maximum execution time
            priority: Task priority level

        Returns:
            BackgroundTaskResult with execution details
        """
        if kwargs is None:
            kwargs = {}

        if timeout is None:
            timeout = self._default_timeout

        if cancellation_token is None:
            cancellation_token = CancellationToken()

        # Track active operation
        with self._lock:
            self._active_operations[task_name] = cancellation_token

        progress_updates: list[ProgressUpdate] = []
        start_time = time.time()

        def progress_reporter(
            current: float,
            total: float,
            message: str,
            stage: str = "Processing",
        ) -> None:
            """Internal progress reporting function."""
            if cancellation_token.is_cancelled:
                return

            update = ProgressUpdate(
                current=current, total=total, message=message, stage=stage
            )
            progress_updates.append(update)

            if progress_callback:
                try:
                    progress_callback(update)
                except Exception:
                    # Don't let callback errors crash the operation
                    pass

        def wrapped_execution() -> T:
            """Wrapped function execution with progress support."""
            try:
                # Add progress reporter to kwargs only if function accepts it
                enhanced_kwargs = kwargs.copy()

                # Check if function accepts these parameters
                sig = inspect.signature(func)

                if "progress_callback" in sig.parameters:
                    enhanced_kwargs["progress_callback"] = progress_reporter
                if "cancellation_token" in sig.parameters:
                    enhanced_kwargs["cancellation_token"] = cancellation_token

                # Execute the function
                result = func(*args, **enhanced_kwargs)

                # Final progress update
                if not cancellation_token.is_cancelled:
                    progress_reporter(
                        100, 100, "Completed successfully", "Finished"
                    )

                return result

            except Exception as e:
                # Report error in progress
                progress_reporter(0, 100, f"Error: {e}", "Failed")
                raise

        try:
            # Submit task to coordinator
            future = self._coordinator.submit_function(
                func=wrapped_execution,
                priority=priority,
                name=task_name,
                timeout=timeout,
            )

            # Wait for completion with timeout
            try:
                result = future.result(timeout=timeout)
                execution_time = time.time() - start_time

                if cancellation_token.is_cancelled:
                    return BackgroundTaskResult(
                        status=TaskStatus.CANCELLED,
                        execution_time=execution_time,
                        progress_updates=progress_updates,
                        cancellation_reason=cancellation_token.cancellation_reason,
                    )
                else:
                    return BackgroundTaskResult(
                        status=TaskStatus.COMPLETED,
                        result=result,
                        execution_time=execution_time,
                        progress_updates=progress_updates,
                    )

            except Exception as e:
                execution_time = time.time() - start_time
                return BackgroundTaskResult(
                    status=TaskStatus.FAILED,
                    error=e,
                    execution_time=execution_time,
                    progress_updates=progress_updates,
                )

        finally:
            # Clean up tracking
            with self._lock:
                self._active_operations.pop(task_name, None)

    def execute_async(
        self,
        func: Callable[..., T],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        task_name: str = "AsyncTask",
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> tuple[Future[T], CancellationToken]:
        """Execute a function asynchronously without blocking.

        Args:
            func: Function to execute
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            task_name: Name for task tracking
            priority: Task priority level

        Returns:
            Tuple of (Future for result, CancellationToken for cancellation)
        """
        if kwargs is None:
            kwargs = {}

        cancellation_token = CancellationToken()

        # Track active operation
        with self._lock:
            self._active_operations[task_name] = cancellation_token

        def wrapped_execution() -> T:
            """Wrapped function execution with cancellation support."""
            try:
                # Add cancellation token to kwargs only if function accepts it
                enhanced_kwargs = kwargs.copy()

                # Check if function accepts cancellation_token parameter
                sig = inspect.signature(func)

                if "cancellation_token" in sig.parameters:
                    enhanced_kwargs["cancellation_token"] = cancellation_token

                return func(*args, **enhanced_kwargs)
            finally:
                # Clean up tracking
                with self._lock:
                    self._active_operations.pop(task_name, None)

        future = self._coordinator.submit_function(
            func=wrapped_execution, priority=priority, name=task_name
        )

        return future, cancellation_token

    def cancel_operation(
        self, task_name: str, reason: str = "User cancellation"
    ) -> bool:
        """Cancel an active operation.

        Args:
            task_name: Name of task to cancel
            reason: Reason for cancellation

        Returns:
            True if operation was found and cancelled
        """
        with self._lock:
            token = self._active_operations.get(task_name)
            if token:
                token.cancel(reason)
                # Also try to cancel the future
                self._coordinator.cancel_task(task_name)
                return True
            return False

    def cancel_all_operations(self, reason: str = "Shutdown requested") -> int:
        """Cancel all active operations.

        Args:
            reason: Reason for cancellation

        Returns:
            Number of operations cancelled
        """
        with self._lock:
            operations = list(self._active_operations.items())

        cancelled_count = 0
        for task_name, token in operations:
            token.cancel(reason)
            self._coordinator.cancel_task(task_name)
            cancelled_count += 1

        return cancelled_count

    def get_active_operations(self) -> list[str]:
        """Get list of active operation names."""
        with self._lock:
            return list(self._active_operations.keys())

    def wait_for_all_operations(self, timeout: float | None = None) -> bool:
        """Wait for all active operations to complete.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if all operations completed, False if timeout
        """
        with self._lock:
            task_names = list(self._active_operations.keys())

        if not task_names:
            return True

        try:
            self._coordinator.wait_for_tasks(task_names, timeout=timeout)
            return True
        except Exception:
            return False

    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the wrapper and clean up resources.

        Args:
            timeout: Maximum time to wait for operations to complete
        """
        # Cancel all operations
        self.cancel_all_operations("Wrapper shutdown")

        # Wait for operations to complete
        self.wait_for_all_operations(timeout=timeout)

        # Shutdown coordinator if we own it
        if self._owns_coordinator:
            self._coordinator.shutdown(timeout=timeout)

    def __enter__(self) -> "UIResponsiveWrapper":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.shutdown()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.shutdown(timeout=5.0)
        except Exception:
            pass  # Best effort cleanup
