"""Thread coordination system for CrackSeg GUI operations.

This module provides centralized thread management and coordination
for maintaining UI responsiveness during training operations.
"""

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskPriority(Enum):
    """Priority levels for background tasks."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreadTask:
    """Represents a task to be executed in a background thread.

    Attributes:
        func: Function to execute
        args: Positional arguments for function
        kwargs: Keyword arguments for function
        priority: Task priority level
        name: Human-readable task name
        timeout: Maximum execution time in seconds
        created_at: Task creation timestamp
    """

    func: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    name: str = "UnnamedTask"
    timeout: float | None = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate task parameters."""
        if not callable(self.func):
            raise ValueError("Task function must be callable")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive")


class ThreadCoordinator:
    """Centralized thread coordination for UI responsiveness.

    Manages multiple thread pools and coordinates task execution
    to maintain UI responsiveness during long-running operations.

    Features:
    - Priority-based task scheduling
    - Configurable thread pools for different task types
    - Graceful shutdown with timeout handling
    - Thread-safe operations with proper synchronization
    - Resource monitoring and cleanup

    Example:
        >>> coordinator = ThreadCoordinator(max_workers=4)
        >>> task = ThreadTask(
        ...     func=expensive_operation,
        ...     args=(data,),
        ...     priority=TaskPriority.HIGH,
        ...     name="DataProcessing"
        ... )
        >>> future = coordinator.submit_task(task)
        >>> result = future.result(timeout=30.0)
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_io_workers: int = 2,
        enable_monitoring: bool = True,
    ) -> None:
        """Initialize the thread coordinator.

        Args:
            max_workers: Maximum number of worker threads for CPU tasks
            max_io_workers: Maximum number of threads for I/O operations
            enable_monitoring: Whether to enable thread monitoring
        """
        self._max_workers = max_workers
        self._max_io_workers = max_io_workers
        self._enable_monitoring = enable_monitoring

        # Thread pools for different task types
        self._cpu_executor: ThreadPoolExecutor | None = None
        self._io_executor: ThreadPoolExecutor | None = None

        # Task tracking
        self._active_tasks: dict[str, Future[Any]] = {}
        self._completed_tasks: list[str] = []
        self._failed_tasks: list[tuple[str, Exception]] = []

        # Synchronization
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Monitoring
        self._monitor_thread: threading.Thread | None = None
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
        }

        # Initialize thread pools
        self._initialize_executors()

        if self._enable_monitoring:
            self._start_monitoring()

    def _initialize_executors(self) -> None:
        """Initialize thread pool executors."""
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="CrackSeg-CPU"
        )
        self._io_executor = ThreadPoolExecutor(
            max_workers=self._max_io_workers, thread_name_prefix="CrackSeg-IO"
        )

    def submit_task(
        self, task: ThreadTask, use_io_pool: bool = False
    ) -> Future[Any]:
        """Submit a task for background execution.

        Args:
            task: Task to execute
            use_io_pool: Whether to use I/O thread pool instead of CPU pool

        Returns:
            Future object for tracking task completion

        Raises:
            RuntimeError: If coordinator is shut down
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("ThreadCoordinator is shut down")

        executor = self._io_executor if use_io_pool else self._cpu_executor
        if executor is None:
            raise RuntimeError("Thread executor not initialized")

        # Wrap task execution for monitoring
        def wrapped_execution() -> Any:
            start_time = time.time()
            try:
                result = task.func(*task.args, **task.kwargs)

                with self._lock:
                    self._stats["tasks_completed"] += 1
                    self._stats["total_execution_time"] += (
                        time.time() - start_time
                    )
                    self._completed_tasks.append(task.name)

                return result

            except Exception as e:
                with self._lock:
                    self._stats["tasks_failed"] += 1
                    self._failed_tasks.append((task.name, e))
                raise

        # Submit task to appropriate executor
        future = executor.submit(wrapped_execution)

        with self._lock:
            self._stats["tasks_submitted"] += 1
            self._active_tasks[task.name] = future

        # Add completion callback to clean up tracking
        def cleanup_callback(fut: Future[Any]) -> None:
            with self._lock:
                self._active_tasks.pop(task.name, None)

        future.add_done_callback(cleanup_callback)
        return future

    def submit_function(
        self,
        func: Callable[..., Any],
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        name: str = "AnonymousTask",
        timeout: float | None = None,
        use_io_pool: bool = False,
        **kwargs: Any,
    ) -> Future[Any]:
        """Convenience method to submit a function as a task.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            priority: Task priority
            name: Task name for tracking
            timeout: Maximum execution time
            use_io_pool: Whether to use I/O thread pool
            **kwargs: Keyword arguments for function

        Returns:
            Future object for tracking task completion
        """
        task = ThreadTask(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            name=name,
            timeout=timeout,
        )
        return self.submit_task(task, use_io_pool=use_io_pool)

    def cancel_task(self, task_name: str) -> bool:
        """Cancel a running task by name.

        Args:
            task_name: Name of task to cancel

        Returns:
            True if task was cancelled, False if not found or already done
        """
        with self._lock:
            future = self._active_tasks.get(task_name)
            if future is not None:
                return future.cancel()
            return False

    def wait_for_tasks(
        self,
        task_names: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Wait for specific tasks or all active tasks to complete.

        Args:
            task_names: Specific task names to wait for (None = all active)
            timeout: Maximum time to wait

        Returns:
            Dictionary mapping task names to their results
        """
        with self._lock:
            if task_names is None:
                futures_to_wait = dict(self._active_tasks)
            else:
                futures_to_wait = {
                    name: future
                    for name, future in self._active_tasks.items()
                    if name in task_names
                }

        results = {}
        start_time = time.time()

        for name, future in futures_to_wait.items():
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
                if remaining_timeout <= 0:
                    break

            try:
                results[name] = future.result(timeout=remaining_timeout)
            except Exception as e:
                results[name] = e

        return results

    def get_active_tasks(self) -> list[str]:
        """Get list of currently active task names."""
        with self._lock:
            return list(self._active_tasks.keys())

    def get_statistics(self) -> dict[str, Any]:
        """Get coordinator statistics.

        Returns:
            Dictionary with task execution statistics
        """
        with self._lock:
            stats = self._stats.copy()
            additional_stats = {
                "active_tasks": len(self._active_tasks),
                "recent_completed": self._completed_tasks[-10:],
                "recent_failed": self._failed_tasks[-5:],
                "average_execution_time": (
                    self._stats["total_execution_time"]
                    / max(1, self._stats["tasks_completed"])
                ),
            }
            return {**stats, **additional_stats}

    def _start_monitoring(self) -> None:
        """Start the monitoring thread."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_tasks,
            name="ThreadCoordinator-Monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_tasks(self) -> None:
        """Monitor thread health and task execution."""
        while not self._shutdown_event.wait(timeout=5.0):
            try:
                # Check for stuck tasks (running longer than expected)
                current_time = time.time()
                with self._lock:
                    stuck_tasks = []
                    for name, future in self._active_tasks.items():
                        # Heuristic: tasks running > 5 min might be stuck
                        if (
                            not future.done()
                            and current_time - time.time() > 300
                        ):
                            stuck_tasks.append(name)

                # Log warnings for stuck tasks (use proper logging in prod)
                if stuck_tasks:
                    print(f"Warning: Potentially stuck tasks: {stuck_tasks}")

            except Exception:
                # Monitoring should not crash the coordinator
                pass

    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Shutdown the thread coordinator gracefully.

        Args:
            wait: Whether to wait for active tasks to complete
            timeout: Maximum time to wait for shutdown
        """
        self._shutdown_event.set()

        # Stop monitoring thread
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        # Shutdown executors
        if self._cpu_executor:
            self._cpu_executor.shutdown(wait=wait)
        if self._io_executor:
            self._io_executor.shutdown(wait=wait)

    def __enter__(self) -> "ThreadCoordinator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.shutdown()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.shutdown(wait=False, timeout=5.0)
        except Exception:
            pass  # Best effort cleanup
