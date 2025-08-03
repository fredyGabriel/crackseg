"""Task result management for UI responsive threading.

This module provides data structures and utilities for handling the results
of background task execution, including success/failure status, timing,
and progress history.
"""

from dataclasses import dataclass
from typing import Any, TypeVar

from .progress_tracking import ProgressUpdate
from .task_status import TaskStatus

T = TypeVar("T")


@dataclass
class BackgroundTaskResult[T]:
    """Result of a background task execution.

    Comprehensive result container that captures all aspects of task execution
    including status, results, errors, timing, and progress history.

    Attributes:
        status: Final status of the task
        result: Task result (if successful)
        error: Exception that occurred (if failed)
        execution_time: Total execution time in seconds
        progress_updates: List of progress updates received
        cancellation_reason: Reason for cancellation (if cancelled)

    Example:
        >>> result = BackgroundTaskResult(
        ...     status=TaskStatus.COMPLETED,
        ...     result="Task completed successfully",
        ...     execution_time=2.5
        ... )
        >>> if result.is_successful:
        ...     print(f"Success: {result.result}")
    """

    status: TaskStatus
    result: T | None = None
    error: Exception | None = None
    execution_time: float = 0.0
    progress_updates: list[ProgressUpdate] | None = None
    cancellation_reason: str | None = None

    def __post_init__(self) -> None:
        """Initialize empty progress updates list if not provided."""
        if self.progress_updates is None:
            self.progress_updates = []

    @property
    def is_successful(self) -> bool:
        """Check if task completed successfully.

        Returns:
            True if task completed without errors or cancellation
        """
        return self.status == TaskStatus.COMPLETED and self.error is None

    @property
    def is_failed(self) -> bool:
        """Check if task failed with an error.

        Returns:
            True if task status is FAILED or error is present
        """
        return self.status == TaskStatus.FAILED or self.error is not None

    @property
    def is_cancelled(self) -> bool:
        """Check if task was cancelled.

        Returns:
            True if task was cancelled before completion
        """
        return self.status == TaskStatus.CANCELLED

    @property
    def is_terminal(self) -> bool:
        """Check if task has reached a terminal state.

        Returns:
            True if task is completed, failed, or cancelled
        """
        return self.status.is_terminal()

    @property
    def final_progress(self) -> ProgressUpdate | None:
        """Get the final progress update.

        Returns:
            Last progress update or None if no updates were recorded
        """
        return self.progress_updates[-1] if self.progress_updates else None

    @property
    def progress_count(self) -> int:
        """Get the number of progress updates recorded.

        Returns:
            Count of progress updates
        """
        return len(self.progress_updates) if self.progress_updates else 0

    @property
    def error_message(self) -> str | None:
        """Get human-readable error message.

        Returns:
            Error message string or None if no error
        """
        if self.error:
            return str(self.error)
        elif self.is_cancelled and self.cancellation_reason:
            return f"Cancelled: {self.cancellation_reason}"
        return None

    def add_progress_update(self, update: ProgressUpdate) -> None:
        """Add a progress update to the history.

        Args:
            update: Progress update to add
        """
        if self.progress_updates is None:
            self.progress_updates = []
        self.progress_updates.append(update)

    def get_progress_summary(self) -> dict[str, Any]:
        """Get summary of progress updates.

        Returns:
            Dictionary with progress statistics
        """
        if not self.progress_updates:
            return {"total_updates": 0, "final_percentage": 0.0}

        final_update = self.progress_updates[-1]
        return {
            "total_updates": len(self.progress_updates),
            "final_percentage": final_update.percentage,
            "final_message": final_update.message,
            "final_stage": final_update.stage,
            "stages": list({update.stage for update in self.progress_updates}),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation.

        Returns:
            Dictionary representation of the task result
        """
        return {
            "status": self.status.value,
            "is_successful": self.is_successful,
            "is_failed": self.is_failed,
            "is_cancelled": self.is_cancelled,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "cancellation_reason": self.cancellation_reason,
            "progress_summary": self.get_progress_summary(),
        }

    @classmethod
    def success(
        cls,
        result: T,
        execution_time: float = 0.0,
        progress_updates: list[ProgressUpdate] | None = None,
    ) -> "BackgroundTaskResult[T]":
        """Create a successful result.

        Args:
            result: Task result value
            execution_time: Task execution time
            progress_updates: Progress update history

        Returns:
            BackgroundTaskResult indicating success
        """
        return cls(
            status=TaskStatus.COMPLETED,
            result=result,
            execution_time=execution_time,
            progress_updates=progress_updates,
        )

    @classmethod
    def failure(
        cls,
        error: Exception,
        execution_time: float = 0.0,
        progress_updates: list[ProgressUpdate] | None = None,
    ) -> "BackgroundTaskResult[Any]":
        """Create a failed result.

        Args:
            error: Exception that caused failure
            execution_time: Task execution time
            progress_updates: Progress update history

        Returns:
            BackgroundTaskResult indicating failure
        """
        return cls(
            status=TaskStatus.FAILED,
            error=error,
            execution_time=execution_time,
            progress_updates=progress_updates,
        )

    @classmethod
    def cancelled(
        cls,
        reason: str,
        execution_time: float = 0.0,
        progress_updates: list[ProgressUpdate] | None = None,
    ) -> "BackgroundTaskResult[Any]":
        """Create a cancelled result.

        Args:
            reason: Cancellation reason
            execution_time: Task execution time
            progress_updates: Progress update history

        Returns:
            BackgroundTaskResult indicating cancellation
        """
        return cls(
            status=TaskStatus.CANCELLED,
            cancellation_reason=reason,
            execution_time=execution_time,
            progress_updates=progress_updates,
        )


class ResultCollector:
    """Utility for collecting and managing task results.

    Provides a convenient interface for accumulating results from
    multiple background tasks and querying their collective status.

    Example:
        >>> collector = ResultCollector()
        >>> collector.add_result("task1", result1)
        >>> collector.add_result("task2", result2)
        >>> success_rate = collector.success_rate
    """

    def __init__(self) -> None:
        """Initialize result collector."""
        self._results: dict[str, BackgroundTaskResult[Any]] = {}

    def add_result(self, name: str, result: BackgroundTaskResult[Any]) -> None:
        """Add a named result to the collection.

        Args:
            name: Unique name for the result
            result: Task result to add
        """
        self._results[name] = result

    def get_result(self, name: str) -> BackgroundTaskResult[Any] | None:
        """Get a result by name.

        Args:
            name: Result name

        Returns:
            Task result or None if not found
        """
        return self._results.get(name)

    def remove_result(self, name: str) -> bool:
        """Remove a result by name.

        Args:
            name: Result name

        Returns:
            True if result was found and removed
        """
        return self._results.pop(name, None) is not None

    @property
    def result_count(self) -> int:
        """Get total number of results."""
        return len(self._results)

    @property
    def successful_results(self) -> list[BackgroundTaskResult[Any]]:
        """Get list of successful results."""
        return [
            result for result in self._results.values() if result.is_successful
        ]

    @property
    def failed_results(self) -> list[BackgroundTaskResult[Any]]:
        """Get list of failed results."""
        return [
            result for result in self._results.values() if result.is_failed
        ]

    @property
    def cancelled_results(self) -> list[BackgroundTaskResult[Any]]:
        """Get list of cancelled results."""
        return [
            result for result in self._results.values() if result.is_cancelled
        ]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage.

        Returns:
            Success rate (0.0 to 100.0)
        """
        if not self._results:
            return 0.0
        return (len(self.successful_results) / len(self._results)) * 100.0

    @property
    def total_execution_time(self) -> float:
        """Calculate total execution time of all results.

        Returns:
            Total execution time in seconds
        """
        return sum(result.execution_time for result in self._results.values())

    def clear_all(self) -> None:
        """Remove all results from the collection."""
        self._results.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all results.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "total_results": self.result_count,
            "successful": len(self.successful_results),
            "failed": len(self.failed_results),
            "cancelled": len(self.cancelled_results),
            "success_rate": self.success_rate,
            "total_execution_time": self.total_execution_time,
        }
