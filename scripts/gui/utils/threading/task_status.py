"""Task status definitions for UI responsive threading.

This module defines task states and status-related enumerations
for background task execution and UI responsiveness management.
"""

from enum import Enum


class TaskStatus(Enum):
    """Status of a background task.

    Represents the current state of a background task execution,
    allowing UI components to track and respond to task lifecycle events.

    Values:
        PENDING: Task is queued but not yet started
        RUNNING: Task is currently executing
        COMPLETED: Task finished successfully
        FAILED: Task encountered an error during execution
        CANCELLED: Task was cancelled before completion

    Example:
        >>> status = TaskStatus.RUNNING
        >>> if status == TaskStatus.COMPLETED:
        ...     print("Task finished successfully")
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Check if this status represents a terminal state.

        Returns:
            True if the task has finished (success, failure, or cancellation)
        """
        return self in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    def is_active(self) -> bool:
        """Check if this status represents an active state.

        Returns:
            True if the task is currently pending or running
        """
        return self in (TaskStatus.PENDING, TaskStatus.RUNNING)
