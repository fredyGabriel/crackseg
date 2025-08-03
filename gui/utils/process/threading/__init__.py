"""UI responsive threading utilities for CrackSeg GUI operations.

This module provides comprehensive utilities for maintaining UI responsiveness
during long-running operations by offloading work to background threads.

Refactored Architecture:
- task_status: Task state definitions and enumerations
- progress_tracking: Progress reporting and callback mechanisms
- cancellation: Cancellation tokens and management
- task_results: Result containers and collectors
- ui_wrapper: Main UIResponsiveWrapper for background execution

Example:
    >>> from  scripts.gui.utils.threading  import  UIResponsiveWrapper
    >>> wrapper = UIResponsiveWrapper()
    >>> result = wrapper.execute_with_progress(
    ...     func=expensive_function,
    ...     progress_callback=lambda p: print(f"{p.percentage:.1f}%")
    ... )
"""

# Core functionality exports
from .cancellation import (
    CancellationError,
    CancellationManager,
    CancellationToken,
)

# Thread coordination exports
from .coordinator import TaskPriority, ThreadCoordinator, ThreadTask
from .progress_tracking import (
    ProgressCallback,
    ProgressTracker,
    ProgressUpdate,
)
from .task_results import BackgroundTaskResult, ResultCollector

# Task management exports
from .task_status import TaskStatus
from .ui_wrapper import UIResponsiveWrapper

# Backward compatibility aliases (keeping same names from original file)
# These ensure existing code continues to work without changes

# Main classes (no change needed)
UIResponsiveWrapper = UIResponsiveWrapper
TaskStatus = TaskStatus
ProgressUpdate = ProgressUpdate
CancellationToken = CancellationToken
BackgroundTaskResult = BackgroundTaskResult

# Type aliases (no change needed)
ProgressCallback = ProgressCallback

# Additional utility exports for enhanced functionality
ProgressTracker = ProgressTracker
CancellationError = CancellationError
CancellationManager = CancellationManager
ResultCollector = ResultCollector

# Thread coordination aliases
TaskPriority = TaskPriority
ThreadCoordinator = ThreadCoordinator
ThreadTask = ThreadTask

# Public API - all exports available at package level
__all__ = [
    # Core UI responsiveness
    "UIResponsiveWrapper",
    # Task management
    "TaskStatus",
    "BackgroundTaskResult",
    # Progress tracking
    "ProgressUpdate",
    "ProgressCallback",
    "ProgressTracker",
    # Cancellation support
    "CancellationToken",
    "CancellationError",
    "CancellationManager",
    # Result management
    "ResultCollector",
    # Thread coordination
    "TaskPriority",
    "ThreadCoordinator",
    "ThreadTask",
]

# Module metadata
__version__ = "2.0.0"
__author__ = "CrackSeg Development Team"
__description__ = (
    "UI responsive threading utilities for long-running operations"
)
