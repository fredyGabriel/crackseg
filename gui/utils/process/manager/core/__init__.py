"""Core process management components.

This package contains the core process management functionality including
the main ProcessManager, states, error handling, and core utilities.
"""

from .process_manager import ProcessManager, TrainingProcessError
from .states import (
    AbortCallback,
    AbortLevel,
    AbortProgress,
    AbortResult,
    ProcessInfo,
    ProcessState,
)

__all__ = [
    "ProcessManager",
    "TrainingProcessError",
    "AbortCallback",
    "AbortLevel",
    "AbortProgress",
    "AbortResult",
    "ProcessInfo",
    "ProcessState",
]
