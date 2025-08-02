"""Process state definitions for CrackSeg training execution.

This module defines process states, process information structures,
and state management utilities for training subprocess monitoring.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TrainingProcessError(Exception):
    """Custom exception for training process errors.

    Raised when training subprocess management fails due to:
    - Process already running when starting new training
    - Invalid command construction
    - Working directory doesn't exist
    - Process termination failures
    - Override validation errors

    Examples:
        >>> raise TrainingProcessError("Training process is already running")
        >>> raise TrainingProcessError("Invalid overrides detected: key=value")
    """

    pass


class ProcessState(Enum):
    """Training process execution states."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class AbortLevel(Enum):
    """Different levels of process abort intensity."""

    GRACEFUL = "graceful"  # Send SIGTERM/CTRL_BREAK, wait for timeout
    FORCE = "force"  # Send SIGKILL immediately
    NUCLEAR = "nuclear"  # Kill process tree + cleanup orphans


@dataclass
class AbortProgress:
    """Progress information during abort operation."""

    stage: str  # Current abort stage
    message: str  # Human-readable progress message
    progress_percent: float  # Progress as percentage (0-100)
    elapsed_time: float  # Time since abort started
    estimated_remaining: float | None = None  # Estimated time remaining


@dataclass
class AbortResult:
    """Result of an abort operation with detailed information."""

    success: bool  # Whether abort completed successfully
    abort_level_used: AbortLevel  # Level that was actually used
    process_killed: bool  # Whether main process was terminated
    children_killed: int  # Number of child processes terminated
    zombies_cleaned: int  # Number of zombie processes cleaned
    total_time: float  # Total time for abort operation
    error_message: str | None = None  # Error message if abort failed
    warnings: list[str] = field(default_factory=list)  # Non-fatal warnings


# Type alias for abort progress callbacks
type AbortCallback = Callable[[AbortProgress], None]


@dataclass
class ProcessInfo:
    """Information about a running training process."""

    pid: int | None = None
    command: list[str] = field(default_factory=list)
    start_time: float | None = None
    state: ProcessState = ProcessState.IDLE
    return_code: int | None = None
    error_message: str | None = None
    working_directory: Path | None = None
