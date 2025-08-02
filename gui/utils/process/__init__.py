"""Process management system for training execution.

This package provides comprehensive process management capabilities for
training execution including monitoring, logging, cleanup, and override handling.

Structure:
- core/: Core process management and states
- monitoring/: Process monitoring and resource tracking
- logging/: Log streaming and integration
- overrides/: Configuration override handling
- cleanup/: Process cleanup and termination
"""

# Import main components from subpackages
from .cleanup import ProcessCleanup
from .core import (
    AbortCallback,
    AbortLevel,
    AbortProgress,
    AbortResult,
    ProcessInfo,
    ProcessManager,
    ProcessState,
    TrainingProcessError,
)
from .logging import LogStreamer

# Backward compatibility
from .manager_backup import ProcessManager as LegacyProcessManager
from .monitoring import ProcessMonitor
from .overrides import OverrideHandler

__all__ = [
    # Core components
    "ProcessManager",
    "TrainingProcessError",
    "AbortCallback",
    "AbortLevel",
    "AbortProgress",
    "AbortResult",
    "ProcessInfo",
    "ProcessState",
    # Specialized components
    "ProcessMonitor",
    "LogStreamer",
    "OverrideHandler",
    "ProcessCleanup",
    # Legacy compatibility
    "LegacyProcessManager",
]
