"""Secure subprocess management for training execution.

This module provides robust subprocess management for executing training
runs with process control, monitoring, and resource cleanup. Designed
for long-running training processes with real-time monitoring capabilities.
"""

# Import the refactored ProcessManager from the new module structure
from .core import ProcessManager, TrainingProcessError

# Re-export for backward compatibility
__all__ = ["ProcessManager", "TrainingProcessError"]
