"""Log source implementations for different streaming scenarios.

This package provides specific log source classes for stdout streaming
and Hydra log file watching with cross-platform compatibility.

The package is organized by source type for better maintainability:
- stdout_reader: OutputStreamReader for subprocess stdout/stderr
- file_watcher: HydraLogWatcher for Hydra log file monitoring
"""

from .file_watcher import HydraLogWatcher
from .stdout_reader import OutputStreamReader

__all__ = [
    "OutputStreamReader",
    "HydraLogWatcher",
]
