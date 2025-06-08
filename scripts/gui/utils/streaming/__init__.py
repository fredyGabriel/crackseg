"""Real-time log streaming system for training processes.

This package provides thread-safe, real-time log streaming capabilities
for training processes with support for multiple log sources and GUI.
"""

from .core import LogStreamManager, StreamedLog
from .exceptions import LogStreamingError
from .sources import HydraLogWatcher, OutputStreamReader

__all__ = [
    "LogStreamManager",
    "StreamedLog",
    "LogStreamingError",
    "HydraLogWatcher",
    "OutputStreamReader",
]
