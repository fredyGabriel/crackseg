"""Coverage monitoring system for CrackSeg project.

This module provides comprehensive coverage monitoring capabilities including
real-time analysis, historical trend tracking, and automated alerting.
"""

from .config import AlertConfig, CoverageMetrics, CoverageMonitorConfig
from .core import CoverageMonitor

__all__ = [
    "CoverageMonitor",
    "CoverageMetrics",
    "AlertConfig",
    "CoverageMonitorConfig",
]
