"""Core test reporting functionality - now modular.

This module provides backwards compatibility imports for the refactored
reporting system that has been split into focused, smaller modules.
"""

# Re-export everything from the new modular structure
from tests.e2e.reporting.config import ReportConfig
from tests.e2e.reporting.generator import TestReportGenerator
from tests.e2e.reporting.models import (
    ExecutionSummary,
    ReportFormat,
    ReportMode,
    TestResult,
    TestStatus,
)

__all__ = [
    "TestReportGenerator",
    "ReportConfig",
    "ReportMode",
    "ReportFormat",
    "TestStatus",
    "TestResult",
    "ExecutionSummary",
]
