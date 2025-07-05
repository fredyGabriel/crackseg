"""Type definitions and models for the test reporting system.

This module contains all data models, enums, and type definitions used
across the reporting framework components.
"""

from enum import Enum
from typing import Any, TypedDict

__all__ = [
    "ReportMode",
    "ReportFormat",
    "TestStatus",
    "TestResult",
    "ExecutionSummary",
]


class ReportMode(Enum):
    """Report generation modes."""

    BASIC = "basic"  # Essential pass/fail statistics
    COMPREHENSIVE = "comprehensive"  # Full metrics and analysis
    PERFORMANCE_FOCUSED = "performance"  # Performance metrics only
    FAILURE_ANALYSIS = "failure_analysis"  # Failure patterns and insights
    DASHBOARD = "dashboard"  # Visual dashboard with charts


class ReportFormat(Enum):
    """Output formats for reports."""

    HTML = "html"
    JSON = "json"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JUNIT_XML = "junit_xml"


class TestStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    XFAIL = "xfail"
    XPASS = "xpass"


class TestResult(TypedDict):
    """Type definition for individual test results."""

    test_id: str
    test_name: str
    status: str
    duration: float
    start_time: float
    end_time: float
    error_message: str | None
    failure_reason: str | None
    performance_data: dict[str, Any] | None
    artifacts: list[str]  # Paths to screenshots, videos, etc.


class ExecutionSummary(TypedDict):
    """Type definition for test execution summary."""

    total_tests: int
    passed: int
    failed: int
    skipped: int
    error: int
    success_rate: float
    total_duration: float
    start_time: str
    end_time: str
