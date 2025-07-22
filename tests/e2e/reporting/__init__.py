"""
Comprehensive test reporting framework for E2E testing. This module
provides a configurable reporting system that integrates with existing
performance monitoring, capture systems, and parallel execution
framework to generate comprehensive test reports with metrics, trends,
and actionable insights. Key Features: - Test execution reports with
pass/fail statistics - Performance dashboards with trends and insights
- Failure analysis and classification - CI/CD integration and export
capabilities - Configurable reporting modes and output formats
"""

from tests.e2e.reporting.analysis import FailureAnalyzer, TestTrendAnalyzer
from tests.e2e.reporting.config import ReportConfig
from tests.e2e.reporting.exporters import (
    CICDReportExporter,
    HTMLReportExporter,
    JSONReportExporter,
)
from tests.e2e.reporting.generator import TestReportGenerator
from tests.e2e.reporting.models import ReportFormat, ReportMode

# Note: PerformanceDashboard will be implemented if needed in future
__all__ = [
    "TestReportGenerator",
    "ReportConfig",
    "ReportMode",
    "ReportFormat",
    "FailureAnalyzer",
    "TestTrendAnalyzer",
    "HTMLReportExporter",
    "JSONReportExporter",
    "CICDReportExporter",
]
