"""Performance reporting module with dashboards and trend analysis.

This module provides automated performance reporting capabilities including:
- Interactive HTML dashboards with Plotly visualizations
- Trend analysis with confidence scoring and historical comparison
- Regression detection and alerting integration
- Multiple export formats (HTML, JSON, PDF)
- Actionable insights and performance recommendations
- CI/CD pipeline integration for automated reporting

Example Usage:
    >>> from tests.e2e.performance.reporting import (
    ...     PerformanceReportGenerator,
    ...     ReportConfiguration,
    ...     create_performance_report_generator,
    ...     generate_ci_performance_report
    ... )
    >>>
    >>> # Create report generator with custom configuration
    >>> config = ReportConfiguration(
    ...     export_formats=["html", "json"],
    ...     include_trend_analysis=True,
    ...     chart_theme="plotly_dark"
    ... )
    >>> generator = PerformanceReportGenerator(
    ...     storage_path="reports/",
    ...     config=config
    ... )
    >>>
    >>> # Generate comprehensive report from benchmark results
    >>> results = {"benchmark_results": {...}}
    >>> output_files = generator.generate_comprehensive_report(
    ...     results, commit_sha="abc123"
    ... )
    >>>
    >>> # Quick CI/CD integration
    >>> files = generate_ci_performance_report(
    ...     results_path="performance-results.json",
    ...     output_dir="ci-reports/"
    ... )
"""

from __future__ import annotations

# Specialized components for advanced usage
from tests.e2e.performance.reporting.analysis import PerformanceAnalyzer

# Configuration and type definitions
from tests.e2e.performance.reporting.config import (
    CHART_THEMES,
    DEFAULT_EXPORT_FORMATS,
    RISK_THRESHOLDS,
    ReportConfiguration,
    ReportMetrics,
    TrendData,
)

# Core reporting classes
from tests.e2e.performance.reporting.core import PerformanceReportGenerator

# Factory functions
from tests.e2e.performance.reporting.factory_functions import (
    create_performance_report_generator,
    generate_ci_performance_report,
)
from tests.e2e.performance.reporting.formats import ReportFormatter
from tests.e2e.performance.reporting.visualizations import (
    PerformanceVisualizer,
)

__all__ = [
    # Main interfaces
    "PerformanceReportGenerator",
    "create_performance_report_generator",
    "generate_ci_performance_report",
    # Configuration
    "ReportConfiguration",
    "ReportMetrics",
    "TrendData",
    "DEFAULT_EXPORT_FORMATS",
    "CHART_THEMES",
    "RISK_THRESHOLDS",
    # Specialized components
    "PerformanceAnalyzer",
    "ReportFormatter",
    "PerformanceVisualizer",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "CrackSeg Performance Team"
__description__ = (
    "Comprehensive performance reporting with dashboards and trend analysis"
)
