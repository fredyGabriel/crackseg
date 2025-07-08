"""Factory functions for creating performance report generators.

This module provides factory functions and convenience methods for external
systems to create and use performance report generators.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tests.e2e.performance.reporting.config import ReportConfiguration
from tests.e2e.performance.reporting.core import PerformanceReportGenerator

logger = logging.getLogger(__name__)


def create_performance_report_generator(
    storage_path: Path | str = "performance-reports",
    historical_data_path: Path | str = "performance-historical-data",
    config: ReportConfiguration | None = None,
) -> PerformanceReportGenerator:
    """Factory function to create a performance report generator instance.

    Args:
        storage_path: Directory to store generated reports
        historical_data_path: Directory to store historical performance data
        config: Report configuration (uses defaults if None)

    Returns:
        Configured PerformanceReportGenerator instance
    """
    return PerformanceReportGenerator(
        storage_path=str(storage_path),
        historical_data_path=str(historical_data_path),
        config=config,
    )


def create_default_ci_generator(
    output_dir: Path | str = "performance-reports",
) -> PerformanceReportGenerator:
    """Create a performance report generator with CI/CD defaults.

    Args:
        output_dir: Directory for report outputs

    Returns:
        PerformanceReportGenerator configured for CI/CD use
    """
    config = ReportConfiguration(
        export_formats=["html", "json"],
        include_trend_analysis=True,
        include_regression_analysis=True,
        include_recommendations=True,
        chart_theme="plotly_white",
    )

    return create_performance_report_generator(
        storage_path=output_dir, config=config
    )


def create_local_development_generator(
    output_dir: Path | str = "local-performance-reports",
) -> PerformanceReportGenerator:
    """Create a performance report generator for local development.

    Args:
        output_dir: Directory for report outputs

    Returns:
        PerformanceReportGenerator configured for local development
    """
    config = ReportConfiguration(
        export_formats=["html"],
        include_trend_analysis=True,
        include_regression_analysis=False,  # Skip in development
        include_recommendations=True,
        chart_theme="plotly_dark",  # Better for development
    )

    return create_performance_report_generator(
        storage_path=output_dir, config=config
    )


def generate_ci_performance_report(
    results_path: (
        Path | str
    ) = "performance-gate-results/consolidated-report.json",
    output_dir: Path | str = "performance-reports",
    commit_sha: str | None = None,
) -> dict[str, Path]:
    """Generate performance report from CI/CD pipeline results.

    Args:
        results_path: Path to the performance results JSON file
        output_dir: Directory to store generated reports
        commit_sha: Git commit SHA for tracking (optional)

    Returns:
        Dictionary mapping format names to output file paths

    Raises:
        FileNotFoundError: If results file doesn't exist
        ValueError: If results file cannot be parsed
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    try:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        raise ValueError(
            f"Failed to load results from {results_path}: {e}"
        ) from e

    generator = create_default_ci_generator(output_dir=output_dir)
    return generator.generate_comprehensive_report(results, commit_sha)


def generate_development_report(
    results_data: dict[str, Any],
    output_dir: Path | str = "local-performance-reports",
) -> dict[str, Path]:
    """Generate performance report for local development.

    Args:
        results_data: Performance results dictionary
        output_dir: Directory to store generated reports

    Returns:
        Dictionary mapping format names to output file paths
    """
    generator = create_local_development_generator(output_dir=output_dir)
    return generator.generate_comprehensive_report(results_data)


def validate_results_file(results_path: Path | str) -> dict[str, Any]:
    """Validate a performance results file before processing.

    Args:
        results_path: Path to the results file to validate

    Returns:
        Validation result dictionary with 'is_valid', 'errors', 'warnings'
    """
    results_path = Path(results_path)
    validation: dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
    }

    # Check file existence
    if not results_path.exists():
        validation["is_valid"] = False
        validation["errors"].append(f"File not found: {results_path}")
        return validation

    # Check file format
    if results_path.suffix.lower() != ".json":
        validation["warnings"].append(
            f"Unexpected file extension: {results_path.suffix}"
        )

    # Validate JSON content
    try:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        validation["is_valid"] = False
        validation["errors"].append(f"Invalid JSON: {e}")
        return validation

    # Validate content structure
    if not isinstance(results, dict):
        validation["is_valid"] = False
        validation["errors"].append("Results must be a JSON object")
        return validation

    if "benchmark_results" not in results:
        validation["warnings"].append("No 'benchmark_results' found in file")

    return validation


def get_available_themes() -> list[str]:
    """Get list of available chart themes.

    Returns:
        List of theme names
    """
    return ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]


def get_available_formats() -> list[str]:
    """Get list of available export formats.

    Returns:
        List of format names
    """
    return ["html", "json", "pdf"]


def create_custom_config(
    export_formats: list[str] | None = None,
    chart_theme: str = "plotly_white",
    include_trend_analysis: bool = True,
    include_regression_analysis: bool = True,
    include_recommendations: bool = True,
    time_window_hours: int = 24,
) -> ReportConfiguration:
    """Create a custom report configuration.

    Args:
        export_formats: List of formats to export (default: ["html", "json"])
        chart_theme: Chart theme name
        include_trend_analysis: Whether to include trend analysis
        include_regression_analysis: Whether to include regression analysis
        include_recommendations: Whether to include recommendations
        time_window_hours: Time window for historical data analysis

    Returns:
        Configured ReportConfiguration instance
    """
    if export_formats is None:
        export_formats = ["html", "json"]

    return ReportConfiguration(
        export_formats=export_formats,
        chart_theme=chart_theme,
        include_trend_analysis=include_trend_analysis,
        include_regression_analysis=include_regression_analysis,
        include_recommendations=include_recommendations,
        time_window_hours=time_window_hours,
    )
