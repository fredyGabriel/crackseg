"""
Configuration management for the test reporting system. This module
provides configuration classes and settings for controlling report
generation behavior, output formats, and performance thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path

from tests.e2e.reporting.models import ReportFormat, ReportMode

__all__ = ["ReportConfig"]


@dataclass
class ReportConfig:
    """
    Configuration for test report generation. Attributes: mode: Report
    generation mode formats: Output formats to generate output_dir:
    Directory for generated reports include_performance: Include
    performance metrics include_artifacts: Include screenshots/videos
    include_trends: Include trend analysis performance_thresholds:
    Performance threshold definitions retention_days: Days to keep
    historical reports auto_cleanup: Enable automatic cleanup of old
    reports compress_artifacts: Compress large artifacts generate_summary:
    Generate executive summary export_metrics: Export metrics to external
    systems
    """

    mode: ReportMode = ReportMode.COMPREHENSIVE
    formats: list[ReportFormat] = field(
        default_factory=lambda: [ReportFormat.HTML, ReportFormat.JSON]
    )
    output_dir: Path = field(default_factory=lambda: Path("test-reports"))
    include_performance: bool = True
    include_artifacts: bool = True
    include_trends: bool = True
    performance_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "page_load_max": 3.0,
            "interaction_max": 1.0,
            "memory_max_mb": 1000.0,
        }
    )
    retention_days: int = 30
    auto_cleanup: bool = True
    compress_artifacts: bool = True
    generate_summary: bool = True
    export_metrics: bool = False
