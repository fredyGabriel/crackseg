"""Configuration classes for coverage monitoring system.

This module contains the data classes and configuration objects used
by the coverage monitoring system, providing validation and type safety.
"""

from dataclasses import dataclass


@dataclass
class CoverageMetrics:
    """Data class for coverage metrics with validation."""

    timestamp: str
    overall_coverage: float
    total_statements: int
    covered_statements: int
    missing_statements: int
    modules_count: int
    modules_above_threshold: int
    critical_gaps: int
    branch_coverage: float | None = None
    test_count: int | None = None
    execution_time: float | None = None

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if not 0.0 <= self.overall_coverage <= 100.0:
            raise ValueError(f"Invalid coverage: {self.overall_coverage}")
        if self.total_statements <= 0:
            raise ValueError(
                f"Invalid total statements: {self.total_statements}"
            )


@dataclass
class AlertConfig:
    """Configuration for coverage alerts."""

    enabled: bool = True
    email_recipients: list[str] | None = None
    slack_webhook: str | None = None
    threshold_warning: float = 75.0
    threshold_critical: float = 70.0
    trend_alert_days: int = 7
    trend_decline_threshold: float = 5.0

    def __post_init__(self) -> None:
        """Set default email recipients if none provided."""
        if self.email_recipients is None:
            self.email_recipients = []


@dataclass
class CoverageMonitorConfig:
    """Configuration for the coverage monitor."""

    target_threshold: float = 80.0
    output_dir: str = "outputs/coverage_monitoring"
    db_path: str | None = None
    alert_config: AlertConfig | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.0 <= self.target_threshold <= 100.0:
            raise ValueError(
                f"Invalid target threshold: {self.target_threshold}"
            )
        if self.alert_config is None:
            self.alert_config = AlertConfig()
