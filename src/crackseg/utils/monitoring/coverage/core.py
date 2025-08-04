"""Core coverage monitoring functionality.

This module provides the main CoverageMonitor class that orchestrates
all coverage monitoring components including analysis, alerts, trends,
and reporting.
"""

import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from .alerts import CoverageAlertSystem
from .analysis import CoverageAnalyzer
from .config import AlertConfig, CoverageMetrics
from .reporting import CoverageReporter
from .trends import CoverageTrendAnalyzer

logger = logging.getLogger(__name__)


class CoverageMonitor:
    """Continuous coverage monitoring with historical tracking and alerts.

    This class provides comprehensive coverage monitoring capabilities
    including real-time analysis, historical trend tracking, and automated
    alerting. Designed to integrate seamlessly with existing CI/CD
    infrastructure.
    """

    def __init__(
        self,
        target_threshold: float = 80.0,
        output_dir: Path = Path("artifacts/coverage_monitoring"),
        alert_config: AlertConfig | None = None,
        db_path: Path | None = None,
    ) -> None:
        """Initialize the coverage monitor.

        Args:
            target_threshold: Target coverage percentage for validation
            output_dir: Directory for output reports and data storage
            alert_config: Configuration for automated alerts
            db_path: Path to SQLite database for historical data
        """
        self.target_threshold = target_threshold
        self.output_dir = Path(output_dir)
        self.alert_config = alert_config or AlertConfig()

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "historical").mkdir(exist_ok=True)
        (self.output_dir / "alerts").mkdir(exist_ok=True)

        # Database for historical data
        self.db_path = db_path or (self.output_dir / "coverage_history.db")
        self._init_database()

        # Initialize components
        self.analyzer = CoverageAnalyzer(target_threshold)
        self.alert_system = CoverageAlertSystem(
            self.alert_config, self.output_dir, self.db_path, target_threshold
        )
        self.trend_analyzer = CoverageTrendAnalyzer(self.db_path)
        self.reporter = CoverageReporter(self.output_dir, target_threshold)

        # Current metrics storage
        self.current_metrics: CoverageMetrics | None = None

        logger.info(
            f"Coverage monitor initialized with {target_threshold}% target"
        )

    def _init_database(self) -> None:
        """Initialize SQLite database for historical coverage data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coverage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_coverage REAL NOT NULL,
                    total_statements INTEGER NOT NULL,
                    covered_statements INTEGER NOT NULL,
                    missing_statements INTEGER NOT NULL,
                    modules_count INTEGER NOT NULL,
                    modules_above_threshold INTEGER NOT NULL,
                    critical_gaps INTEGER NOT NULL,
                    branch_coverage REAL,
                    test_count INTEGER,
                    execution_time REAL,
                    commit_hash TEXT,
                    branch_name TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON coverage_history(timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_coverage
                ON coverage_history(overall_coverage)
            """
            )

    def run_coverage_analysis(
        self, include_branch: bool = True, save_artifacts: bool = True
    ) -> CoverageMetrics:
        """Run comprehensive coverage analysis with multiple report formats.

        Args:
            include_branch: Include branch coverage analysis
            save_artifacts: Save HTML and XML reports

        Returns:
            CoverageMetrics object with complete analysis results
        """
        # Run analysis using the analyzer component
        metrics = self.analyzer.run_coverage_analysis(
            include_branch=include_branch, save_artifacts=save_artifacts
        )

        # Store metrics
        self.current_metrics = metrics
        self._store_historical_data(metrics)

        # Generate comprehensive reports
        if save_artifacts:
            trend_data = self.trend_analyzer.analyze_trends(days=7)
            self.reporter.generate_summary_report(metrics, trend_data)
            self.trend_analyzer.generate_trend_report(self.output_dir)

        return metrics

    def _store_historical_data(self, metrics: CoverageMetrics) -> None:
        """Store coverage metrics in historical database.

        Args:
            metrics: Coverage metrics to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO coverage_history (
                    timestamp, overall_coverage, total_statements,
                    covered_statements, missing_statements, modules_count,
                    modules_above_threshold, critical_gaps, branch_coverage,
                    test_count, execution_time, commit_hash, branch_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.timestamp,
                    metrics.overall_coverage,
                    metrics.total_statements,
                    metrics.covered_statements,
                    metrics.missing_statements,
                    metrics.modules_count,
                    metrics.modules_above_threshold,
                    metrics.critical_gaps,
                    metrics.branch_coverage,
                    metrics.test_count,
                    metrics.execution_time,
                    self._get_git_commit_hash(),
                    self._get_git_branch(),
                ),
            )

        logger.info("Historical coverage data stored successfully")

    def _get_git_commit_hash(self) -> str | None:
        """Get current git commit hash.

        Returns:
            Git commit hash or None if not available
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _get_git_branch(self) -> str | None:
        """Get current git branch name.

        Returns:
            Git branch name or None if not available
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def check_and_send_alerts(self) -> bool:
        """Check coverage thresholds and send alerts if necessary.

        Returns:
            True if alerts were sent, False otherwise
        """
        if not self.current_metrics:
            return False

        return self.alert_system.check_and_send_alerts(self.current_metrics)

    def analyze_trends(self, days: int = 30) -> dict[str, Any]:
        """Analyze coverage trends over specified period.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        return self.trend_analyzer.analyze_trends(days)

    def generate_coverage_badge(self) -> Path:
        """Generate coverage badge for README.

        Returns:
            Path to generated badge file
        """
        if not self.current_metrics:
            raise ValueError(
                "No coverage metrics available for badge generation"
            )

        return self.reporter.generate_coverage_badge(self.current_metrics)
