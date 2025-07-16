"""Continuous coverage monitoring system for CrackSeg project.

This module provides comprehensive coverage monitoring with real-time tracking,
historical analysis, and automated alerting capabilities. Integrates with
existing CI/CD infrastructure while providing enhanced monitoring features.

Features:
- Real-time coverage tracking with configurable thresholds
- Historical trend analysis and metric storage
- Automated alerting when coverage drops below targets
- Integration with CI/CD pipelines and reporting systems
- Comprehensive reporting in multiple formats (HTML, JSON, text)
- Coverage badge generation and maintenance

Example:
    >>> monitor = CoverageMonitor(target_threshold=80.0)
    >>> monitor.run_coverage_analysis()
    >>> alert_sent = monitor.check_and_send_alerts()
    >>> trend_data = monitor.analyze_trends()
"""

import json
import logging
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        output_dir: Path = Path("outputs/coverage_monitoring"),
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
        start_time = datetime.now()

        logger.info("Starting coverage analysis...")

        # Build pytest command with comprehensive reporting
        cmd = [
            "pytest",
            "--cov=src",
            "--cov=scripts/gui",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-report=xml:coverage.xml",
        ]

        if include_branch:
            cmd.append("--cov-branch")

        if save_artifacts:
            cmd.extend(
                [
                    "--cov-report=html:htmlcov",
                    f"--html={self.output_dir}/reports/test_report.html",
                    "--self-contained-html",
                ]
            )

        # Execute coverage analysis
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )

            if result.returncode not in [0, 1]:  # Allow test failures
                logger.warning(
                    "Coverage analysis completed with warnings: "
                    f"{result.returncode}"
                )

        except subprocess.CalledProcessError as e:
            logger.error(f"Coverage analysis failed: {e}")
            raise

        # Parse coverage results
        metrics = self._parse_coverage_results(
            execution_time=(datetime.now() - start_time).total_seconds()
        )

        # Store metrics
        self.current_metrics = metrics
        self._store_historical_data(metrics)

        # Generate comprehensive reports
        if save_artifacts:
            self._generate_summary_report(metrics)
            self._generate_trend_report()

        logger.info(
            f"Coverage analysis completed: {metrics.overall_coverage:.1f}%"
        )
        return metrics

    def _parse_coverage_results(
        self, execution_time: float
    ) -> CoverageMetrics:
        """Parse coverage results from generated files."""

        # Parse JSON coverage report
        coverage_data: dict[str, Any] = {}
        if Path("coverage.json").exists():
            with open("coverage.json") as f:
                coverage_data = json.load(f)

        # Parse XML for additional metadata
        branch_coverage: float | None = None
        if Path("coverage.xml").exists():
            tree = ET.parse("coverage.xml")
            root = tree.getroot()
            branch_rate = root.get("branch-rate")
            if branch_rate is not None:
                branch_coverage = float(branch_rate) * 100

        # Extract key metrics with type safety
        totals = cast(dict[str, Any], coverage_data.get("totals", {}))

        overall_coverage = cast(float, totals.get("percent_covered", 0.0))
        total_statements = cast(int, totals.get("num_statements", 0))
        covered_statements = cast(int, totals.get("covered_lines", 0))
        missing_statements = cast(int, totals.get("missing_lines", 0))

        # Analyze modules
        files = cast(dict[str, Any], coverage_data.get("files", {}))
        modules_count = len(files)
        modules_above_threshold = sum(
            1
            for file_data in files.values()
            if cast(
                float,
                cast(dict[str, Any], file_data.get("summary", {})).get(
                    "percent_covered", 0
                ),
            )
            >= self.target_threshold
        )

        # Identify critical gaps (modules with <50% coverage)
        critical_gaps = sum(
            1
            for file_data in files.values()
            if cast(
                float,
                cast(dict[str, Any], file_data.get("summary", {})).get(
                    "percent_covered", 0
                ),
            )
            < 50.0
        )

        return CoverageMetrics(
            timestamp=datetime.now().isoformat(),
            overall_coverage=overall_coverage,
            total_statements=total_statements,
            covered_statements=covered_statements,
            missing_statements=missing_statements,
            modules_count=modules_count,
            modules_above_threshold=modules_above_threshold,
            critical_gaps=critical_gaps,
            branch_coverage=branch_coverage,
            execution_time=execution_time,
        )

    def _get_git_commit_hash(self) -> str | None:
        """Get current git commit hash."""
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

    def _store_historical_data(self, metrics: CoverageMetrics) -> None:
        """Store coverage metrics in historical database."""
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

    def _get_git_branch(self) -> str | None:
        """Get current git branch name."""
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
        if not self.alert_config.enabled or not self.current_metrics:
            return False

        alerts_sent = False

        # Check current coverage thresholds
        coverage = self.current_metrics.overall_coverage

        if coverage < self.alert_config.threshold_critical:
            critical_msg = (
                f"Coverage dropped to {coverage:.1f}% "
                f"(below {self.alert_config.threshold_critical}%)"
            )
            self._send_alert("CRITICAL", critical_msg, self.current_metrics)
            alerts_sent = True

        elif coverage < self.alert_config.threshold_warning:
            warning_msg = (
                f"Coverage at {coverage:.1f}% "
                f"(below {self.alert_config.threshold_warning}%)"
            )
            self._send_alert("WARNING", warning_msg, self.current_metrics)
            alerts_sent = True

        # Check trend alerts
        if self._check_trend_alerts():
            alerts_sent = True

        return alerts_sent

    def _check_trend_alerts(self) -> bool:
        """Check for concerning coverage trends."""
        # Ensure we have current metrics
        if not self.current_metrics:
            return False

        # Get coverage data from last N days
        cutoff_date = (
            datetime.now() - timedelta(days=self.alert_config.trend_alert_days)
        ).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT overall_coverage, timestamp
                FROM coverage_history
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """,
                (cutoff_date,),
            )

            data = cursor.fetchall()

        if len(data) < 2:
            return False

        # Calculate trend
        first_coverage = data[0][0]
        latest_coverage = data[-1][0]
        decline = first_coverage - latest_coverage

        if decline >= self.alert_config.trend_decline_threshold:
            trend_msg = (
                f"Coverage declined by {decline:.1f}% over "
                f"{self.alert_config.trend_alert_days} days"
            )
            self._send_alert("TREND", trend_msg, self.current_metrics)
            return True

        return False

    def _send_alert(
        self, alert_type: str, message: str, metrics: CoverageMetrics
    ) -> None:
        """Send coverage alert via configured channels."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": timestamp,
            "metrics": asdict(metrics),
            "target_threshold": self.target_threshold,
        }

        # Save alert to file
        alert_file = (
            self.output_dir
            / "alerts"
            / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(alert_file, "w") as f:
            json.dump(alert_data, f, indent=2)

        # Send email alerts
        if self.alert_config.email_recipients:
            self._send_email_alert(alert_type, message, metrics)

        # Send Slack alerts
        if self.alert_config.slack_webhook:
            self._send_slack_alert(alert_type, message, metrics)

        logger.warning(f"Coverage alert sent: {alert_type} - {message}")

    def _send_email_alert(
        self, alert_type: str, message: str, metrics: CoverageMetrics
    ) -> None:
        """Send email alert (placeholder implementation)."""
        # Note: Requires SMTP configuration in production
        email_msg = (
            f"Email alert would be sent to "
            f"{self.alert_config.email_recipients}: {message}"
        )
        logger.info(email_msg)

    def _send_slack_alert(
        self, alert_type: str, message: str, metrics: CoverageMetrics
    ) -> None:
        """Send Slack alert (placeholder implementation)."""
        # Note: Requires Slack webhook configuration in production
        logger.info(f"Slack alert would be sent: {message}")

    def analyze_trends(self, days: int = 30) -> dict[str, Any]:
        """Analyze coverage trends over specified period.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, overall_coverage, modules_above_threshold,
                       critical_gaps
                FROM coverage_history
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """,
                (cutoff_date,),
            )

            data = cursor.fetchall()

        if not data:
            return {"error": "No historical data available"}

        # Calculate trends
        coverages = [row[1] for row in data]
        modules_above = [row[2] for row in data]
        critical_gaps = [row[3] for row in data]

        trend_analysis = {
            "period_days": days,
            "data_points": len(data),
            "coverage_trend": {
                "start": coverages[0] if coverages else 0,
                "end": coverages[-1] if coverages else 0,
                "change": (
                    coverages[-1] - coverages[0] if len(coverages) >= 2 else 0
                ),
                "max": max(coverages) if coverages else 0,
                "min": min(coverages) if coverages else 0,
                "average": sum(coverages) / len(coverages) if coverages else 0,
            },
            "modules_trend": {
                "start": modules_above[0] if modules_above else 0,
                "end": modules_above[-1] if modules_above else 0,
                "change": (
                    modules_above[-1] - modules_above[0]
                    if len(modules_above) >= 2
                    else 0
                ),
            },
            "critical_gaps_trend": {
                "start": critical_gaps[0] if critical_gaps else 0,
                "end": critical_gaps[-1] if critical_gaps else 0,
                "change": (
                    critical_gaps[-1] - critical_gaps[0]
                    if len(critical_gaps) >= 2
                    else 0
                ),
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return trend_analysis

    def _generate_summary_report(self, metrics: CoverageMetrics) -> None:
        """Generate comprehensive coverage summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"coverage_summary_{timestamp}.json"
        report_path = self.output_dir / "reports" / report_filename

        # Get recent trend data
        trend_data = self.analyze_trends(days=7)

        report = {
            "current_metrics": asdict(metrics),
            "target_threshold": self.target_threshold,
            "status": (
                "PASS"
                if metrics.overall_coverage >= self.target_threshold
                else "FAIL"
            ),
            "gap_to_target": self.target_threshold - metrics.overall_coverage,
            "trend_analysis": trend_data,
            "recommendations": self._generate_recommendations(metrics),
            "generated_at": datetime.now().isoformat(),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Summary report generated: {report_path}")

    def _generate_recommendations(self, metrics: CoverageMetrics) -> list[str]:
        """Generate actionable recommendations based on coverage metrics."""
        recommendations: list[str] = []

        if metrics.overall_coverage < self.target_threshold:
            gap = self.target_threshold - metrics.overall_coverage
            recommendations.append(
                f"Increase overall coverage by {gap:.1f}% to reach target"
            )

        if metrics.critical_gaps > 0:
            recommendations.append(
                f"Address {metrics.critical_gaps} modules with <50% coverage"
            )

        coverage_ratio = (
            metrics.modules_above_threshold / metrics.modules_count
            if metrics.modules_count > 0
            else 0
        )
        if coverage_ratio < 0.8:
            recommendations.append(
                "Focus on bringing more modules above threshold coverage"
            )

        if metrics.missing_statements > 1000:
            recommendations.append(
                "Prioritize high-impact modules to reduce missing statements "
                "efficiently"
            )

        return recommendations

    def _generate_trend_report(self) -> None:
        """Generate detailed trend analysis report."""
        trend_data = self.analyze_trends(days=30)

        report_path = (
            self.output_dir
            / "reports"
            / f"trend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_path, "w") as f:
            json.dump(trend_data, f, indent=2)

        logger.info(f"Trend report generated: {report_path}")

    def generate_coverage_badge(self) -> Path:
        """Generate coverage badge for README.

        Returns:
            Path to generated badge file
        """
        if not self.current_metrics:
            raise ValueError(
                "No coverage metrics available for badge generation"
            )

        coverage = self.current_metrics.overall_coverage

        # Determine badge color based on coverage
        if coverage >= 90:
            color = "brightgreen"
        elif coverage >= 80:
            color = "green"
        elif coverage >= 70:
            color = "yellow"
        elif coverage >= 60:
            color = "orange"
        else:
            color = "red"

        # Create badge URL (shields.io format)
        badge_url = (
            f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{color}"
        )

        # Save badge info
        badge_info = {
            "coverage": coverage,
            "color": color,
            "url": badge_url,
            "generated_at": datetime.now().isoformat(),
        }

        badge_file = self.output_dir / "coverage_badge.json"
        with open(badge_file, "w") as f:
            json.dump(badge_info, f, indent=2)

        logger.info(f"Coverage badge generated: {coverage:.1f}% ({color})")
        return badge_file
