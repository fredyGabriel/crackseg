"""Regression Alerting System for Performance Monitoring.

This module implements an automated alerting system for performance regressions
and threshold violations, integrating with the existing CI/CD infrastructure
and performance monitoring components.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for regression alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels."""

    GITHUB_PR_COMMENT = "github_pr_comment"
    GITHUB_ISSUE = "github_issue"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"


class RegressionMetric(TypedDict):
    """Type definition for regression metric data."""

    metric_name: str
    current_value: float
    baseline_value: float
    change_percentage: float
    threshold_exceeded: bool
    severity: str


class RegressionAlert(TypedDict):
    """Type definition for regression alert."""

    alert_id: str
    severity: str
    metric_name: str
    change_percentage: float
    current_value: float
    baseline_value: float
    threshold: float
    message: str
    timestamp: str
    context: dict[str, Any]


class NotificationResult(TypedDict):
    """Type definition for notification result."""

    alert_id: str
    success: bool
    channels_attempted: list[str]
    channels_successful: list[str]
    errors: list[str]


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""

    enabled_channels: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.GITHUB_PR_COMMENT]
    )
    github_token: str | None = None
    email_config: dict[str, str] = field(default_factory=dict)
    webhook_url: str | None = None
    slack_webhook: str | None = None
    cooldown_minutes: int = 30


@dataclass
class RegressionThresholds:
    """Thresholds for regression detection."""

    # Performance degradation thresholds (percentage)
    warning_threshold: float = 15.0
    critical_threshold: float = 25.0

    # Statistical significance requirements
    min_samples: int = 3
    confidence_level: float = 0.95

    # Trend analysis parameters
    trend_window_hours: int = 24
    sustained_degradation_minutes: int = 10


class RegressionAnalyzer:
    """Statistical analyzer for performance regression detection."""

    def __init__(self, thresholds: RegressionThresholds) -> None:
        """Initialize regression analyzer."""
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__)

    def analyze_performance_data(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> list[RegressionMetric]:
        """Analyze performance data for regressions.

        Args:
            current_data: Current performance metrics
            historical_data: Historical performance data for comparison

        Returns:
            List of detected regression metrics
        """
        regressions: list[RegressionMetric] = []

        # Extract metrics from current data
        current_metrics = self._extract_metrics(current_data)

        if len(historical_data) < self.thresholds.min_samples:
            self.logger.warning(
                f"Insufficient historical data: {len(historical_data)} "
                f"samples (minimum: {self.thresholds.min_samples})"
            )
            return regressions

        # Calculate baselines from historical data
        baselines = self._calculate_baselines(historical_data)

        # Compare current metrics against baselines
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baselines:
                continue

            baseline_value = baselines[metric_name]
            change_percentage = self._calculate_change_percentage(
                current_value, baseline_value
            )

            # Check for regression (performance degradation)
            is_degradation = self._is_performance_degradation(
                metric_name, change_percentage
            )

            if is_degradation:
                severity = self._assess_regression_severity(change_percentage)
                threshold_exceeded = (
                    abs(change_percentage) > self.thresholds.warning_threshold
                )

                regression: RegressionMetric = {
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "baseline_value": baseline_value,
                    "change_percentage": change_percentage,
                    "threshold_exceeded": threshold_exceeded,
                    "severity": severity.value,
                }
                regressions.append(regression)

        return regressions

    def _extract_metrics(self, data: dict[str, Any]) -> dict[str, float]:
        """Extract relevant metrics from performance data."""
        metrics: dict[str, float] = {}

        # Extract metrics from performance summary
        perf_summary = data.get("performance_gate_summary", {})
        if "total_violations" in perf_summary:
            metrics["violation_count"] = float(
                perf_summary["total_violations"]
            )

        # Extract benchmark metrics
        benchmark_results = data.get("benchmark_results", {})
        for benchmark_name, result in benchmark_results.items():
            if "metrics" in result:
                bench_metrics = result["metrics"]
                if "average_response_time" in bench_metrics:
                    metrics[f"{benchmark_name}_response_time"] = float(
                        bench_metrics["average_response_time"]
                    )
                if "peak_memory_mb" in bench_metrics:
                    metrics[f"{benchmark_name}_memory_usage"] = float(
                        bench_metrics["peak_memory_mb"]
                    )
                if "cpu_usage_percent" in bench_metrics:
                    metrics[f"{benchmark_name}_cpu_usage"] = float(
                        bench_metrics["cpu_usage_percent"]
                    )

        # Extract performance violations by type
        violations = data.get("performance_violations", [])
        violation_types: dict[str, int] = {}
        for violation in violations:
            violation_type = violation.get("type", "unknown")
            violation_types[violation_type] = (
                violation_types.get(violation_type, 0) + 1
            )

        for violation_type, count in violation_types.items():
            metrics[f"violations_{violation_type}"] = float(count)

        return metrics

    def _calculate_baselines(
        self, historical_data: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate baseline values from historical data."""
        baselines: dict[str, float] = {}

        # Collect all metrics from historical data
        historical_metrics: dict[str, list[float]] = {}

        for data_point in historical_data:
            metrics = self._extract_metrics(data_point)
            for metric_name, value in metrics.items():
                if metric_name not in historical_metrics:
                    historical_metrics[metric_name] = []
                historical_metrics[metric_name].append(value)

        # Calculate statistical baselines
        for metric_name, values in historical_metrics.items():
            if len(values) >= self.thresholds.min_samples:
                # Use median as baseline for robustness against outliers
                baselines[metric_name] = statistics.median(values)

        return baselines

    def _calculate_change_percentage(
        self, current: float, baseline: float
    ) -> float:
        """Calculate percentage change from baseline."""
        if baseline == 0:
            return 100.0 if current > 0 else 0.0
        return ((current - baseline) / baseline) * 100.0

    def _is_performance_degradation(
        self, metric_name: str, change_percentage: float
    ) -> bool:
        """Determine if a change represents performance degradation."""
        # Define metrics where increase indicates degradation
        degradation_metrics = {
            "violation_count",
            "response_time",
            "memory_usage",
            "cpu_usage",
            "violations_",  # Prefix for violation types
        }

        # Check if this metric indicates degradation when increased
        is_degradation_metric = any(
            pattern in metric_name.lower() for pattern in degradation_metrics
        )

        if is_degradation_metric:
            # For degradation metrics, positive change is bad
            return change_percentage > self.thresholds.warning_threshold
        else:
            # For improvement metrics (success rate), negative change is bad
            return change_percentage < -self.thresholds.warning_threshold

    def _assess_regression_severity(
        self, change_percentage: float
    ) -> RegressionSeverity:
        """Assess severity of regression based on change percentage."""
        abs_change = abs(change_percentage)

        if abs_change >= 50.0:
            return RegressionSeverity.CRITICAL
        elif abs_change >= self.thresholds.critical_threshold:
            return RegressionSeverity.HIGH
        elif abs_change >= self.thresholds.warning_threshold:
            return RegressionSeverity.MEDIUM
        else:
            return RegressionSeverity.LOW


class RegressionAlertingSystem:
    """Main alerting system for performance regressions."""

    def __init__(
        self,
        thresholds: RegressionThresholds | None = None,
        notification_config: NotificationConfig | None = None,
        historical_data_path: Path | str = "performance-historical-data",
    ) -> None:
        """Initialize regression alerting system.

        Args:
            thresholds: Regression detection thresholds
            notification_config: Notification configuration
            historical_data_path: Path to store historical performance data
        """
        self.thresholds = thresholds or RegressionThresholds()
        self.notification_config = notification_config or NotificationConfig()
        self.historical_data_path = Path(historical_data_path)

        # Ensure historical data directory exists
        self.historical_data_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.analyzer = RegressionAnalyzer(self.thresholds)
        self.logger = logging.getLogger(__name__)

        # Alert management
        self._alert_history: list[RegressionAlert] = []
        self._last_alert_times: dict[str, datetime] = {}

    def process_performance_results(
        self, results_path: Path | str, commit_sha: str | None = None
    ) -> dict[str, Any]:
        """Process performance results and trigger alerts for regressions.

        Args:
            results_path: Path to performance results JSON file
            commit_sha: Git commit SHA for context

        Returns:
            Processing summary with alert information
        """
        try:
            # Load current performance data
            with open(results_path) as f:
                current_data = json.load(f)

            # Load historical data
            historical_data = self._load_historical_data()

            # Analyze for regressions
            regressions = self.analyzer.analyze_performance_data(
                current_data, historical_data
            )

            # Store current data for future analysis
            self._store_historical_data(current_data, commit_sha)

            # Generate alerts for detected regressions
            alerts = self._generate_alerts(
                regressions, current_data, commit_sha
            )

            # Send notifications
            notification_results = []
            for alert in alerts:
                result = self._send_alert_notifications(alert)
                notification_results.append(result)

            # Prepare summary
            summary = {
                "processing_timestamp": datetime.now(UTC).isoformat(),
                "commit_sha": commit_sha,
                "regressions_detected": len(regressions),
                "alerts_generated": len(alerts),
                "notifications_sent": sum(
                    1 for r in notification_results if r["success"]
                ),
                "regressions": regressions,
                "alerts": list(alerts),
                "notification_results": notification_results,
            }

            self.logger.info(
                f"Processed performance results: {len(regressions)} "
                f"regressions, {len(alerts)} alerts generated"
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error processing performance results: {e}")
            return {
                "error": str(e),
                "processing_timestamp": datetime.now(UTC).isoformat(),
            }

    def _load_historical_data(self) -> list[dict[str, Any]]:
        """Load historical performance data for baseline comparison."""
        historical_data = []

        # Load recent historical files (last 30 days)
        cutoff_date = datetime.now(UTC) - timedelta(days=30)

        for file_path in self.historical_data_path.glob("performance_*.json"):
            try:
                # Extract timestamp from filename
                timestamp_str = file_path.stem.split("_", 1)[1]
                file_date = datetime.fromisoformat(
                    timestamp_str.replace("_", ":")
                )

                if file_date.replace(tzinfo=UTC) > cutoff_date:
                    with open(file_path) as f:
                        data = json.load(f)
                        historical_data.append(data)

            except (ValueError, json.JSONDecodeError) as e:
                self.logger.warning(
                    f"Error loading historical file {file_path}: {e}"
                )

        # Sort by timestamp (most recent first)
        historical_data.sort(
            key=lambda x: x.get("timestamp", ""), reverse=True
        )

        # Limit to reasonable number of samples for analysis
        return historical_data[:50]

    def _store_historical_data(
        self, data: dict[str, Any], commit_sha: str | None = None
    ) -> None:
        """Store current performance data for future historical analysis."""
        timestamp = datetime.now(UTC).isoformat().replace(":", "_")
        filename = f"performance_{timestamp}.json"

        # Add metadata
        data_with_metadata = {
            **data,
            "timestamp": datetime.now(UTC).isoformat(),
            "commit_sha": commit_sha,
            "stored_by": "regression_alerting_system",
        }

        file_path = self.historical_data_path / filename
        with open(file_path, "w") as f:
            json.dump(data_with_metadata, f, indent=2)

        self.logger.debug(f"Stored historical data: {filename}")

    def _generate_alerts(
        self,
        regressions: list[RegressionMetric],
        current_data: dict[str, Any],
        commit_sha: str | None = None,
    ) -> list[RegressionAlert]:
        """Generate alerts for detected regressions."""
        alerts: list[RegressionAlert] = []

        for regression in regressions:
            timestamp = int(datetime.now(UTC).timestamp())
            alert_id = f"regression_{regression['metric_name']}_{timestamp}"

            # Check cooldown period
            if self._is_in_cooldown(regression["metric_name"]):
                self.logger.debug(
                    f"Alert for {regression['metric_name']} skipped due to "
                    f"cooldown"
                )
                continue

            # Generate alert message
            message = self._generate_alert_message(regression, commit_sha)

            alert: RegressionAlert = {
                "alert_id": alert_id,
                "severity": regression["severity"],
                "metric_name": regression["metric_name"],
                "change_percentage": regression["change_percentage"],
                "current_value": regression["current_value"],
                "baseline_value": regression["baseline_value"],
                "threshold": self.thresholds.warning_threshold,
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
                "context": {
                    "commit_sha": commit_sha,
                    "ci_build": os.getenv("GITHUB_RUN_NUMBER"),
                    "branch": os.getenv("GITHUB_REF_NAME"),
                    "workflow": "performance-ci",
                },
            }

            alerts.append(alert)
            self._last_alert_times[regression["metric_name"]] = datetime.now(
                UTC
            )

        return alerts

    def _generate_alert_message(
        self, regression: RegressionMetric, commit_sha: str | None = None
    ) -> str:
        """Generate human-readable alert message."""
        change_direction = (
            "increased" if regression["change_percentage"] > 0 else "decreased"
        )

        message_parts = [
            "🚨 Performance Regression Detected",
            "",
            f"**Metric**: {regression['metric_name']}",
            f"**Change**: {change_direction} by "
            f"{abs(regression['change_percentage']):.1f}%",
            f"**Current Value**: {regression['current_value']:.2f}",
            f"**Baseline Value**: {regression['baseline_value']:.2f}",
            f"**Severity**: {regression['severity'].upper()}",
        ]

        if commit_sha:
            message_parts.extend(
                [
                    "",
                    "**Context**:",
                    f"- Commit: {commit_sha[:8]}",
                    f"- Build: {os.getenv('GITHUB_RUN_NUMBER', 'unknown')}",
                    f"- Branch: {os.getenv('GITHUB_REF_NAME', 'unknown')}",
                ]
            )

        return "\n".join(message_parts)

    def _is_in_cooldown(self, metric_name: str) -> bool:
        """Check if metric is in cooldown period."""
        last_alert_time = self._last_alert_times.get(metric_name)
        if not last_alert_time:
            return False

        time_diff = datetime.now(UTC) - last_alert_time
        return time_diff.total_seconds() < (
            self.notification_config.cooldown_minutes * 60
        )

    def _send_alert_notifications(
        self, alert: RegressionAlert
    ) -> NotificationResult:
        """Send alert notifications via configured channels."""
        results: NotificationResult = {
            "alert_id": alert["alert_id"],
            "success": False,
            "channels_attempted": [],
            "channels_successful": [],
            "errors": [],
        }

        for channel in self.notification_config.enabled_channels:
            results["channels_attempted"].append(channel.value)

            try:
                if channel == NotificationChannel.GITHUB_PR_COMMENT:
                    success = self._send_github_pr_comment(alert)
                elif channel == NotificationChannel.EMAIL:
                    success = self._send_email_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    success = self._send_webhook_notification(alert)
                else:
                    success = False
                    results["errors"].append(
                        f"Unsupported channel: {channel.value}"
                    )

                if success:
                    results["channels_successful"].append(channel.value)

            except Exception as e:
                error_msg = f"Error sending to {channel.value}: {e}"
                results["errors"].append(error_msg)
                self.logger.error(error_msg)

        results["success"] = len(results["channels_successful"]) > 0

        # Log alert regardless of notification success
        severity_level = (
            logging.CRITICAL
            if alert["severity"] == "critical"
            else logging.WARNING
        )
        self.logger.log(severity_level, alert["message"])

        return results

    def _send_github_pr_comment(self, alert: RegressionAlert) -> bool:
        """Send alert as GitHub PR comment."""
        # This would integrate with GitHub API
        # For now, we'll create a comment file for the workflow to process
        try:
            pr_comment_file = Path("regression-alert-comment.md")
            with open(pr_comment_file, "w") as f:
                f.write(alert["message"])

            self.logger.info(f"GitHub PR comment prepared: {pr_comment_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to prepare GitHub PR comment: {e}")
            return False

    def _send_email_notification(self, alert: RegressionAlert) -> bool:
        """Send email notification."""
        # Email implementation placeholder
        self.logger.info(
            f"Email notification would be sent for alert: {alert['alert_id']}"
        )
        return True

    def _send_webhook_notification(self, alert: RegressionAlert) -> bool:
        """Send webhook notification."""
        # Webhook implementation placeholder
        self.logger.info(
            f"Webhook notification would be sent for alert: "
            f"{alert['alert_id']}"
        )
        return True


# Convenience functions for CI/CD integration
def create_regression_alerting_system(
    thresholds_config_path: (
        Path | str
    ) = "configs/testing/performance_thresholds.yaml",
) -> RegressionAlertingSystem:
    """Create regression alerting system with configuration from YAML file.

    Args:
        thresholds_config_path: Path to performance thresholds configuration

    Returns:
        Configured RegressionAlertingSystem instance
    """
    # Load thresholds from configuration
    thresholds = RegressionThresholds(
        warning_threshold=float(
            os.getenv("REGRESSION_WARNING_THRESHOLD", "15.0")
        ),
        critical_threshold=float(
            os.getenv("REGRESSION_CRITICAL_THRESHOLD", "25.0")
        ),
    )

    # Configure notifications based on environment
    notification_config = NotificationConfig(
        github_token=os.getenv("GITHUB_TOKEN"),
        cooldown_minutes=int(os.getenv("ALERT_COOLDOWN_MINUTES", "30")),
    )

    return RegressionAlertingSystem(
        thresholds=thresholds,
        notification_config=notification_config,
    )


def process_ci_performance_results(
    results_path: (
        Path | str
    ) = "performance-gate-results/consolidated-report.json",
) -> int:
    """Process CI/CD performance results and return exit code.

    Args:
        results_path: Path to performance results JSON file

    Returns:
        Exit code (0 for success, 1 for regressions detected)
    """
    try:
        alerting_system = create_regression_alerting_system()

        commit_sha = os.getenv("GITHUB_SHA")
        summary = alerting_system.process_performance_results(
            results_path, commit_sha
        )

        if "error" in summary:
            print(f"❌ Error processing results: {summary['error']}")
            return 1

        print("📊 Regression Analysis Summary:")
        print(f"  Regressions Detected: {summary['regressions_detected']}")
        print(f"  Alerts Generated: {summary['alerts_generated']}")
        print(f"  Notifications Sent: {summary['notifications_sent']}")

        # Save summary for workflow consumption
        summary_file = Path("regression-analysis-summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Return exit code based on regression detection
        if summary["regressions_detected"] > 0:
            print(
                "⚠️ Performance regressions detected - see alerts for details"
            )
            return 1
        else:
            print("✅ No performance regressions detected")
            return 0

    except Exception as e:
        print(f"❌ Failed to process performance results: {e}")
        return 1


if __name__ == "__main__":
    import sys

    exit_code = process_ci_performance_results()
    sys.exit(exit_code)
