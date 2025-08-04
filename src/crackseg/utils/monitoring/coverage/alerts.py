"""Alert system for coverage monitoring.

This module handles coverage alerts including threshold checking,
trend analysis, and multi-channel alert delivery.
"""

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from .config import AlertConfig, CoverageMetrics

logger = logging.getLogger(__name__)


class CoverageAlertSystem:
    """Handles coverage alerts and notifications."""

    def __init__(
        self,
        alert_config: AlertConfig,
        output_dir: Path,
        db_path: Path,
        target_threshold: float,
    ) -> None:
        """Initialize the alert system.

        Args:
            alert_config: Alert configuration
            output_dir: Output directory for alert files
            db_path: Database path for historical data
            target_threshold: Target coverage threshold
        """
        self.alert_config = alert_config
        self.output_dir = output_dir
        self.db_path = db_path
        self.target_threshold = target_threshold

    def check_and_send_alerts(self, current_metrics: CoverageMetrics) -> bool:
        """Check coverage thresholds and send alerts if necessary.

        Args:
            current_metrics: Current coverage metrics

        Returns:
            True if alerts were sent, False otherwise
        """
        if not self.alert_config.enabled or not current_metrics:
            return False

        alerts_sent = False

        # Check current coverage thresholds
        coverage = current_metrics.overall_coverage

        if coverage < self.alert_config.threshold_critical:
            critical_msg = (
                f"Coverage dropped to {coverage:.1f}% "
                f"(below {self.alert_config.threshold_critical}%)"
            )
            self._send_alert("CRITICAL", critical_msg, current_metrics)
            alerts_sent = True

        elif coverage < self.alert_config.threshold_warning:
            warning_msg = (
                f"Coverage at {coverage:.1f}% "
                f"(below {self.alert_config.threshold_warning}%)"
            )
            self._send_alert("WARNING", warning_msg, current_metrics)
            alerts_sent = True

        # Check trend alerts
        if self._check_trend_alerts(current_metrics):
            alerts_sent = True

        return alerts_sent

    def _check_trend_alerts(self, current_metrics: CoverageMetrics) -> bool:
        """Check for concerning coverage trends.

        Args:
            current_metrics: Current coverage metrics

        Returns:
            True if trend alert was sent, False otherwise
        """
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
            self._send_alert("TREND", trend_msg, current_metrics)
            return True

        return False

    def _send_alert(
        self, alert_type: str, message: str, metrics: CoverageMetrics
    ) -> None:
        """Send coverage alert via configured channels.

        Args:
            alert_type: Type of alert (CRITICAL, WARNING, TREND)
            message: Alert message
            metrics: Coverage metrics
        """
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
        alert_file.parent.mkdir(parents=True, exist_ok=True)
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
        """Send email alert (placeholder implementation).

        Args:
            alert_type: Type of alert
            message: Alert message
            metrics: Coverage metrics
        """
        # Note: Requires SMTP configuration in production
        email_msg = (
            f"Email alert would be sent to "
            f"{self.alert_config.email_recipients}: {message}"
        )
        logger.info(email_msg)

    def _send_slack_alert(
        self, alert_type: str, message: str, metrics: CoverageMetrics
    ) -> None:
        """Send Slack alert (placeholder implementation).

        Args:
            alert_type: Type of alert
            message: Alert message
            metrics: Coverage metrics
        """
        # Note: Requires Slack webhook configuration in production
        logger.info(f"Slack alert would be sent: {message}")
