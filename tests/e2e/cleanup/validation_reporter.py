"""Validation Reporting and Alerting System.

This module provides comprehensive reporting capabilities and alerting
mechanisms for cleanup validation results, integrating with performance
thresholds and audit trails.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .audit_trail import AuditSummary, AuditTrailGenerator
from .environment_readiness import ReadinessReport
from .post_cleanup_validator import WorkflowResult
from .validation_system import ValidationResult

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReportFormat(Enum):
    """Report output formats."""

    HTML = "html"
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class AlertRule:
    """Configuration for alert generation rules."""

    name: str
    description: str
    condition: str  # Python expression to evaluate
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 60  # Minimum time between identical alerts
    recipients: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    test_id: str
    timestamp: datetime
    overall_status: str
    total_duration: float
    validation_result: ValidationResult | None = None
    workflow_results: list[WorkflowResult] = field(default_factory=list)
    readiness_report: ReadinessReport | None = None
    audit_summary: AuditSummary | None = None
    alerts_generated: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class AlertNotification:
    """Alert notification details."""

    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    test_id: str
    component: str
    recipients: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ValidationReporter:
    """Generates comprehensive validation reports and manages alerting."""

    def __init__(
        self,
        thresholds_config: dict[str, Any] | None = None,
        alert_rules: list[AlertRule] | None = None,
        email_config: dict[str, str] | None = None,
    ) -> None:
        """Initialize validation reporter."""
        self.thresholds_config = thresholds_config or {}
        self.alert_rules = alert_rules or self._default_alert_rules()
        self.email_config = email_config or {}

        self.logger = logging.getLogger(__name__)
        self._alert_history: dict[str, datetime] = {}  # For cooldown tracking
        self.audit_generator = AuditTrailGenerator()

    def _default_alert_rules(self) -> list[AlertRule]:
        """Generate default alert rules."""
        return [
            AlertRule(
                name="resource_leak_detected",
                description="Alert when resource leaks are detected",
                condition=(
                    "validation_result and validation_result.leak_detected"
                ),
                severity=AlertSeverity.HIGH,
                recipients=["devops@example.com"],
            ),
            AlertRule(
                name="validation_failed",
                description="Alert when validation fails",
                condition=(
                    "validation_result and "
                    "validation_result.status.value == 'failed'"
                ),
                severity=AlertSeverity.MEDIUM,
                recipients=["qa@example.com"],
            ),
            AlertRule(
                name="rollback_required",
                description="Alert when rollback is required",
                condition=(
                    "validation_result and "
                    "validation_result.rollback_performed"
                ),
                severity=AlertSeverity.CRITICAL,
                recipients=["devops@example.com", "qa@example.com"],
            ),
            AlertRule(
                name="environment_not_ready",
                description="Alert when environment is not ready",
                condition=(
                    "readiness_report and "
                    "readiness_report.overall_status.value == 'not_ready'"
                ),
                severity=AlertSeverity.HIGH,
                recipients=["devops@example.com"],
            ),
            AlertRule(
                name="performance_degradation",
                description="Alert when performance thresholds are violated",
                condition=(
                    "len(workflow_results) > 0 and "
                    "any(len(wr.threshold_violations) > 0 "
                    "for wr in workflow_results)"
                ),
                severity=AlertSeverity.MEDIUM,
                recipients=["performance@example.com"],
            ),
            AlertRule(
                name="excessive_cleanup_time",
                description="Alert when cleanup takes too long",
                condition=(
                    "total_duration > "
                    "thresholds_config.get('max_cleanup_duration_s', 300)"
                ),
                severity=AlertSeverity.LOW,
                recipients=["devops@example.com"],
            ),
        ]

    def generate_validation_report(
        self,
        test_id: str,
        validation_result: ValidationResult | None = None,
        workflow_results: list[WorkflowResult] | None = None,
        readiness_report: ReadinessReport | None = None,
        audit_summary: AuditSummary | None = None,
    ) -> ValidationReport:
        """Generate comprehensive validation report."""
        start_time = time.time()

        self.logger.info(f"Generating validation report for test {test_id}")

        # Determine overall status
        overall_status = self._determine_overall_status(
            validation_result, workflow_results or [], readiness_report
        )

        # Collect performance metrics
        performance_metrics = self._collect_performance_metrics(
            validation_result, workflow_results or [], readiness_report
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            validation_result,
            workflow_results or [],
            readiness_report,
            audit_summary,
        )

        # Create report
        report = ValidationReport(
            test_id=test_id,
            timestamp=datetime.now(UTC),
            overall_status=overall_status,
            total_duration=time.time() - start_time,
            validation_result=validation_result,
            workflow_results=workflow_results or [],
            readiness_report=readiness_report,
            audit_summary=audit_summary,
            recommendations=recommendations,
            performance_metrics=performance_metrics,
        )

        # Check for alerts
        alerts_generated = self._check_alert_rules(report)
        report.alerts_generated = alerts_generated

        self.logger.info(
            f"Validation report generated for test {test_id}: "
            f"status={overall_status}, alerts={len(alerts_generated)}"
        )

        return report

    def generate_alert(
        self,
        rule: AlertRule,
        report: ValidationReport,
        custom_message: str | None = None,
    ) -> AlertNotification:
        """Generate alert notification from rule and report."""
        alert_id = f"{report.test_id}_{rule.name}_{int(time.time())}"

        title = f"Cleanup Validation Alert: {rule.description}"

        if custom_message:
            message = custom_message
        else:
            message = self._generate_alert_message(rule, report)

        alert = AlertNotification(
            alert_id=alert_id,
            severity=rule.severity,
            title=title,
            message=message,
            timestamp=datetime.now(UTC),
            test_id=report.test_id,
            component="validation_reporter",
            recipients=rule.recipients,
            metadata={
                "rule_name": rule.name,
                "overall_status": report.overall_status,
                "total_duration": report.total_duration,
            },
        )

        # Record alert in audit trail
        self.audit_generator.record_alert_sent(
            alert_type=rule.name,
            severity=rule.severity.value,
            recipients=rule.recipients,
            message=message,
        )

        return alert

    def send_alert(self, alert: AlertNotification) -> bool:
        """Send alert notification via configured channels."""
        try:
            # Check cooldown period
            if self._is_in_cooldown(alert):
                self.logger.debug(
                    f"Alert {alert.alert_id} skipped due to cooldown"
                )
                return False

            # Update alert history
            self._alert_history[alert.alert_id] = alert.timestamp

            # Send email if configured
            if self.email_config and alert.recipients:
                email_sent = self._send_email_alert(alert)
                if email_sent:
                    self.logger.info(f"Email alert sent: {alert.alert_id}")

            # Log alert
            self.logger.warning(
                f"ALERT [{alert.severity.value.upper()}] {alert.title}: "
                f"{alert.message}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to send alert {alert.alert_id}: {e}")
            return False

    def export_report(
        self,
        report: ValidationReport,
        format_type: ReportFormat = ReportFormat.HTML,
        output_path: Path | None = None,
    ) -> Path:
        """Export validation report to file."""
        if not output_path:
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = (
                f"validation_report_{report.test_id}_{timestamp}."
                f"{format_type.value}"
            )
            output_path = Path("test-artifacts/reports") / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format_type == ReportFormat.HTML:
                content = self._generate_html_report(report)
            elif format_type == ReportFormat.JSON:
                import json

                content = json.dumps(
                    self._report_to_dict(report), indent=2, default=str
                )
            elif format_type == ReportFormat.MARKDOWN:
                content = self._generate_markdown_report(report)
            else:  # TEXT
                content = self._generate_text_report(report)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"Report exported to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            raise

    def _determine_overall_status(
        self,
        validation_result: ValidationResult | None,
        workflow_results: list[WorkflowResult],
        readiness_report: ReadinessReport | None,
    ) -> str:
        """Determine overall validation status."""
        if validation_result and validation_result.leak_detected:
            return "FAILED_LEAK_DETECTED"

        if (
            readiness_report
            and readiness_report.overall_status.value == "not_ready"
        ):
            return "FAILED_ENVIRONMENT_NOT_READY"

        if workflow_results:
            failed_workflows = [
                wr for wr in workflow_results if wr.steps_failed > 0
            ]
            if failed_workflows:
                return "FAILED_WORKFLOW_VALIDATION"

        if validation_result and validation_result.rollback_performed:
            return "COMPLETED_WITH_ROLLBACK"

        return "PASSED"

    def _collect_performance_metrics(
        self,
        validation_result: ValidationResult | None,
        workflow_results: list[WorkflowResult],
        readiness_report: ReadinessReport | None,
    ) -> dict[str, float]:
        """Collect performance metrics from all validation components."""
        metrics = {}

        if validation_result:
            metrics["validation_duration"] = (
                validation_result.validation_duration
            )

        if workflow_results:
            total_workflow_duration = sum(
                wr.total_duration for wr in workflow_results
            )
            metrics["total_workflow_duration"] = total_workflow_duration
            metrics["workflows_executed"] = len(workflow_results)

            # Collect individual workflow metrics
            for wr in workflow_results:
                metrics.update(wr.performance_metrics)

        if readiness_report:
            metrics["readiness_check_duration"] = (
                readiness_report.total_duration
            )
            metrics["components_checked"] = readiness_report.components_checked
            metrics["components_ready"] = readiness_report.components_ready

        return metrics

    def _generate_recommendations(
        self,
        validation_result: ValidationResult | None,
        workflow_results: list[WorkflowResult],
        readiness_report: ReadinessReport | None,
        audit_summary: AuditSummary | None,
    ) -> list[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        if validation_result and validation_result.leak_detected:
            recommendations.append(
                "Resource leaks detected - review cleanup procedures and "
                "implement more aggressive cleanup strategies"
            )

        if workflow_results:
            failed_workflows = [
                wr for wr in workflow_results if wr.steps_failed > 0
            ]
            if failed_workflows:
                recommendations.append(
                    f"Workflow validation failures detected in "
                    f"{len(failed_workflows)} workflows - investigate step "
                    f"failures and adjust thresholds if necessary"
                )

        if readiness_report and readiness_report.components_failed > 0:
            recommendations.append(
                "Environment readiness issues detected - ensure all required "
                "components are properly configured and available"
            )

        if audit_summary and audit_summary.threshold_violations > 3:
            recommendations.append(
                "Multiple performance threshold violations detected - "
                "consider optimizing cleanup procedures or adjusting "
                "performance thresholds"
            )

        return recommendations

    def _check_alert_rules(self, report: ValidationReport) -> list[str]:
        """Check alert rules and generate alerts if conditions are met."""
        alerts_generated = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            try:
                # Create evaluation context
                context = {
                    "validation_result": report.validation_result,
                    "workflow_results": report.workflow_results,
                    "readiness_report": report.readiness_report,
                    "audit_summary": report.audit_summary,
                    "total_duration": report.total_duration,
                    "thresholds_config": self.thresholds_config,
                }

                # Evaluate rule condition
                if eval(rule.condition, {"__builtins__": {}}, context):
                    alert = self.generate_alert(rule, report)
                    sent = self.send_alert(alert)

                    if sent:
                        alerts_generated.append(rule.name)

            except Exception as e:
                self.logger.error(
                    f"Error evaluating alert rule {rule.name}: {e}"
                )

        return alerts_generated

    def _generate_alert_message(
        self, rule: AlertRule, report: ValidationReport
    ) -> str:
        """Generate alert message based on rule and report."""
        message_parts = [
            f"Test ID: {report.test_id}",
            f"Overall Status: {report.overall_status}",
            f"Duration: {report.total_duration:.2f}s",
        ]

        if report.validation_result:
            message_parts.append(
                f"Validation Status: {report.validation_result.status.value}"
            )
            if report.validation_result.leak_detected:
                message_parts.append("⚠️ Resource leaks detected")

        if report.workflow_results:
            failed_workflows = [
                wr for wr in report.workflow_results if wr.steps_failed > 0
            ]
            if failed_workflows:
                message_parts.append(
                    f"Failed Workflows: {len(failed_workflows)}"
                )

        if (
            report.readiness_report
            and report.readiness_report.components_failed > 0
        ):
            failed_count = report.readiness_report.components_failed
            message_parts.append(
                f"Environment Issues: {failed_count} components failed"
            )

        return "\n".join(message_parts)

    def _is_in_cooldown(self, alert: AlertNotification) -> bool:
        """Check if alert is in cooldown period."""
        rule = next(
            (r for r in self.alert_rules if r.name in alert.alert_id), None
        )
        if not rule:
            return False

        last_alert_time = self._alert_history.get(alert.alert_id)
        if not last_alert_time:
            return False

        time_diff = (alert.timestamp - last_alert_time).total_seconds()
        return time_diff < (rule.cooldown_minutes * 60)

    def _send_email_alert(self, alert: AlertNotification) -> bool:
        """Send email alert notification."""
        try:
            # Email implementation placeholder
            # In real implementation, would use SMTP configuration
            self.logger.info(
                f"Email alert would be sent to: {alert.recipients}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False

    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML format report."""
        # HTML template implementation
        return f"""
        <html>
        <head><title>Validation Report - {report.test_id}</title></head>
        <body>
        <h1>Cleanup Validation Report</h1>
        <h2>Test ID: {report.test_id}</h2>
        <p>Status: {report.overall_status}</p>
        <p>Duration: {report.total_duration:.2f}s</p>
        <!-- Additional HTML content would go here -->
        </body>
        </html>
        """

    def _generate_markdown_report(self, report: ValidationReport) -> str:
        """Generate Markdown format report."""
        return f"""# Cleanup Validation Report

## Test Information
- **Test ID**: {report.test_id}
- **Status**: {report.overall_status}
- **Duration**: {report.total_duration:.2f}s
- **Timestamp**: {report.timestamp}

## Validation Results
{self._format_validation_section(report)}

## Recommendations
{chr(10).join(f"- {rec}" for rec in report.recommendations)}
"""

    def _generate_text_report(self, report: ValidationReport) -> str:
        """Generate plain text format report."""
        return f"""Cleanup Validation Report
Test ID: {report.test_id}
Status: {report.overall_status}
Duration: {report.total_duration:.2f}s
Timestamp: {report.timestamp}

Recommendations:
{chr(10).join(f"- {rec}" for rec in report.recommendations)}
"""

    def _format_validation_section(self, report: ValidationReport) -> str:
        """Format validation results section for reports."""
        if not report.validation_result:
            return "No validation results available"

        return f"""- Status: {report.validation_result.status.value}
- Leak Detected: {report.validation_result.leak_detected}
- Duration: {report.validation_result.validation_duration:.2f}s
- Rollback Performed: {report.validation_result.rollback_performed}"""

    def _report_to_dict(self, report: ValidationReport) -> dict[str, Any]:
        """Convert report to dictionary for JSON export."""
        # Implementation would convert all report fields to serializable format
        return {
            "test_id": report.test_id,
            "overall_status": report.overall_status,
            "timestamp": report.timestamp.isoformat(),
            "total_duration": report.total_duration,
            # Additional fields would be included here
        }
