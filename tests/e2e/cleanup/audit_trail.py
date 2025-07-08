"""Cleanup Audit Trail Generation System.

This module provides comprehensive audit trail functionality for tracking
cleanup operations, decisions, and outcomes for compliance and debugging.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .validation_system import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Audit trail detail levels."""

    BASIC = "basic"  # Essential operations only
    DETAILED = "detailed"  # Detailed operations and decisions
    COMPREHENSIVE = "comprehensive"  # Full trace with metrics
    DEBUG = "debug"  # Maximum detail for troubleshooting


class AuditEventType(Enum):
    """Types of audit events."""

    CLEANUP_STARTED = "cleanup_started"
    CLEANUP_COMPLETED = "cleanup_completed"
    CLEANUP_FAILED = "cleanup_failed"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    ROLLBACK_COMPLETED = "rollback_completed"
    RESOURCE_LEAK_DETECTED = "resource_leak_detected"
    THRESHOLD_VIOLATION = "threshold_violation"
    ENVIRONMENT_CHECK = "environment_check"
    ALERT_SENT = "alert_sent"


@dataclass
class AuditEvent:
    """Individual audit trail event."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    test_id: str
    component: str
    message: str
    level: AuditLevel
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    resource_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AuditSummary:
    """Summary of audit trail for a test cycle."""

    test_id: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    events_count: int
    cleanup_operations: int
    validation_operations: int
    rollback_operations: int
    errors_count: int
    warnings_count: int
    resource_leaks_detected: int
    threshold_violations: int
    final_status: str
    recommendations: list[str] = field(default_factory=list)


class AuditTrailGenerator:
    """Generates and manages cleanup operation audit trails."""

    def __init__(
        self,
        audit_level: AuditLevel = AuditLevel.DETAILED,
        audit_dir: Path | None = None,
    ) -> None:
        """Initialize audit trail generator."""
        self.audit_level = audit_level
        self.audit_dir = audit_dir or Path("test-artifacts/audit-trails")
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self._events: list[AuditEvent] = []
        self._current_test_id: str | None = None
        self._test_start_time: datetime | None = None

    def start_test_audit(self, test_id: str) -> None:
        """Start audit trail for a new test."""
        self._current_test_id = test_id
        self._test_start_time = datetime.now(UTC)
        self._events.clear()

        self._add_event(
            event_type=AuditEventType.CLEANUP_STARTED,
            component="audit_system",
            message=f"Started audit trail for test {test_id}",
            level=AuditLevel.BASIC,
        )

    def end_test_audit(self, final_status: str = "completed") -> AuditSummary:
        """End audit trail for current test and generate summary."""
        if not self._current_test_id or not self._test_start_time:
            raise RuntimeError("No active test audit to end")

        end_time = datetime.now(UTC)
        total_duration = (end_time - self._test_start_time).total_seconds()

        # Generate summary
        summary = self._generate_audit_summary(
            self._current_test_id,
            self._test_start_time,
            end_time,
            total_duration,
            final_status,
        )

        # Save audit trail to file
        self._save_audit_trail(self._current_test_id, summary)

        # Reset for next test
        self._current_test_id = None
        self._test_start_time = None

        return summary

    def record_cleanup_operation(
        self,
        operation_name: str,
        success: bool,
        duration: float,
        details: dict[str, Any] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Record a cleanup operation in the audit trail."""
        event_type = (
            AuditEventType.CLEANUP_COMPLETED
            if success
            else AuditEventType.CLEANUP_FAILED
        )

        status_text = "succeeded" if success else "failed"
        message = (
            f"Cleanup operation '{operation_name}' {status_text} "
            f"in {duration:.2f}s"
        )

        self._add_event(
            event_type=event_type,
            component="cleanup_manager",
            message=message,
            level=AuditLevel.DETAILED,
            duration_seconds=duration,
            metadata=details or {},
            errors=errors or [],
        )

    def record_validation_result(
        self,
        validation_result: ValidationResult,
        workflow_type: str | None = None,
    ) -> None:
        """Record a validation result in the audit trail."""
        event_type = (
            AuditEventType.VALIDATION_COMPLETED
            if validation_result.status
            in [ValidationStatus.PASSED, ValidationStatus.ROLLBACK_COMPLETED]
            else AuditEventType.VALIDATION_STARTED
        )

        message = (
            f"Validation {'(' + workflow_type + ') ' if workflow_type else ''}"
            f"completed with status: {validation_result.status.value}"
        )

        metadata = {
            "validation_duration": validation_result.validation_duration,
            "leak_detected": validation_result.leak_detected,
            "resource_differences": validation_result.resource_differences,
            "rollback_performed": validation_result.rollback_performed,
            "rollback_success": validation_result.rollback_success,
        }

        self._add_event(
            event_type=event_type,
            component="validation_system",
            message=message,
            level=AuditLevel.DETAILED,
            duration_seconds=validation_result.validation_duration,
            metadata=metadata,
            errors=validation_result.validation_errors,
        )

        # Record resource leak if detected
        if validation_result.leak_detected:
            self.record_resource_leak(validation_result.resource_differences)

    def record_resource_leak(
        self, resource_differences: dict[str, float]
    ) -> None:
        """Record detection of resource leaks."""
        leak_details = {
            resource: diff
            for resource, diff in resource_differences.items()
            if diff > 0
        }

        message = f"Resource leak detected: {leak_details}"

        self._add_event(
            event_type=AuditEventType.RESOURCE_LEAK_DETECTED,
            component="leak_detector",
            message=message,
            level=AuditLevel.BASIC,
            metadata={"leak_details": leak_details},
        )

    def record_threshold_violation(
        self,
        violation_type: str,
        threshold_value: float,
        actual_value: float,
        component: str,
    ) -> None:
        """Record performance threshold violations."""
        message = (
            f"Threshold violation in {component}: {violation_type} "
            f"exceeded {threshold_value} (actual: {actual_value})"
        )

        metadata = {
            "violation_type": violation_type,
            "threshold_value": threshold_value,
            "actual_value": actual_value,
            "severity": (
                "high" if actual_value > threshold_value * 1.5 else "medium"
            ),
        }

        self._add_event(
            event_type=AuditEventType.THRESHOLD_VIOLATION,
            component=component,
            message=message,
            level=AuditLevel.BASIC,
            metadata=metadata,
        )

    def record_environment_check(
        self,
        check_name: str,
        success: bool,
        duration: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record environment readiness checks."""
        status_text = "passed" if success else "failed"
        message = (
            f"Environment check '{check_name}' {status_text} "
            f"in {duration:.2f}s"
        )

        self._add_event(
            event_type=AuditEventType.ENVIRONMENT_CHECK,
            component="environment_checker",
            message=message,
            level=AuditLevel.DETAILED,
            duration_seconds=duration,
            metadata=details or {},
        )

    def record_rollback_operation(
        self,
        success: bool,
        duration: float,
        procedures: list[str] | None = None,
    ) -> None:
        """Record rollback operations."""
        event_type = (
            AuditEventType.ROLLBACK_COMPLETED
            if success
            else AuditEventType.ROLLBACK_TRIGGERED
        )

        status_text = "completed successfully" if success else "failed"
        message = f"Rollback operation {status_text} in {duration:.2f}s"

        metadata = {"procedures_executed": procedures or []}

        self._add_event(
            event_type=event_type,
            component="rollback_manager",
            message=message,
            level=AuditLevel.BASIC,
            duration_seconds=duration,
            metadata=metadata,
        )

    def record_alert_sent(
        self,
        alert_type: str,
        severity: str,
        recipients: list[str],
        message: str,
    ) -> None:
        """Record alert notifications sent."""
        audit_message = (
            f"Alert sent: {alert_type} ({severity}) to "
            f"{len(recipients)} recipients"
        )

        metadata = {
            "alert_type": alert_type,
            "severity": severity,
            "recipients": recipients,
            "alert_message": message,
        }

        self._add_event(
            event_type=AuditEventType.ALERT_SENT,
            component="alert_system",
            message=audit_message,
            level=AuditLevel.BASIC,
            metadata=metadata,
        )

    def _add_event(
        self,
        event_type: AuditEventType,
        component: str,
        message: str,
        level: AuditLevel,
        duration_seconds: float = 0.0,
        metadata: dict[str, Any] | None = None,
        resource_snapshots: dict[str, dict[str, Any]] | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Add an event to the audit trail."""
        if level.value > self.audit_level.value:
            return  # Skip events below configured audit level

        event_id = f"{self._current_test_id}_{len(self._events):04d}"

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(UTC),
            test_id=self._current_test_id or "unknown",
            component=component,
            message=message,
            level=level,
            duration_seconds=duration_seconds,
            metadata=metadata or {},
            resource_snapshots=resource_snapshots or {},
            errors=errors or [],
            warnings=warnings or [],
        )

        self._events.append(event)

        # Log event if appropriate level
        if level in [AuditLevel.BASIC, AuditLevel.DETAILED]:
            self.logger.info(f"[{component}] {message}")
        elif level == AuditLevel.DEBUG:
            self.logger.debug(f"[{component}] {message}")

    def _generate_audit_summary(
        self,
        test_id: str,
        start_time: datetime,
        end_time: datetime,
        total_duration: float,
        final_status: str,
    ) -> AuditSummary:
        """Generate audit summary from collected events."""
        cleanup_operations = len(
            [
                e
                for e in self._events
                if e.event_type
                in [
                    AuditEventType.CLEANUP_COMPLETED,
                    AuditEventType.CLEANUP_FAILED,
                ]
            ]
        )

        validation_operations = len(
            [
                e
                for e in self._events
                if e.event_type == AuditEventType.VALIDATION_COMPLETED
            ]
        )

        rollback_operations = len(
            [
                e
                for e in self._events
                if e.event_type
                in [
                    AuditEventType.ROLLBACK_TRIGGERED,
                    AuditEventType.ROLLBACK_COMPLETED,
                ]
            ]
        )

        errors_count = sum(len(e.errors) for e in self._events)
        warnings_count = sum(len(e.warnings) for e in self._events)

        resource_leaks_detected = len(
            [
                e
                for e in self._events
                if e.event_type == AuditEventType.RESOURCE_LEAK_DETECTED
            ]
        )

        threshold_violations = len(
            [
                e
                for e in self._events
                if e.event_type == AuditEventType.THRESHOLD_VIOLATION
            ]
        )

        # Generate recommendations based on events
        recommendations = self._generate_recommendations()

        return AuditSummary(
            test_id=test_id,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            events_count=len(self._events),
            cleanup_operations=cleanup_operations,
            validation_operations=validation_operations,
            rollback_operations=rollback_operations,
            errors_count=errors_count,
            warnings_count=warnings_count,
            resource_leaks_detected=resource_leaks_detected,
            threshold_violations=threshold_violations,
            final_status=final_status,
            recommendations=recommendations,
        )

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on audit events."""
        recommendations = []

        # Check for frequent resource leaks
        leak_events = [
            e
            for e in self._events
            if e.event_type == AuditEventType.RESOURCE_LEAK_DETECTED
        ]

        if len(leak_events) > 2:
            recommendations.append(
                "Consider reviewing cleanup procedures due to "
                "frequent resource leaks"
            )

        # Check for rollback frequency
        rollback_events = [
            e
            for e in self._events
            if e.event_type == AuditEventType.ROLLBACK_TRIGGERED
        ]

        if len(rollback_events) > 1:
            recommendations.append(
                "High rollback frequency detected - review test stability"
            )

        # Check for threshold violations
        violation_events = [
            e
            for e in self._events
            if e.event_type == AuditEventType.THRESHOLD_VIOLATION
        ]

        if len(violation_events) > 3:
            recommendations.append(
                "Multiple threshold violations detected - consider adjusting "
                "limits or improving performance"
            )

        return recommendations

    def _save_audit_trail(self, test_id: str, summary: AuditSummary) -> None:
        """Save audit trail to file."""
        try:
            # Create audit trail data
            audit_data = {
                "summary": asdict(summary),
                "events": [asdict(event) for event in self._events],
                "metadata": {
                    "audit_level": self.audit_level.value,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "generator_version": "1.0.0",
                },
            }

            # Save as JSON file
            audit_file = (
                self.audit_dir / f"audit_{test_id}_{int(time.time())}.json"
            )

            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(audit_data, f, indent=2, default=str)

            self.logger.info(f"Audit trail saved to {audit_file}")

        except Exception as e:
            self.logger.error(f"Failed to save audit trail for {test_id}: {e}")

    def get_audit_events(
        self, filter_by_type: AuditEventType | None = None
    ) -> list[AuditEvent]:
        """Get audit events, optionally filtered by type."""
        if filter_by_type:
            return [e for e in self._events if e.event_type == filter_by_type]
        return self._events.copy()

    def get_events_summary(self) -> dict[str, int]:
        """Get summary of events by type."""
        summary = {}
        for event in self._events:
            event_type = event.event_type.value
            summary[event_type] = summary.get(event_type, 0) + 1
        return summary
