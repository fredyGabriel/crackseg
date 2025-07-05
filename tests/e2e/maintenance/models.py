"""Data models and types for the test maintenance framework.

This module defines all the data structures used throughout the maintenance
framework, including health status, maintenance reports, review results,
and configuration types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict


class HealthStatus(Enum):
    """Overall health status of the test suite."""

    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MaintenanceMode(Enum):
    """Different modes for maintenance operations."""

    PASSIVE = "passive"  # Monitor only, no automatic actions
    STANDARD = "standard"  # Basic automated maintenance
    AGGRESSIVE = "aggressive"  # Full optimization and cleanup
    CUSTOM = "custom"  # User-defined maintenance rules


class ReviewFrequency(Enum):
    """Frequency options for automated review cycles."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class OptimizationLevel(Enum):
    """Levels of performance optimization."""

    LIGHT = "light"  # Minimal changes, safe optimizations
    MODERATE = "moderate"  # Balanced optimization approach
    AGGRESSIVE = "aggressive"  # Maximum optimization, may affect stability


class TestHealthMetric(TypedDict):
    """Type definition for individual health metrics."""

    metric_name: str
    current_value: float
    threshold_value: float
    status: str
    trend: str
    last_updated: datetime


class PerformanceTrend(TypedDict):
    """Type definition for performance trend data."""

    metric_name: str
    trend_direction: str  # "improving", "degrading", "stable"
    change_percentage: float
    data_points: list[dict[str, Any]]
    recommendation: str


@dataclass
class TestSuiteHealthReport:
    """Comprehensive health report for the test suite."""

    overall_health: HealthStatus
    timestamp: datetime
    metrics: list[TestHealthMetric] = field(default_factory=list)
    performance_trends: list[PerformanceTrend] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    requires_maintenance: bool = False

    @property
    def critical_issues_count(self) -> int:
        """Count of critical issues requiring immediate attention."""
        return len(
            [issue for issue in self.issues if "critical" in issue.lower()]
        )

    @property
    def performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        if not self.performance_trends:
            return 50.0  # Neutral score

        improving_count = sum(
            1
            for trend in self.performance_trends
            if trend["trend_direction"] == "improving"
        )

        total_trends = len(self.performance_trends)
        return (improving_count / total_trends) * 100


@dataclass
class ReviewResult:
    """Result from an automated review cycle."""

    review_id: str
    timestamp: datetime
    review_type: str
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    severity: str = "low"
    estimated_effort_hours: float = 0.0

    @property
    def requires_immediate_action(self) -> bool:
        """Check if review requires immediate action."""
        return self.severity in ["high", "critical"]


@dataclass
class MaintenanceAction:
    """Individual maintenance action to be performed."""

    action_id: str
    action_type: str
    description: str
    estimated_duration: float
    risk_level: str = "low"
    dependencies: list[str] = field(default_factory=list)
    automated: bool = True

    @property
    def is_safe_to_automate(self) -> bool:
        """Check if action is safe for automation."""
        return self.risk_level in ["low", "medium"] and self.automated


@dataclass
class MaintenanceReport:
    """Comprehensive report from a maintenance cycle."""

    cycle_id: str
    start_time: datetime
    end_time: datetime | None = None
    actions_performed: list[MaintenanceAction] = field(default_factory=list)
    actions_skipped: list[MaintenanceAction] = field(default_factory=list)
    health_before: TestSuiteHealthReport | None = None
    health_after: TestSuiteHealthReport | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        """Calculate total maintenance duration in seconds."""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Calculate success rate of maintenance actions."""
        total_actions = len(self.actions_performed) + len(self.actions_skipped)
        if total_actions == 0:
            return 100.0
        return (len(self.actions_performed) / total_actions) * 100

    @property
    def health_improvement(self) -> float:
        """Calculate health improvement percentage."""
        if not self.health_before or not self.health_after:
            return 0.0

        before_score = self.health_before.performance_score
        after_score = self.health_after.performance_score

        if before_score == 0:
            return 100.0 if after_score > 0 else 0.0

        return ((after_score - before_score) / before_score) * 100


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""

    recommendation_id: str
    category: str
    description: str
    expected_improvement: str
    implementation_effort: str
    risk_assessment: str
    code_locations: list[str] = field(default_factory=list)
    automated_fix_available: bool = False

    @property
    def priority_score(self) -> int:
        """Calculate priority score (1-10, higher is more urgent)."""
        effort_scores = {"low": 3, "medium": 2, "high": 1}
        risk_scores = {"low": 3, "medium": 2, "high": 1}

        effort_score = effort_scores.get(self.implementation_effort.lower(), 1)
        risk_score = risk_scores.get(self.risk_assessment.lower(), 1)
        automation_bonus = 2 if self.automated_fix_available else 0

        return min(10, effort_score + risk_score + automation_bonus)


class MaintenanceScheduleEntry(TypedDict):
    """Type definition for maintenance schedule entries."""

    schedule_id: str
    maintenance_type: str
    frequency: str
    next_execution: datetime
    last_execution: datetime | None
    enabled: bool
    configuration: dict[str, Any]
