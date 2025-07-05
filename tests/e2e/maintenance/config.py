"""Configuration for the test maintenance framework.

This module provides configurable settings for all aspects of test maintenance,
including health monitoring thresholds, maintenance schedules, optimization
levels, and integration settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.e2e.maintenance.models import (
    MaintenanceMode,
    OptimizationLevel,
    ReviewFrequency,
)


@dataclass
class HealthMonitoringConfig:
    """Configuration for test suite health monitoring."""

    # Performance thresholds
    max_average_test_duration: float = 30.0  # seconds
    max_memory_usage_mb: float = 512.0
    min_success_rate: float = 95.0  # percentage
    max_failure_streak: int = 3

    # Trend monitoring
    performance_degradation_threshold: float = 10.0  # percentage
    trend_analysis_days: int = 7
    alert_on_consecutive_failures: int = 2

    # Resource monitoring
    max_cpu_usage: float = 80.0  # percentage
    max_disk_usage_gb: float = 5.0
    check_interval_minutes: int = 15

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns:
            List of validation errors, empty if valid
        """
        errors = []

        if self.max_average_test_duration <= 0:
            errors.append("max_average_test_duration must be positive")

        if not 0 <= self.min_success_rate <= 100:
            errors.append("min_success_rate must be between 0 and 100")

        if self.max_failure_streak < 1:
            errors.append("max_failure_streak must be at least 1")

        if self.performance_degradation_threshold <= 0:
            errors.append("performance_degradation_threshold must be positive")

        return errors


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""

    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    enabled_optimizations: list[str] = field(
        default_factory=lambda: [
            "cleanup_artifacts",
            "optimize_parallelization",
            "update_test_timeouts",
            "consolidate_duplicate_tests",
        ]
    )

    # Safety settings
    backup_before_optimization: bool = True
    max_optimization_duration: float = 300.0  # seconds
    rollback_on_failure: bool = True

    # Performance targets
    target_test_duration_reduction: float = 20.0  # percentage
    target_memory_reduction: float = 15.0  # percentage
    target_parallelization_improvement: float = 25.0  # percentage

    def get_optimization_settings(self) -> dict[str, Any]:
        """Get optimization settings based on level.

        Returns:
            Dictionary of optimization settings
        """
        base_settings = {
            "backup_enabled": self.backup_before_optimization,
            "rollback_enabled": self.rollback_on_failure,
            "max_duration": self.max_optimization_duration,
        }

        if self.optimization_level == OptimizationLevel.LIGHT:
            return {
                **base_settings,
                "aggressive_cleanup": False,
                "modify_test_structure": False,
                "parallel_optimization": True,
                "timeout_adjustments": True,
            }
        elif self.optimization_level == OptimizationLevel.MODERATE:
            return {
                **base_settings,
                "aggressive_cleanup": True,
                "modify_test_structure": True,
                "parallel_optimization": True,
                "timeout_adjustments": True,
            }
        else:  # AGGRESSIVE
            return {
                **base_settings,
                "aggressive_cleanup": True,
                "modify_test_structure": True,
                "parallel_optimization": True,
                "timeout_adjustments": True,
                "experimental_optimizations": True,
            }


@dataclass
class ReviewConfig:
    """Configuration for automated review cycles."""

    frequency: ReviewFrequency = ReviewFrequency.WEEKLY
    enabled_reviews: list[str] = field(
        default_factory=lambda: [
            "code_quality",
            "test_coverage",
            "performance_regression",
            "maintenance_needed",
        ]
    )

    # Review criteria
    minimum_test_coverage: float = 80.0  # percentage
    code_quality_threshold: float = 8.0  # out of 10
    performance_regression_threshold: float = 15.0  # percentage

    # Notification settings
    notify_on_critical_issues: bool = True
    notification_email: str | None = None
    create_maintenance_tickets: bool = True

    def get_review_schedule(self) -> dict[str, int]:
        """Get review schedule in hours.

        Returns:
            Dictionary mapping review types to interval hours
        """
        base_intervals = {
            ReviewFrequency.DAILY: 24,
            ReviewFrequency.WEEKLY: 168,  # 7 days
            ReviewFrequency.MONTHLY: 720,  # 30 days
        }

        base_interval = base_intervals.get(self.frequency, 168)

        return {
            "code_quality": base_interval,
            "test_coverage": base_interval * 2,  # Less frequent
            "performance_regression": base_interval // 2,  # More frequent
            "maintenance_needed": base_interval,
        }


@dataclass
class SchedulingConfig:
    """Configuration for maintenance scheduling."""

    # Execution windows
    preferred_execution_hours: list[int] = field(
        default_factory=lambda: [2, 3, 4]  # 2-4 AM
    )
    avoid_business_hours: bool = True
    max_concurrent_operations: int = 2

    # Scheduling rules
    minimum_interval_hours: int = 24
    emergency_maintenance_enabled: bool = True
    maintenance_window_duration: int = 120  # minutes

    # Resource constraints
    max_resource_usage_during_maintenance: float = 50.0  # percentage
    pause_on_active_tests: bool = True


@dataclass
class IntegrationConfig:
    """Configuration for integration with existing systems."""

    # Reporting integration
    use_existing_reporting: bool = True
    reporting_output_dir: Path = Path("test-results/maintenance")
    generate_html_reports: bool = True
    generate_json_reports: bool = True

    # Performance monitoring integration
    integrate_performance_monitoring: bool = True
    performance_data_retention_days: int = 30

    # Capture system integration
    capture_maintenance_evidence: bool = True
    capture_before_after_screenshots: bool = True
    video_recording_enabled: bool = False

    # External tool integration
    slack_webhook_url: str | None = None
    jira_integration_enabled: bool = False
    github_integration_enabled: bool = False


@dataclass
class MaintenanceConfig:
    """Main configuration for the test maintenance framework."""

    mode: MaintenanceMode = MaintenanceMode.STANDARD

    # Sub-configurations
    health_monitoring: HealthMonitoringConfig = field(
        default_factory=HealthMonitoringConfig
    )
    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )
    review: ReviewConfig = field(default_factory=ReviewConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)

    # General settings
    enable_automated_maintenance: bool = True
    maintenance_data_dir: Path = Path(".maintenance")
    log_level: str = "INFO"
    dry_run_mode: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.maintenance_data_dir.mkdir(parents=True, exist_ok=True)

        # Validate sub-configurations
        health_errors = self.health_monitoring.validate()
        if health_errors:
            raise ValueError(
                f"Health monitoring config errors: {health_errors}"
            )

    @classmethod
    def for_ci_environment(cls) -> "MaintenanceConfig":
        """Create configuration optimized for CI environments.

        Returns:
            MaintenanceConfig tuned for CI/CD pipelines
        """
        return cls(
            mode=MaintenanceMode.PASSIVE,
            health_monitoring=HealthMonitoringConfig(
                check_interval_minutes=5,
                alert_on_consecutive_failures=1,
            ),
            optimization=OptimizationConfig(
                optimization_level=OptimizationLevel.LIGHT,
                backup_before_optimization=False,
            ),
            review=ReviewConfig(
                frequency=ReviewFrequency.DAILY,
                notify_on_critical_issues=False,
            ),
            scheduling=SchedulingConfig(
                avoid_business_hours=False,
                emergency_maintenance_enabled=False,
            ),
        )

    @classmethod
    def for_development(cls) -> "MaintenanceConfig":
        """Create configuration for development environments.

        Returns:
            MaintenanceConfig tuned for development use
        """
        return cls(
            mode=MaintenanceMode.AGGRESSIVE,
            health_monitoring=HealthMonitoringConfig(
                check_interval_minutes=30,
                max_failure_streak=5,
            ),
            optimization=OptimizationConfig(
                optimization_level=OptimizationLevel.AGGRESSIVE,
                backup_before_optimization=True,
            ),
            review=ReviewConfig(
                frequency=ReviewFrequency.DAILY,
            ),
            dry_run_mode=True,  # Safe for development
        )
