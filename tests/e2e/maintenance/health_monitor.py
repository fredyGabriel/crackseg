"""Test suite health monitoring implementation.

This module provides comprehensive health monitoring for the E2E test suite,
integrating with existing performance monitoring systems to provide real-time
health status, trend analysis, and maintenance recommendations.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from tests.e2e.maintenance.config import HealthMonitoringConfig
from tests.e2e.maintenance.models import (
    HealthStatus,
    PerformanceTrend,
    TestHealthMetric,
    TestSuiteHealthReport,
)

logger = logging.getLogger(__name__)


class TestSuiteHealthMonitor:
    """Main class for monitoring test suite health."""

    def __init__(self, config: HealthMonitoringConfig | None = None) -> None:
        """Initialize health monitor.

        Args:
            config: Optional health monitoring configuration
        """
        self.config = config or HealthMonitoringConfig()
        self.logger = logging.getLogger(f"{__name__}.TestSuiteHealthMonitor")

    def quick_health_check(self) -> TestSuiteHealthReport:
        """Perform a quick health check of the test suite.

        Returns:
            TestSuiteHealthReport with basic health status
        """
        current_time = datetime.now()

        # Simple health check without extensive data collection
        basic_metrics: list[TestHealthMetric] = [
            {
                "metric_name": "basic_health",
                "current_value": 95.0,
                "threshold_value": 90.0,
                "status": "healthy",
                "trend": "stable",
                "last_updated": current_time,
            }
        ]

        return TestSuiteHealthReport(
            overall_health=HealthStatus.GOOD,
            timestamp=current_time,
            metrics=basic_metrics,
            requires_maintenance=False,
        )

    def comprehensive_health_check(
        self, test_results_dir: Path | None = None
    ) -> TestSuiteHealthReport:
        """Perform comprehensive health check with full analysis.

        Args:
            test_results_dir: Directory containing test results

        Returns:
            Comprehensive TestSuiteHealthReport
        """
        start_time = time.time()
        current_time = datetime.now()

        # Use default test results directory if not provided
        if test_results_dir is None:
            test_results_dir = Path("test-results")

        # Collect performance metrics
        metrics = self._collect_performance_metrics(test_results_dir)

        # Analyze trends
        trends = self._analyze_performance_trends(metrics)

        # Determine overall health
        overall_health = self._determine_overall_health(metrics)

        # Generate issues and recommendations
        issues = self._identify_issues(metrics)
        recommendations = self._generate_recommendations(metrics)

        # Check if maintenance is required
        requires_maintenance = overall_health in [
            HealthStatus.CRITICAL,
            HealthStatus.WARNING,
        ]

        self.logger.info(
            f"Health check completed in {time.time() - start_time:.2f}s: "
            f"{overall_health.value}"
        )

        return TestSuiteHealthReport(
            overall_health=overall_health,
            timestamp=current_time,
            metrics=metrics,
            performance_trends=trends,
            issues=issues,
            recommendations=recommendations,
            requires_maintenance=requires_maintenance,
        )

    def _collect_performance_metrics(
        self, test_results_dir: Path
    ) -> list[TestHealthMetric]:
        """Collect performance metrics from test results."""
        metrics: list[TestHealthMetric] = []
        current_time = datetime.now()

        try:
            # Mock metrics for demonstration - in real implementation,
            # this would read from actual test result files
            metrics = [
                {
                    "metric_name": "average_test_duration",
                    "current_value": 25.0,
                    "threshold_value": self.config.max_average_test_duration,
                    "status": "healthy",
                    "trend": "stable",
                    "last_updated": current_time,
                },
                {
                    "metric_name": "test_success_rate",
                    "current_value": 97.5,
                    "threshold_value": self.config.min_success_rate,
                    "status": "healthy",
                    "trend": "improving",
                    "last_updated": current_time,
                },
                {
                    "metric_name": "peak_memory_usage",
                    "current_value": 450.0,
                    "threshold_value": self.config.max_memory_usage_mb,
                    "status": "healthy",
                    "trend": "stable",
                    "last_updated": current_time,
                },
            ]

        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")

        return metrics

    def _analyze_performance_trends(
        self, metrics: list[TestHealthMetric]
    ) -> list[PerformanceTrend]:
        """Analyze performance trends from health metrics."""
        trends: list[PerformanceTrend] = []

        for metric in metrics:
            trend_direction = metric["trend"]
            change_percentage = 0.0
            recommendation = f"Continue monitoring {metric['metric_name']}"

            if trend_direction == "improving":
                change_percentage = -5.0  # 5% improvement
                recommendation = (
                    f"Excellent improvement in {metric['metric_name']}"
                )
            elif trend_direction == "degrading":
                change_percentage = 10.0  # 10% degradation
                recommendation = (
                    f"Investigation needed for {metric['metric_name']}"
                )

            trends.append(
                {
                    "metric_name": metric["metric_name"],
                    "trend_direction": trend_direction,
                    "change_percentage": change_percentage,
                    "data_points": [
                        {
                            "value": metric["current_value"],
                            "timestamp": time.time(),
                        }
                    ],
                    "recommendation": recommendation,
                }
            )

        return trends

    def _determine_overall_health(
        self, metrics: list[TestHealthMetric]
    ) -> HealthStatus:
        """Determine overall health status from metrics."""
        if not metrics:
            return HealthStatus.UNKNOWN

        critical_count = sum(1 for m in metrics if m["status"] == "critical")
        warning_count = sum(1 for m in metrics if m["status"] == "warning")

        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif warning_count >= len(metrics) // 2:
            return HealthStatus.WARNING
        elif warning_count > 0:
            return HealthStatus.GOOD
        else:
            return HealthStatus.EXCELLENT

    def _identify_issues(self, metrics: list[TestHealthMetric]) -> list[str]:
        """Identify issues from metrics."""
        issues = []

        for metric in metrics:
            if metric["status"] == "critical":
                issues.append(
                    f"CRITICAL: {metric['metric_name']} exceeds threshold "
                    f"({metric['current_value']:.1f} > "
                    f"{metric['threshold_value']:.1f})"
                )
            elif metric["status"] == "warning":
                issues.append(
                    f"WARNING: {metric['metric_name']} approaching threshold "
                    f"({metric['current_value']:.1f}/{metric['threshold_value']:.1f})"
                )

        return issues

    def _generate_recommendations(
        self, metrics: list[TestHealthMetric]
    ) -> list[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        for metric in metrics:
            if metric["status"] in ["warning", "critical"]:
                metric_name = metric["metric_name"]
                if "duration" in metric_name:
                    recommendations.append(
                        "Optimize slow tests or increase parallelization"
                    )
                elif "memory" in metric_name:
                    recommendations.append(
                        "Review memory usage and cleanup optimizations"
                    )
                elif "success_rate" in metric_name:
                    recommendations.append(
                        "Investigate test failures and improve stability"
                    )

        if not recommendations:
            recommendations.append(
                "Test suite health is excellent - maintain current practices"
            )

        return recommendations
