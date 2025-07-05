"""Operations analysis for stakeholder reporting.

This module provides operations-level analysis and reporting capabilities
with monitoring, maintenance, and capacity planning insights.
"""

from typing import Any


class OperationsAnalyzer:
    """Provides operations-level analysis capabilities."""

    def assess_resource_health(self, data: dict[str, Any]) -> str:
        """Assess overall resource health status."""
        cleanup_data = data.get("resource_cleanup", {})
        cleanup_rate = cleanup_data.get("cleanup_effectiveness_rate", 0)

        # Check for resource leaks
        leak_count = len(cleanup_data.get("resource_leak_detection", []))

        if cleanup_rate >= 95 and leak_count == 0:
            return "excellent"
        elif cleanup_rate >= 90 and leak_count <= 1:
            return "good"
        elif cleanup_rate >= 85 and leak_count <= 2:
            return "acceptable"
        else:
            return "poor"

    def assess_cleanup_effectiveness(self, data: dict[str, Any]) -> float:
        """Assess cleanup effectiveness percentage."""
        cleanup_data = data.get("resource_cleanup", {})
        return cleanup_data.get("cleanup_effectiveness_rate", 0.0)

    def assess_concurrent_stability(self, data: dict[str, Any]) -> str:
        """Assess concurrent operations stability."""
        concurrent_data = data.get("concurrent_operations", {})
        stability_rate = concurrent_data.get("stability_rate", 0)
        race_conditions = concurrent_data.get("race_conditions_detected", 0)
        deadlocks = concurrent_data.get("deadlock_scenarios", 0)

        if stability_rate >= 99 and race_conditions == 0 and deadlocks == 0:
            return "excellent"
        elif stability_rate >= 95 and race_conditions <= 1 and deadlocks == 0:
            return "good"
        elif stability_rate >= 90 and race_conditions <= 2 and deadlocks <= 1:
            return "acceptable"
        else:
            return "poor"

    def identify_maintenance_needs(self, data: dict[str, Any]) -> list[str]:
        """Identify maintenance needs and recommendations."""
        maintenance_needs = []

        # Check resource cleanup issues
        cleanup_data = data.get("resource_cleanup", {})
        if cleanup_data.get("cleanup_effectiveness_rate", 100) < 95:
            maintenance_needs.append("Optimize resource cleanup procedures")

        # Check for memory leaks
        leaks = cleanup_data.get("resource_leak_detection", [])
        if leaks:
            maintenance_needs.append("Address detected memory leaks")

        # Check performance issues
        performance_data = data.get("performance_metrics", {})
        if not performance_data.get("page_load_compliance", True):
            maintenance_needs.append("Optimize page load performance")

        # Check concurrent operations
        concurrent_data = data.get("concurrent_operations", {})
        if concurrent_data.get("stability_rate", 100) < 95:
            maintenance_needs.append("Enhance concurrent operations stability")

        return maintenance_needs

    def generate_operational_alerts(self, data: dict[str, Any]) -> list[str]:
        """Generate operational alerts and warnings."""
        alerts = []

        # Critical resource issues
        cleanup_data = data.get("resource_cleanup", {})
        if cleanup_data.get("cleanup_effectiveness_rate", 100) < 90:
            alerts.append("CRITICAL: Resource cleanup effectiveness below 90%")

        # Performance alerts
        performance_data = data.get("performance_metrics", {})
        if not performance_data.get("page_load_compliance", True):
            alerts.append("WARNING: Page load times exceed SLA requirements")

        # Memory leak alerts
        leaks = cleanup_data.get("resource_leak_detection", [])
        if leaks:
            alerts.append(f"WARNING: {len(leaks)} memory leaks detected")

        return alerts

    def generate_maintenance_recommendations(
        self, data: dict[str, Any]
    ) -> list[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        # Resource optimization recommendations
        cleanup_data = data.get("resource_cleanup", {})
        if cleanup_data.get("cleanup_effectiveness_rate", 100) < 95:
            recommendations.append("Implement enhanced resource monitoring")
            recommendations.append(
                "Schedule regular cleanup optimization reviews"
            )

        # Performance recommendations
        performance_data = data.get("performance_metrics", {})
        bottlenecks = performance_data.get("bottleneck_analysis", {}).get(
            "identified_bottlenecks", []
        )
        if bottlenecks:
            recommendations.append(
                "Address identified performance bottlenecks"
            )

        # Monitoring recommendations
        recommendations.extend(
            [
                "Implement continuous monitoring dashboards",
                "Set up automated alerting for resource thresholds",
                "Schedule regular performance health checks",
            ]
        )

        return recommendations

    def analyze_capacity_requirements(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze capacity requirements and scaling needs."""
        performance_data = data.get("performance_metrics", {})
        concurrent_data = data.get("concurrent_operations", {})

        resource_util = performance_data.get("resource_utilization", {})
        max_concurrent = concurrent_data.get("max_concurrent_users", 0)

        return {
            "current_capacity": max_concurrent,
            "cpu_utilization_peak": resource_util.get("cpu_usage_peak", 0),
            "memory_utilization_peak": resource_util.get(
                "memory_usage_peak", 0
            ),
            "scaling_recommendations": [
                "Monitor resource utilization trends",
                "Plan for 2x capacity buffer",
                "Implement auto-scaling policies",
            ],
            "capacity_planning_horizon": "6_months",
        }

    def assess_sla_compliance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Assess SLA compliance across all metrics."""
        performance_data = data.get("performance_metrics", {})
        cleanup_data = data.get("resource_cleanup", {})

        return {
            "page_load_sla": performance_data.get(
                "page_load_compliance", False
            ),
            "config_validation_sla": performance_data.get(
                "config_validation_compliance", False
            ),
            "cleanup_time_sla": cleanup_data.get(
                "cleanup_time_compliance", False
            ),
            "overall_compliance": self._calculate_overall_sla_compliance(data),
            "sla_violations": self._identify_sla_violations(data),
        }

    def _calculate_overall_sla_compliance(self, data: dict[str, Any]) -> float:
        """Calculate overall SLA compliance percentage."""
        performance_data = data.get("performance_metrics", {})
        cleanup_data = data.get("resource_cleanup", {})

        compliance_metrics = [
            performance_data.get("page_load_compliance", False),
            performance_data.get("config_validation_compliance", False),
            cleanup_data.get("cleanup_time_compliance", False),
        ]

        compliant_count = sum(1 for metric in compliance_metrics if metric)
        total_count = len(compliance_metrics)

        return (
            (compliant_count / total_count * 100) if total_count > 0 else 0.0
        )

    def _identify_sla_violations(self, data: dict[str, Any]) -> list[str]:
        """Identify specific SLA violations."""
        violations = []

        performance_data = data.get("performance_metrics", {})
        cleanup_data = data.get("resource_cleanup", {})

        if not performance_data.get("page_load_compliance", True):
            violations.append("Page load time exceeds 2-second SLA")

        if not performance_data.get("config_validation_compliance", True):
            violations.append("Config validation exceeds 500ms SLA")

        if not cleanup_data.get("cleanup_time_compliance", True):
            violations.append("Resource cleanup exceeds 30-second SLA")

        return violations
