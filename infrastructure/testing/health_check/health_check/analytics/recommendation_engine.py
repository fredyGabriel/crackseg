"""System recommendations generation."""

from typing import Any


class RecommendationEngine:
    """Generate recommendations based on health check results."""

    def generate_recommendations(
        self,
        service_results: dict[str, Any],
        services: dict[str, Any],
        dependencies_satisfied: bool,
    ) -> list[str]:
        """Generate recommendations based on health check results.

        Args:
            service_results: Health check results
            services: Service configurations
            dependencies_satisfied: Whether dependencies are satisfied

        Returns:
            List of recommendations
        """
        # Import locally to avoid module resolution issues
        from ..models import HealthStatus

        recommendations: list[str] = []

        # Check for critical failures
        critical_failures = [
            name
            for name, result in service_results.items()
            if result.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]
            and services[name].critical
        ]

        if critical_failures:
            recommendations.append(
                f"Critical services failed: {', '.join(critical_failures)}. "
                "Consider restarting these services."
            )

        # Check for dependency issues
        if not dependencies_satisfied:
            recommendations.append(
                "Service dependencies not satisfied. "
                "Check service startup order."
            )

        # Check for high response times
        slow_services = [
            name
            for name, result in service_results.items()
            if result.response_time > 5.0  # 5 seconds threshold
        ]

        if slow_services:
            recommendations.append(
                f"High response times detected: {', '.join(slow_services)}. "
                "Monitor resource usage."
            )

        # Check for services in starting state for too long
        starting_services = [
            name
            for name, result in service_results.items()
            if result.status == HealthStatus.STARTING
        ]

        if starting_services:
            recommendations.append(
                f"Services still starting: {', '.join(starting_services)}. "
                "Allow more time or check logs for issues."
            )

        return recommendations
