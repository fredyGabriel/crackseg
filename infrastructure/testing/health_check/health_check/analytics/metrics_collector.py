"""System metrics collection."""

from typing import Any


class MetricsCollector:
    """System-wide metrics collection and analysis."""

    def collect_system_metrics(
        self, service_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect system-wide metrics.

        Args:
            service_results: Health check results

        Returns:
            Dictionary of system metrics
        """
        # Import locally to avoid module resolution issues
        from ..models import HealthStatus

        total_services = len(service_results)
        healthy_services = sum(
            1
            for r in service_results.values()
            if r.status == HealthStatus.HEALTHY
        )
        unhealthy_services = sum(
            1
            for r in service_results.values()
            if r.status == HealthStatus.UNHEALTHY
        )
        starting_services = sum(
            1
            for r in service_results.values()
            if r.status == HealthStatus.STARTING
        )

        if total_services > 0:
            avg_response_time = (
                sum(r.response_time for r in service_results.values())
                / total_services
            )
            max_response_time = max(
                r.response_time for r in service_results.values()
            )
        else:
            avg_response_time = 0.0
            max_response_time = 0.0

        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": unhealthy_services,
            "starting_services": starting_services,
            "health_percentage": (
                (healthy_services / total_services) * 100
                if total_services > 0
                else 0
            ),
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "last_check_duration": sum(
                r.response_time for r in service_results.values()
            ),
        }
