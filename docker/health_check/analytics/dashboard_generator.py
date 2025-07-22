"""Dashboard data generation."""

from typing import Any


class DashboardGenerator:
    """Generate data for monitoring dashboard."""

    def generate_dashboard_data(
        self, health_history: list[Any]
    ) -> dict[str, Any]:
        """Generate data for monitoring dashboard.

        Args:
            health_history: List of historical health reports.

        Returns:
            Dashboard data structure
        """
        if not health_history:
            return {"error": "No health check data available"}

        latest_report = health_history[-1]

        # Prepare service statuses
        service_statuses = {}
        for name, result in latest_report.services.items():
            service_statuses[name] = {
                "status": result.status.value,
                "response_time": result.response_time,
                "last_check": result.timestamp.isoformat(),
                "details": result.details,
                "error": result.error_message,
            }

        # Historical data for trends (last 10 checks)
        historical_data: list[dict[str, Any]] = []
        for report in health_history[-10:]:
            historical_data.append(
                {
                    "timestamp": report.timestamp.isoformat(),
                    "overall_status": report.overall_status.value,
                    "health_percentage": report.metrics.get(
                        "health_percentage", 0
                    ),
                    "avg_response_time": report.metrics.get(
                        "avg_response_time", 0
                    ),
                }
            )

        return {
            "timestamp": latest_report.timestamp.isoformat(),
            "overall_status": latest_report.overall_status.value,
            "services": service_statuses,
            "dependencies_satisfied": latest_report.dependencies_satisfied,
            "recommendations": latest_report.recommendations,
            "metrics": latest_report.metrics,
            "historical_data": historical_data,
        }
