"""Main health check orchestration logic."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, cast

from ..analytics import (
    DashboardGenerator,
    MetricsCollector,
    RecommendationEngine,
)
from ..checkers import DependencyValidator, DockerChecker, EndpointChecker
from ..models import (
    HealthCheckResult,
    HealthStatus,
    ServiceConfig,
    SystemHealthReport,
)


class HealthOrchestrator:
    """Main orchestrator for health check operations."""

    def __init__(self) -> None:
        """Initialize health orchestrator."""
        self.logger = logging.getLogger("health_orchestrator")

        # Initialize components
        self.docker_checker = DockerChecker()
        self.endpoint_checker = EndpointChecker()
        self.dependency_validator = DependencyValidator()
        self.metrics_collector = MetricsCollector()
        self.recommendation_engine = RecommendationEngine()
        self.dashboard_generator = DashboardGenerator()

        # Health check history
        self.health_history: list[SystemHealthReport] = []

    async def check_service_health(
        self, service: ServiceConfig
    ) -> HealthCheckResult:
        """
        Check health of a single service. Args: service: Service configuration
        to check Returns: Health check result with status and metrics
        """
        start_time = time.time()

        try:
            # First check Docker container status
            docker_status = await self.docker_checker.check_container_status(
                service.container_name
            )

            if docker_status != HealthStatus.HEALTHY:
                return HealthCheckResult(
                    service_name=service.name,
                    status=docker_status,
                    response_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details={"docker_status": docker_status.value},
                    error_message=(
                        f"Docker container status: {docker_status.value}"
                    ),
                )

            # Check service endpoint
            endpoint_result = (
                await self.endpoint_checker.check_service_endpoint(service)
            )

            # Combine results
            response_time = time.time() - start_time

            return HealthCheckResult(
                service_name=service.name,
                status=endpoint_result.get("status", HealthStatus.UNKNOWN),
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "docker_status": docker_status.value,
                    "endpoint_response": endpoint_result,
                    "response_time_ms": int(response_time * 1000),
                },
            )

        except Exception as e:
            self.logger.error(
                "Health check failed for %s: %s", service.name, e
            )
            return HealthCheckResult(
                service_name=service.name,
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                details={"error": str(e)},
                error_message=str(e),
            )

    async def check_all_services(
        self, services: dict[str, ServiceConfig]
    ) -> SystemHealthReport:
        """
        Check health of all configured services. Args: services: Dictionary of
        service configurations Returns: Comprehensive system health report
        """
        self.logger.info("Starting health check for all services...")

        # Execute health checks concurrently with explicit type annotation
        tasks: list[asyncio.Task[HealthCheckResult]] = []
        for service in services.values():
            task = asyncio.create_task(self.check_service_health(service))
            tasks.append(task)

        results: list[HealthCheckResult | BaseException] = (
            await asyncio.gather(*tasks, return_exceptions=True)
        )

        # Process results with explicit type annotations
        service_results: dict[str, HealthCheckResult] = {}
        critical_failures: list[str] = []

        for i, result in enumerate(results):
            service = list(services.values())[i]

            if isinstance(result, Exception):
                service_results[service.name] = HealthCheckResult(
                    service_name=service.name,
                    status=HealthStatus.CRITICAL,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    details={"exception": str(result)},
                    error_message=str(result),
                )
                if service.critical:
                    critical_failures.append(service.name)
            else:
                # result is HealthCheckResult at this point
                health_result = cast(HealthCheckResult, result)
                service_results[service.name] = health_result
                if (
                    health_result.status
                    in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
                    and service.critical
                ):
                    critical_failures.append(service.name)

        # Check dependencies
        dependencies_satisfied = (
            self.dependency_validator.validate_dependencies(
                service_results, services
            )
        )

        # Determine overall status
        overall_status = self._determine_overall_status(
            service_results, dependencies_satisfied, critical_failures
        )

        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            service_results, services, dependencies_satisfied
        )

        # Collect metrics
        metrics = self.metrics_collector.collect_system_metrics(
            service_results
        )

        report = SystemHealthReport(
            timestamp=datetime.now(),
            overall_status=overall_status,
            services=service_results,
            dependencies_satisfied=dependencies_satisfied,
            recommendations=recommendations,
            metrics=metrics,
        )

        # Store in history
        self.health_history.append(report)

        # Keep only last 100 reports
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]

        self.logger.info(
            "Health check completed. Overall status: %s", overall_status.value
        )

        return report

    def _determine_overall_status(
        self,
        service_results: dict[str, HealthCheckResult],
        dependencies_satisfied: bool,
        critical_failures: list[str],
    ) -> HealthStatus:
        """
        Determine overall system health status. Args: service_results: Results
        from individual service checks dependencies_satisfied: Whether
        dependencies are satisfied critical_failures: List of critical service
        failures Returns: Overall system health status
        """
        if critical_failures:
            return HealthStatus.CRITICAL
        elif not dependencies_satisfied:
            return HealthStatus.UNHEALTHY
        elif any(
            r.status == HealthStatus.STARTING for r in service_results.values()
        ):
            return HealthStatus.STARTING
        elif all(
            r.status == HealthStatus.HEALTHY for r in service_results.values()
        ):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNHEALTHY

    def get_health_history(self) -> list[SystemHealthReport]:
        """
        Get health check history. Returns: List of historical health reports
        """
        return self.health_history.copy()

    def generate_dashboard_data(self) -> dict[str, Any]:
        """
        Generate data for monitoring dashboard. Returns: Dashboard data
        structure
        """
        return self.dashboard_generator.generate_dashboard_data(
            self.health_history
        )
