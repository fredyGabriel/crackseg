#!/usr/bin/env python3
"""CrackSeg Health Check System.

Comprehensive health check system for Docker services combining native
Docker health checks with detailed custom monitoring and observability.

This module provides:
- Service health monitoring for all Docker services
- Dependency validation between services
- Custom metrics collection and reporting
- Dashboard and logging capabilities
- Automatic restart mechanisms coordination

Author: CrackSeg Project
Version: 1.0 (Subtask 13.7)
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

import click
import requests

# =============================================================================
# Configuration and Data Models
# =============================================================================


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    UNKNOWN = "unknown"
    CRITICAL = "critical"


@dataclass
class ServiceConfig:
    """Configuration for a monitored service."""

    name: str
    container_name: str
    health_endpoint: str
    port: int
    timeout: int = 10
    retries: int = 3
    dependencies: list[str] | None = None
    critical: bool = True
    networks: list[str] | None = None
    service_discovery: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Initialize dependencies list if None."""
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    service_name: str
    status: HealthStatus
    response_time: float
    timestamp: datetime
    details: dict[str, Any]
    error_message: str | None = None


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""

    timestamp: datetime
    overall_status: HealthStatus
    services: dict[str, HealthCheckResult]
    dependencies_satisfied: bool
    recommendations: list[str]
    metrics: dict[str, Any]


# =============================================================================
# Health Check System Core
# =============================================================================


class HealthCheckSystem:
    """Comprehensive health check system for CrackSeg Docker infrastructure."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize health check system.

        Args:
            config_path: Path to health check configuration file
        """
        self.logger = self._setup_logging()
        self.config_path = (
            config_path or Path(__file__).parent / "health_config.json"
        )
        self.services: dict[str, ServiceConfig] = {}
        self.health_history: list[SystemHealthReport] = []
        self.monitoring_active = False

        # Load configuration
        self._load_configuration()

        self.logger.info(
            "HealthCheckSystem initialized with %d services",
            len(self.services),
        )

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("health_check_system")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_configuration(self) -> None:
        """Load service configuration from file."""
        default_config: dict[str, dict[str, Any]] = {
            "streamlit-app": {
                "name": "streamlit-app",
                "container_name": "crackseg-streamlit-app",
                "health_endpoint": "http://streamlit:8501/_stcore/health",
                "port": 8501,
                "timeout": 15,
                "retries": 3,
                "dependencies": [],
                "critical": True,
                "networks": [
                    "crackseg-frontend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "frontend": "streamlit:8501",
                    "management": "streamlit-mgmt:8501",
                },
            },
            "selenium-hub": {
                "name": "selenium-hub",
                "container_name": "crackseg-selenium-hub",
                "health_endpoint": "http://hub:4444/wd/hub/status",
                "port": 4444,
                "timeout": 10,
                "retries": 3,
                "dependencies": [],
                "critical": True,
                "networks": [
                    "crackseg-backend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "backend": "hub:4444",
                    "management": "hub-mgmt:4444",
                },
            },
            "chrome-node": {
                "name": "chrome-node",
                "container_name": "crackseg-chrome-node",
                "health_endpoint": "http://chrome:5555/wd/hub/status",
                "port": 5555,
                "timeout": 10,
                "retries": 3,
                "dependencies": ["selenium-hub"],
                "critical": True,
                "networks": [
                    "crackseg-backend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "backend": "chrome:5555",
                    "management": "chrome-mgmt:5555",
                },
            },
            "firefox-node": {
                "name": "firefox-node",
                "container_name": "crackseg-firefox-node",
                "health_endpoint": "http://firefox:5555/wd/hub/status",
                "port": 5555,
                "timeout": 10,
                "retries": 3,
                "dependencies": ["selenium-hub"],
                "critical": True,
                "networks": [
                    "crackseg-backend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "backend": "firefox:5555",
                    "management": "firefox-mgmt:5555",
                },
            },
            "test-runner": {
                "name": "test-runner",
                "container_name": "crackseg-test-runner",
                "health_endpoint": "http://test-runner:8080/health",
                "port": 8080,
                "timeout": 5,
                "retries": 2,
                "dependencies": [
                    "streamlit-app",
                    "selenium-hub",
                    "chrome-node",
                ],
                "critical": False,
                "networks": [
                    "crackseg-frontend-network",
                    "crackseg-backend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "frontend": "test-runner-frontend:8080",
                    "backend": "test-runner:8080",
                    "management": "test-runner-mgmt:8080",
                },
            },
            "grid-console": {
                "name": "grid-console",
                "container_name": "crackseg-grid-console",
                "health_endpoint": "http://console:4444/grid/api/hub/status",
                "port": 4444,
                "timeout": 10,
                "retries": 3,
                "dependencies": ["selenium-hub"],
                "critical": False,
                "networks": ["crackseg-management-network"],
                "service_discovery": {"management": "console:4444"},
            },
        }

        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    loaded_config: dict[str, Any] = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(
                    "Failed to load config from %s: %s", self.config_path, e
                )

        # Convert to ServiceConfig objects
        for name, config in default_config.items():
            self.services[name] = ServiceConfig(
                name=cast(str, config["name"]),
                container_name=cast(str, config["container_name"]),
                health_endpoint=cast(str, config["health_endpoint"]),
                port=cast(int, config["port"]),
                timeout=cast(int, config.get("timeout", 10)),
                retries=cast(int, config.get("retries", 3)),
                dependencies=cast(list[str], config.get("dependencies", [])),
                critical=cast(bool, config.get("critical", True)),
                networks=cast(list[str], config.get("networks", None)),
                service_discovery=cast(
                    dict[str, str], config.get("service_discovery", None)
                ),
            )

    async def check_service_health(
        self, service: ServiceConfig
    ) -> HealthCheckResult:
        """Check health of a single service.

        Args:
            service: Service configuration to check

        Returns:
            Health check result with status and metrics
        """
        start_time = time.time()

        try:
            # First check Docker container status
            docker_status = await self._check_docker_container_status(
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
            endpoint_result = await self._check_service_endpoint(service)

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

    async def _check_docker_container_status(
        self, container_name: str
    ) -> HealthStatus:
        """Check Docker container health status.

        Args:
            container_name: Name of Docker container to check

        Returns:
            Docker health status
        """
        try:
            # Check if container exists and is running
            cmd = [
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                container_name,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                # Container might not exist or not running
                cmd_status = [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Status}}",
                    container_name,
                ]
                status_result = subprocess.run(
                    cmd_status, capture_output=True, text=True, timeout=5
                )

                if status_result.returncode != 0:
                    return HealthStatus.UNKNOWN

                container_status = status_result.stdout.strip()
                if container_status == "running":
                    return HealthStatus.HEALTHY  # No health check configured
                else:
                    return HealthStatus.UNHEALTHY

            health_status = result.stdout.strip()

            # Map Docker health status to our enum
            status_mapping = {
                "healthy": HealthStatus.HEALTHY,
                "unhealthy": HealthStatus.UNHEALTHY,
                "starting": HealthStatus.STARTING,
                "none": HealthStatus.HEALTHY,  # No health check configured
                "": HealthStatus.UNKNOWN,
            }

            return status_mapping.get(health_status, HealthStatus.UNKNOWN)

        except subprocess.TimeoutExpired:
            self.logger.warning(
                "Docker command timed out for container %s", container_name
            )
            return HealthStatus.UNKNOWN
        except Exception as e:
            self.logger.error(
                "Error checking Docker status for %s: %s", container_name, e
            )
            return HealthStatus.UNKNOWN

    async def _check_service_endpoint(
        self, service: ServiceConfig
    ) -> dict[str, Any]:
        """Check service-specific endpoint health.

        Args:
            service: Service configuration

        Returns:
            Dictionary with endpoint check results
        """
        try:
            response = requests.get(
                service.health_endpoint,
                timeout=service.timeout,
                verify=False,  # For internal Docker network
            )

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return {
                        "status": HealthStatus.HEALTHY,
                        "http_status": response.status_code,
                        "response_data": response_data,
                        "headers": dict(response.headers),
                    }
                except json.JSONDecodeError:
                    # Response is not JSON, but 200 OK is good enough
                    return {
                        "status": HealthStatus.HEALTHY,
                        "http_status": response.status_code,
                        "content_type": response.headers.get(
                            "content-type", "unknown"
                        ),
                    }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "http_status": response.status_code,
                    "error": f"HTTP {response.status_code}",
                }

        except requests.exceptions.Timeout:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": "Request timeout",
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": "Connection refused",
            }
        except Exception as e:
            return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

    async def check_all_services(self) -> SystemHealthReport:
        """Check health of all configured services.

        Returns:
            Comprehensive system health report
        """
        self.logger.info("Starting health check for all services...")

        # Execute health checks concurrently
        tasks = []
        for service in self.services.values():
            task = asyncio.create_task(self.check_service_health(service))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        service_results = {}
        critical_failures = []

        for i, result in enumerate(results):
            service = list(self.services.values())[i]

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
        dependencies_satisfied = self._validate_dependencies(service_results)

        # Determine overall status
        if critical_failures:
            overall_status = HealthStatus.CRITICAL
        elif not dependencies_satisfied:
            overall_status = HealthStatus.UNHEALTHY
        elif any(
            r.status == HealthStatus.STARTING for r in service_results.values()
        ):
            overall_status = HealthStatus.STARTING
        elif all(
            r.status == HealthStatus.HEALTHY for r in service_results.values()
        ):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNHEALTHY

        # Generate recommendations
        recommendations = self._generate_recommendations(
            service_results, dependencies_satisfied
        )

        # Collect metrics
        metrics = self._collect_system_metrics(service_results)

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

    def _validate_dependencies(
        self, service_results: dict[str, HealthCheckResult]
    ) -> bool:
        """Validate service dependencies are satisfied.

        Args:
            service_results: Results from health checks

        Returns:
            True if all dependencies are satisfied
        """
        for service_name, service_config in self.services.items():
            service_result = service_results.get(service_name)

            if (
                not service_result
                or service_result.status != HealthStatus.HEALTHY
            ):
                continue

            # Check if all dependencies are healthy
            for dependency in service_config.dependencies or []:
                dep_result = service_results.get(dependency)
                if not dep_result or dep_result.status != HealthStatus.HEALTHY:
                    self.logger.warning(
                        "Service %s depends on unhealthy service %s",
                        service_name,
                        dependency,
                    )
                    return False

        return True

    def _generate_recommendations(
        self,
        service_results: dict[str, HealthCheckResult],
        dependencies_satisfied: bool,
    ) -> list[str]:
        """Generate recommendations based on health check results.

        Args:
            service_results: Health check results
            dependencies_satisfied: Whether dependencies are satisfied

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for critical failures
        critical_failures = [
            name
            for name, result in service_results.items()
            if result.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]
            and self.services[name].critical
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

    def _collect_system_metrics(
        self, service_results: dict[str, HealthCheckResult]
    ) -> dict[str, Any]:
        """Collect system-wide metrics.

        Args:
            service_results: Health check results

        Returns:
            Dictionary of system metrics
        """
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

        avg_response_time = (
            sum(r.response_time for r in service_results.values())
            / total_services
        )
        max_response_time = max(
            r.response_time for r in service_results.values()
        )

        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": unhealthy_services,
            "starting_services": starting_services,
            "health_percentage": (healthy_services / total_services) * 100,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "last_check_duration": sum(
                r.response_time for r in service_results.values()
            ),
        }

    def generate_dashboard_data(self) -> dict[str, Any]:
        """Generate data for monitoring dashboard.

        Returns:
            Dashboard data structure
        """
        if not self.health_history:
            return {"error": "No health check data available"}

        latest_report = self.health_history[-1]

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
        historical_data = []
        for report in self.health_history[-10:]:
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

    async def start_monitoring(self, interval: int = 30) -> None:
        """Start continuous health monitoring.

        Args:
            interval: Check interval in seconds
        """
        self.monitoring_active = True
        self.logger.info(
            "Starting continuous health monitoring (interval: %ds)", interval
        )

        while self.monitoring_active:
            try:
                await self.check_all_services()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error("Error during monitoring cycle: %s", e)
                await asyncio.sleep(5)  # Short delay before retry

    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        self.logger.info("Health monitoring stopped")

    def save_report(
        self, report: SystemHealthReport, output_path: Path
    ) -> None:
        """Save health report to file.

        Args:
            report: Health report to save
            output_path: Output file path
        """
        # Convert to serializable format
        report_data: dict[str, Any] = {
            "timestamp": report.timestamp.isoformat(),
            "overall_status": report.overall_status.value,
            "dependencies_satisfied": report.dependencies_satisfied,
            "recommendations": report.recommendations,
            "metrics": report.metrics,
            "services": {},
        }

        for name, result in report.services.items():
            report_data["services"][name] = {
                "status": result.status.value,
                "response_time": result.response_time,
                "timestamp": result.timestamp.isoformat(),
                "details": result.details,
                "error_message": result.error_message,
            }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        self.logger.info("Health report saved to %s", output_path)


# =============================================================================
# CLI Interface
# =============================================================================


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool = False) -> None:
    """CrackSeg Health Check System CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Save report to file")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "dashboard"]),
    default="json",
    help="Output format",
)
def check(output: str | None, output_format: str) -> None:
    """Run health check for all services."""
    import asyncio

    async def run_check() -> None:
        system = HealthCheckSystem()
        report = await system.check_all_services()

        if output_format == "dashboard":
            data = system.generate_dashboard_data()
            output_data = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            # Convert report to JSON
            output_data = json.dumps(
                asdict(report), indent=2, default=str, ensure_ascii=False
            )

        if output:
            with open(output, "w") as f:
                f.write(output_data)
            click.echo(f"Report saved to {output}")
        else:
            click.echo(output_data)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_check())


@cli.command()
@click.option("--interval", "-i", default=30, help="Check interval in seconds")
@click.option(
    "--output-dir", "-d", type=click.Path(), help="Directory for output files"
)
def monitor(interval: int, output_dir: str | None) -> None:
    """Start continuous health monitoring."""
    import asyncio

    async def run_monitor() -> None:
        system = HealthCheckSystem()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        try:
            await system.start_monitoring(interval)
        except KeyboardInterrupt:
            system.stop_monitoring()
            click.echo("Monitoring stopped")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_monitor())


@cli.command()
def dashboard() -> None:
    """Generate dashboard data for monitoring UI."""
    system = HealthCheckSystem()
    # Run a quick check to get latest data
    import asyncio

    async def run_dashboard() -> None:
        await system.check_all_services()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_dashboard())

    data = system.generate_dashboard_data()
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
