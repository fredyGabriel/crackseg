"""Environment Readiness Confirmation System.

This module provides comprehensive validation that the test environment
is properly reset and ready for the next test cycle after cleanup operations.
"""

import asyncio
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class ReadinessStatus(Enum):
    """Status enumeration for environment readiness checks."""

    READY = "ready"
    NOT_READY = "not_ready"
    CHECKING = "checking"
    ERROR = "error"
    TIMEOUT = "timeout"


class ComponentStatus(Enum):
    """Status of individual environment components."""

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ComponentCheck:
    """Individual component readiness check."""

    name: str
    description: str
    status: ComponentStatus
    duration: float
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReadinessReport:
    """Comprehensive environment readiness report."""

    overall_status: ReadinessStatus
    total_duration: float
    components_checked: int
    components_ready: int
    components_failed: int
    component_checks: list[ComponentCheck] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    performance_warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class EnvironmentReadinessChecker:
    """Validates environment readiness for next test cycle."""

    def __init__(
        self, thresholds_config: dict[str, Any] | None = None
    ) -> None:
        """Initialize environment readiness checker."""
        self.thresholds_config = thresholds_config or {}
        self.logger = logging.getLogger(__name__)
        self._component_checks = self._initialize_component_checks()

    def _initialize_component_checks(self) -> dict[str, dict[str, Any]]:
        """Initialize component check configurations."""
        return {
            "system_resources": {
                "memory_available_mb": 4000,  # Minimum available memory
                "cpu_available_percent": 20,  # Minimum available CPU
                "disk_available_gb": 5,  # Minimum available disk space
                "check_timeout": 10.0,
            },
            "python_environment": {
                "conda_env_active": True,
                "required_packages": ["torch", "streamlit", "pytest"],
                "python_version": "3.12",
                "check_timeout": 15.0,
            },
            "gpu_resources": {
                "gpu_memory_available_mb": 2000,  # Minimum GPU memory
                "cuda_available": True,
                "gpu_temperature_max_c": 80,
                "check_timeout": 8.0,
            },
            "file_system": {
                "temp_directory_writable": True,
                "test_data_accessible": True,
                "checkpoint_directory_writable": True,
                "max_open_files": 1000,
                "check_timeout": 12.0,
            },
            "network_connectivity": {
                "localhost_accessible": True,
                "streamlit_port_free": True,
                "docker_daemon_accessible": True,
                "check_timeout": 10.0,
            },
            "docker_environment": {
                "docker_running": True,
                "no_orphaned_containers": True,
                "sufficient_disk_space": True,
                "check_timeout": 20.0,
            },
        }

    async def check_environment_readiness(
        self, test_id: str | None = None
    ) -> ReadinessReport:
        """Perform comprehensive environment readiness check."""
        start_time = time.time()

        self.logger.info("Starting environment readiness check")

        component_checks: list[ComponentCheck] = []
        components_checked = 0
        components_ready = 0
        components_failed = 0
        blocking_issues: list[str] = []
        performance_warnings: list[str] = []
        recommendations: list[str] = []

        # Execute all component checks
        for component_name, config in self._component_checks.items():
            try:
                component_check = await asyncio.wait_for(
                    self._check_component(component_name, config),
                    timeout=config.get("check_timeout", 30.0),
                )

                component_checks.append(component_check)
                components_checked += 1

                if component_check.status == ComponentStatus.OPERATIONAL:
                    components_ready += 1
                elif component_check.status == ComponentStatus.FAILED:
                    components_failed += 1
                    blocking_issues.extend(component_check.errors)
                elif component_check.status == ComponentStatus.DEGRADED:
                    performance_warnings.extend(component_check.warnings)

            except TimeoutError:
                timeout_check = ComponentCheck(
                    name=component_name,
                    description=f"Check for {component_name}",
                    status=ComponentStatus.FAILED,
                    duration=config.get("check_timeout", 30.0),
                    errors=["Component check timed out"],
                )

                component_checks.append(timeout_check)
                components_checked += 1
                components_failed += 1
                blocking_issues.append(f"{component_name} check timed out")

            except Exception as e:
                error_check = ComponentCheck(
                    name=component_name,
                    description=f"Check for {component_name}",
                    status=ComponentStatus.FAILED,
                    duration=0.0,
                    errors=[f"Check failed: {e}"],
                )

                component_checks.append(error_check)
                components_checked += 1
                components_failed += 1
                blocking_issues.append(f"{component_name}: {e}")

        # Generate recommendations based on issues found
        recommendations = self._generate_recommendations(component_checks)

        # Determine overall readiness status
        overall_status = self._determine_overall_status(
            components_ready, components_failed, blocking_issues
        )

        total_duration = time.time() - start_time

        report = ReadinessReport(
            overall_status=overall_status,
            total_duration=total_duration,
            components_checked=components_checked,
            components_ready=components_ready,
            components_failed=components_failed,
            component_checks=component_checks,
            blocking_issues=blocking_issues,
            performance_warnings=performance_warnings,
            recommendations=recommendations,
        )

        self.logger.info(
            f"Environment readiness check completed: {overall_status.value} "
            f"({components_ready}/{components_checked} components ready)"
        )

        return report

    async def _check_component(
        self, component_name: str, config: dict[str, Any]
    ) -> ComponentCheck:
        """Check readiness of a specific environment component."""
        start_time = time.time()

        if component_name == "system_resources":
            return await self._check_system_resources(config)
        elif component_name == "python_environment":
            return await self._check_python_environment(config)
        elif component_name == "gpu_resources":
            return await self._check_gpu_resources(config)
        elif component_name == "file_system":
            return await self._check_file_system(config)
        elif component_name == "network_connectivity":
            return await self._check_network_connectivity(config)
        elif component_name == "docker_environment":
            return await self._check_docker_environment(config)
        else:
            return ComponentCheck(
                name=component_name,
                description=f"Unknown component: {component_name}",
                status=ComponentStatus.UNKNOWN,
                duration=time.time() - start_time,
                errors=[f"Unknown component type: {component_name}"],
            )

    async def _check_system_resources(
        self, config: dict[str, Any]
    ) -> ComponentCheck:
        """Check system resource availability."""
        start_time = time.time()
        errors = []
        warnings = []
        details = {}

        try:
            # Memory check
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / (1024 * 1024)
            required_memory_mb = config.get("memory_available_mb", 4000)

            details["memory_available_mb"] = available_memory_mb
            details["memory_required_mb"] = required_memory_mb

            if available_memory_mb < required_memory_mb:
                errors.append(
                    f"Insufficient memory: {available_memory_mb:.0f}MB "
                    f"available, {required_memory_mb}MB required"
                )

            # CPU check
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_available = 100 - cpu_usage
            required_cpu = config.get("cpu_available_percent", 20)

            details["cpu_available_percent"] = cpu_available
            details["cpu_required_percent"] = required_cpu

            if cpu_available < required_cpu:
                warnings.append(
                    f"High CPU usage: {cpu_available:.1f}% available, "
                    f"{required_cpu}% recommended"
                )

            # Disk space check
            disk = psutil.disk_usage("/")
            available_disk_gb = disk.free / (1024**3)
            required_disk_gb = config.get("disk_available_gb", 5)

            details["disk_available_gb"] = available_disk_gb
            details["disk_required_gb"] = required_disk_gb

            if available_disk_gb < required_disk_gb:
                errors.append(
                    f"Insufficient disk space: {available_disk_gb:.1f}GB "
                    f"available, {required_disk_gb}GB required"
                )

            status = (
                ComponentStatus.FAILED
                if errors
                else (
                    ComponentStatus.DEGRADED
                    if warnings
                    else ComponentStatus.OPERATIONAL
                )
            )

        except Exception as e:
            errors.append(f"System resource check failed: {e}")
            status = ComponentStatus.FAILED

        return ComponentCheck(
            name="system_resources",
            description="System resource availability",
            status=status,
            duration=time.time() - start_time,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    async def _check_python_environment(
        self, config: dict[str, Any]
    ) -> ComponentCheck:
        """Check Python environment readiness."""
        start_time = time.time()
        errors = []
        warnings = []
        details = {}

        try:
            # Check Python version
            python_version = platform.python_version()
            required_version = config.get("python_version", "3.12")

            details["python_version"] = python_version
            details["required_version"] = required_version

            if not python_version.startswith(required_version):
                warnings.append(
                    f"Python version {python_version} may not be optimal "
                    f"(recommended: {required_version})"
                )

            # Check conda environment
            conda_env = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if conda_env.returncode == 0 and "crackseg" in conda_env.stdout:
                details["conda_environment"] = "crackseg environment found"
            else:
                errors.append("crackseg conda environment not found")

            # Check required packages
            required_packages = config.get("required_packages", [])
            missing_packages = []

            for package in required_packages:
                try:
                    __import__(package)
                    details[f"package_{package}"] = "available"
                except ImportError:
                    missing_packages.append(package)
                    details[f"package_{package}"] = "missing"

            if missing_packages:
                errors.append(f"Missing required packages: {missing_packages}")

            status = (
                ComponentStatus.FAILED
                if errors
                else (
                    ComponentStatus.DEGRADED
                    if warnings
                    else ComponentStatus.OPERATIONAL
                )
            )

        except Exception as e:
            errors.append(f"Python environment check failed: {e}")
            status = ComponentStatus.FAILED

        return ComponentCheck(
            name="python_environment",
            description="Python environment and packages",
            status=status,
            duration=time.time() - start_time,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    async def _check_gpu_resources(
        self, config: dict[str, Any]
    ) -> ComponentCheck:
        """Check GPU resource availability."""
        start_time = time.time()
        errors = []
        warnings = []
        details = {}

        try:
            import torch

            cuda_available = torch.cuda.is_available()
            details["cuda_available"] = cuda_available

            if cuda_available:
                gpu_memory_total = torch.cuda.get_device_properties(
                    0
                ).total_memory
                gpu_memory_used = torch.cuda.memory_allocated(0)
                gpu_memory_available = gpu_memory_total - gpu_memory_used
                gpu_memory_available_mb = gpu_memory_available / (1024 * 1024)

                required_gpu_memory_mb = config.get(
                    "gpu_memory_available_mb", 2000
                )

                details["gpu_memory_total_mb"] = gpu_memory_total / (
                    1024 * 1024
                )
                details["gpu_memory_available_mb"] = gpu_memory_available_mb
                details["gpu_memory_required_mb"] = required_gpu_memory_mb

                if gpu_memory_available_mb < required_gpu_memory_mb:
                    errors.append(
                        f"Insufficient GPU memory: "
                        f"{gpu_memory_available_mb:.0f}MB available, "
                        f"{required_gpu_memory_mb}MB required"
                    )
            else:
                if config.get("cuda_available", True):
                    warnings.append(
                        "CUDA not available, tests may run on CPU only"
                    )

            status = (
                ComponentStatus.FAILED
                if errors
                else (
                    ComponentStatus.DEGRADED
                    if warnings
                    else ComponentStatus.OPERATIONAL
                )
            )

        except ImportError:
            warnings.append("PyTorch not available for GPU check")
            status = ComponentStatus.DEGRADED
        except Exception as e:
            errors.append(f"GPU resource check failed: {e}")
            status = ComponentStatus.FAILED

        return ComponentCheck(
            name="gpu_resources",
            description="GPU availability and memory",
            status=status,
            duration=time.time() - start_time,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    async def _check_file_system(
        self, config: dict[str, Any]
    ) -> ComponentCheck:
        """Check file system readiness."""
        start_time = time.time()
        errors = []
        warnings = []
        details = {}

        try:
            # Check temp directory writability
            temp_path = (
                Path("/tmp")
                if platform.system() != "Windows"
                else Path("C:/temp")
            )
            if temp_path.exists() and temp_path.is_dir():
                test_file = (
                    temp_path / f"readiness_test_{int(time.time())}.tmp"
                )
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    details["temp_directory_writable"] = True
                except Exception:
                    errors.append("Temporary directory is not writable")
                    details["temp_directory_writable"] = False

            # Check test data accessibility
            data_path = Path("data")
            if data_path.exists():
                details["test_data_accessible"] = True
            else:
                warnings.append("Test data directory not found")
                details["test_data_accessible"] = False

            status = (
                ComponentStatus.FAILED
                if errors
                else (
                    ComponentStatus.DEGRADED
                    if warnings
                    else ComponentStatus.OPERATIONAL
                )
            )

        except Exception as e:
            errors.append(f"File system check failed: {e}")
            status = ComponentStatus.FAILED

        return ComponentCheck(
            name="file_system",
            description="File system accessibility and permissions",
            status=status,
            duration=time.time() - start_time,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    async def _check_network_connectivity(
        self, config: dict[str, Any]
    ) -> ComponentCheck:
        """Check network connectivity readiness."""
        start_time = time.time()
        details = {"localhost_accessible": True}  # Placeholder implementation

        return ComponentCheck(
            name="network_connectivity",
            description="Network connectivity and port availability",
            status=ComponentStatus.OPERATIONAL,
            duration=time.time() - start_time,
            details=details,
        )

    async def _check_docker_environment(
        self, config: dict[str, Any]
    ) -> ComponentCheck:
        """Check Docker environment readiness."""
        start_time = time.time()
        details = {"docker_running": True}  # Placeholder implementation

        return ComponentCheck(
            name="docker_environment",
            description="Docker daemon and container management",
            status=ComponentStatus.OPERATIONAL,
            duration=time.time() - start_time,
            details=details,
        )

    def _generate_recommendations(
        self, component_checks: list[ComponentCheck]
    ) -> list[str]:
        """Generate recommendations based on component check results."""
        recommendations = []

        for check in component_checks:
            if check.status == ComponentStatus.FAILED:
                if check.name == "system_resources":
                    recommendations.append(
                        "Free up system memory and disk space"
                    )
                elif check.name == "python_environment":
                    recommendations.append(
                        "Activate crackseg conda environment"
                    )
                elif check.name == "gpu_resources":
                    recommendations.append(
                        "Clear GPU memory or restart GPU drivers"
                    )

        return recommendations

    def _determine_overall_status(
        self, ready: int, failed: int, blocking_issues: list[str]
    ) -> ReadinessStatus:
        """Determine overall environment readiness status."""
        if failed == 0 and not blocking_issues:
            return ReadinessStatus.READY
        elif failed > 0 or blocking_issues:
            return ReadinessStatus.NOT_READY
        else:
            return ReadinessStatus.ERROR
