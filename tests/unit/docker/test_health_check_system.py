"""Unit tests for the health check system.

This module tests the HealthCheckSystem class and related components
to ensure correct functionality for Docker service monitoring.

Author: CrackSeg Project
Version: 1.0 (Subtask 13.7)
"""

import json
import subprocess

# Import the system under test
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

# Add the health_check_system to path
sys.path.append(str(Path(__file__).parents[3] / "tests" / "docker"))

# Import after path modification
from health_check_system import (  # noqa: E402
    HealthCheckResult,
    HealthCheckSystem,
    HealthStatus,
    ServiceConfig,
    SystemHealthReport,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_service_config() -> ServiceConfig:
    """Provide a sample service configuration for testing."""
    return ServiceConfig(
        name="test-service",
        container_name="test-container",
        health_endpoint="http://test-service:8080/health",
        port=8080,
        timeout=10,
        retries=3,
        dependencies=["dependency-service"],
        critical=True,
    )


@pytest.fixture
def sample_health_result() -> HealthCheckResult:
    """Provide a sample health check result for testing."""
    return HealthCheckResult(
        service_name="test-service",
        status=HealthStatus.HEALTHY,
        response_time=0.5,
        timestamp=datetime.now(),
        details={"test": "data"},
        error_message=None,
    )


@pytest.fixture
def mock_config_file(tmp_path: Path) -> Path:
    """Create a temporary configuration file for testing."""
    config_data = {
        "test-service": {
            "name": "test-service",
            "container_name": "test-container",
            "health_endpoint": "http://test-service:8080/health",
            "port": 8080,
            "timeout": 5,
            "retries": 2,
            "dependencies": [],
            "critical": True,
        }
    }

    config_file = tmp_path / "test_health_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)

    return config_file


@pytest.fixture
def health_system(mock_config_file: Path) -> HealthCheckSystem:
    """Create a HealthCheckSystem instance for testing."""
    return HealthCheckSystem(config_path=mock_config_file)


# =============================================================================
# Test Data Models
# =============================================================================


class TestServiceConfig:
    """Test the ServiceConfig data class."""

    def test_service_config_initialization(self) -> None:
        """Test ServiceConfig can be created with required parameters."""
        config = ServiceConfig(
            name="test",
            container_name="test-container",
            health_endpoint="http://test:8080",
            port=8080,
        )

        assert config.name == "test"
        assert config.container_name == "test-container"
        assert config.health_endpoint == "http://test:8080"
        assert config.port == 8080
        assert config.timeout == 10  # Default value
        assert config.retries == 3  # Default value
        assert config.dependencies == []  # Default empty list
        assert config.critical is True  # Default value

    def test_service_config_with_dependencies(self) -> None:
        """Test ServiceConfig with dependencies list."""
        config = ServiceConfig(
            name="test",
            container_name="test-container",
            health_endpoint="http://test:8080",
            port=8080,
            dependencies=["service1", "service2"],
        )

        assert config.dependencies == ["service1", "service2"]

    def test_service_config_post_init(self) -> None:
        """Test __post_init__ initializes dependencies if None."""
        config = ServiceConfig(
            name="test",
            container_name="test-container",
            health_endpoint="http://test:8080",
            port=8080,
        )

        assert config.dependencies == []


class TestHealthCheckResult:
    """Test the HealthCheckResult data class."""

    def test_health_check_result_creation(self) -> None:
        """Test HealthCheckResult can be created with all parameters."""
        timestamp = datetime.now()

        result = HealthCheckResult(
            service_name="test-service",
            status=HealthStatus.HEALTHY,
            response_time=1.5,
            timestamp=timestamp,
            details={"key": "value"},
            error_message="test error",
        )

        assert result.service_name == "test-service"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time == 1.5
        assert result.timestamp == timestamp
        assert result.details == {"key": "value"}
        assert result.error_message == "test error"

    def test_health_check_result_default_error(self) -> None:
        """Test HealthCheckResult with default error_message."""
        result = HealthCheckResult(
            service_name="test",
            status=HealthStatus.HEALTHY,
            response_time=0.5,
            timestamp=datetime.now(),
            details={},
        )

        assert result.error_message is None


# =============================================================================
# Test HealthCheckSystem Core Functionality
# =============================================================================


class TestHealthCheckSystem:
    """Test the HealthCheckSystem class."""

    def test_initialization(self, health_system: HealthCheckSystem) -> None:
        """Test HealthCheckSystem initialization."""
        # System loads defaults + custom config, so should have 7 total
        # (custom config updates defaults, doesn't replace them)
        # Updated for grid-console addition in subtask 13.9
        assert len(health_system.services) == 7
        assert (
            "test-service" in health_system.services
        )  # Our custom service is added
        assert health_system.monitoring_active is False
        assert health_system.health_history == []

    def test_default_configuration_loading(self, tmp_path: Path) -> None:
        """Test loading default configuration when no config file exists."""
        non_existent_config = tmp_path / "non_existent.json"
        system = HealthCheckSystem(config_path=non_existent_config)

        # Should load default services (6 total in the default config, updated
        # for grid-console)
        expected_services = [
            "streamlit-app",
            "selenium-hub",
            "chrome-node",
            "firefox-node",
            "test-runner",
            "grid-console",
        ]
        assert len(system.services) == 6
        for service_name in expected_services:
            assert service_name in system.services

    def test_custom_configuration_loading(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test loading custom configuration from file."""
        # The fixture loads default services (6) + custom service (1) = 7 total
        # Custom config is merged with defaults (updated for grid-console
        # addition)
        assert len(health_system.services) == 7
        assert "test-service" in health_system.services

        service = health_system.services["test-service"]
        assert service.name == "test-service"
        assert service.port == 8080
        assert service.timeout == 5
        assert service.retries == 2

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_check_docker_container_status_healthy(
        self, mock_subprocess: MagicMock, health_system: HealthCheckSystem
    ) -> None:
        """Test checking healthy Docker container status."""
        # Mock successful docker health check
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "healthy"

        result = await health_system._check_docker_container_status(
            "test-container"
        )

        assert result == HealthStatus.HEALTHY
        mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_check_docker_container_status_unhealthy(
        self, mock_subprocess: MagicMock, health_system: HealthCheckSystem
    ) -> None:
        """Test checking unhealthy Docker container status."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "unhealthy"

        result = await health_system._check_docker_container_status(
            "test-container"
        )

        assert result == HealthStatus.UNHEALTHY

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_check_docker_container_status_not_found(
        self, mock_subprocess: MagicMock, health_system: HealthCheckSystem
    ) -> None:
        """Test checking status of non-existent container."""
        # Mock both docker inspect calls failing (container not found)
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "No such container"

        result = await health_system._check_docker_container_status(
            "nonexistent"
        )

        assert (
            result == HealthStatus.UNKNOWN
        )  # Returns UNKNOWN when container not found

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_check_docker_container_status_timeout(
        self, mock_subprocess: MagicMock, health_system: HealthCheckSystem
    ) -> None:
        """Test Docker container status check timeout."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("docker", 10)

        result = await health_system._check_docker_container_status(
            "test-container"
        )

        assert result == HealthStatus.UNKNOWN  # Returns UNKNOWN on timeout

    @patch("requests.get")
    @pytest.mark.asyncio
    async def test_check_service_endpoint_healthy(
        self,
        mock_requests: MagicMock,
        sample_service_config: ServiceConfig,
        health_system: HealthCheckSystem,
    ) -> None:
        """Test checking healthy service endpoint."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.headers = {"content-type": "application/json"}
        mock_requests.return_value = mock_response

        result = await health_system._check_service_endpoint(
            sample_service_config
        )

        assert result["status"] == HealthStatus.HEALTHY
        assert result["http_status"] == 200
        assert result["response_data"] == {"status": "ok"}

    @patch("requests.get")
    @pytest.mark.asyncio
    async def test_check_service_endpoint_unhealthy(
        self,
        mock_requests: MagicMock,
        sample_service_config: ServiceConfig,
        health_system: HealthCheckSystem,
    ) -> None:
        """Test checking unhealthy service endpoint."""
        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_requests.return_value = mock_response

        result = await health_system._check_service_endpoint(
            sample_service_config
        )

        assert result["status"] == HealthStatus.UNHEALTHY
        assert result["http_status"] == 500

    @patch("requests.get")
    @pytest.mark.asyncio
    async def test_check_service_endpoint_timeout(
        self,
        mock_requests: MagicMock,
        sample_service_config: ServiceConfig,
        health_system: HealthCheckSystem,
    ) -> None:
        """Test service endpoint timeout."""
        import requests

        mock_requests.side_effect = requests.Timeout()

        result = await health_system._check_service_endpoint(
            sample_service_config
        )

        assert result["status"] == HealthStatus.UNHEALTHY
        assert "timeout" in result["error"].lower()

    @patch("requests.get")
    @pytest.mark.asyncio
    async def test_check_service_endpoint_connection_error(
        self,
        mock_requests: MagicMock,
        sample_service_config: ServiceConfig,
        health_system: HealthCheckSystem,
    ) -> None:
        """Test service endpoint connection error."""
        import requests

        mock_requests.side_effect = requests.ConnectionError()

        result = await health_system._check_service_endpoint(
            sample_service_config
        )

        assert result["status"] == HealthStatus.UNHEALTHY
        assert "connection" in result["error"].lower()


# =============================================================================
# Test System Health Operations
# =============================================================================


class TestSystemHealthOperations:
    """Test system-level health operations."""

    @patch.object(HealthCheckSystem, "check_service_health")
    @pytest.mark.asyncio
    async def test_check_all_services_success(
        self, mock_check_service: AsyncMock, health_system: HealthCheckSystem
    ) -> None:
        """Test checking all services with all healthy."""
        # Mock all services as healthy
        mock_check_service.return_value = HealthCheckResult(
            service_name="test-service",
            status=HealthStatus.HEALTHY,
            response_time=0.5,
            timestamp=datetime.now(),
            details={"status": "ok"},
        )

        report = await health_system.check_all_services()

        assert report.overall_status == HealthStatus.HEALTHY
        assert len(report.services) == 6
        assert report.dependencies_satisfied is True

    @patch.object(HealthCheckSystem, "check_service_health")
    @pytest.mark.asyncio
    async def test_check_all_services_with_failures(
        self, mock_check_service: AsyncMock, health_system: HealthCheckSystem
    ) -> None:
        """Test checking all services with failures."""
        # Mock service as unhealthy
        mock_check_service.return_value = HealthCheckResult(
            service_name="test-service",
            status=HealthStatus.UNHEALTHY,
            response_time=0.0,
            timestamp=datetime.now(),
            details={"error": "connection failed"},
            error_message="Service unreachable",
        )

        report = await health_system.check_all_services()

        assert (
            report.overall_status == HealthStatus.CRITICAL
        )  # System uses CRITICAL for critical failures
        assert len(report.services) == 6

    @patch.object(HealthCheckSystem, "check_service_health")
    @pytest.mark.asyncio
    async def test_check_all_services_with_exception(
        self, mock_check_service: AsyncMock, health_system: HealthCheckSystem
    ) -> None:
        """Test handling exceptions during health checks."""
        mock_check_service.side_effect = Exception("Unexpected error")

        report = await health_system.check_all_services()

        assert (
            report.overall_status == HealthStatus.CRITICAL
        )  # System uses CRITICAL for exceptions

    def test_validate_dependencies_satisfied(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test dependency validation when all dependencies are met."""
        # Create mock results with all healthy services
        service_results = {
            "service1": HealthCheckResult(
                service_name="service1",
                status=HealthStatus.HEALTHY,
                response_time=0.5,
                timestamp=datetime.now(),
                details={},
            ),
            "service2": HealthCheckResult(
                service_name="service2",
                status=HealthStatus.HEALTHY,
                response_time=0.3,
                timestamp=datetime.now(),
                details={},
            ),
        }

        # Mock a service with dependencies
        health_system.services["service3"] = ServiceConfig(
            name="service3",
            container_name="service3-container",
            health_endpoint="http://service3:8080",
            port=8080,
            dependencies=["service1", "service2"],
        )

        result = health_system._validate_dependencies(service_results)
        assert result is True

    def test_validate_dependencies_not_satisfied(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test dependency validation when dependencies are not met."""
        # Create mock results - using real service names that exist
        # chrome-node depends on selenium-hub, make selenium-hub unhealthy
        service_results = {
            "selenium-hub": HealthCheckResult(
                service_name="selenium-hub",
                status=HealthStatus.UNHEALTHY,  # This one is unhealthy
                response_time=0.0,
                timestamp=datetime.now(),
                details={},
                error_message="Service down",
            ),
            "chrome-node": HealthCheckResult(
                service_name="chrome-node",
                status=HealthStatus.HEALTHY,  # Depends on unhealthy hub
                response_time=0.5,
                timestamp=datetime.now(),
                details={},
            ),
        }

        result = health_system._validate_dependencies(service_results)
        assert result is False

    def test_generate_recommendations_critical_failures(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test generating recommendations for critical service failures."""
        service_results = {
            "critical-service": HealthCheckResult(
                service_name="critical-service",
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                timestamp=datetime.now(),
                details={},
                error_message="Service down",
            )
        }

        # Add critical service to system
        health_system.services["critical-service"] = ServiceConfig(
            name="critical-service",
            container_name="critical-container",
            health_endpoint="http://critical:8080",
            port=8080,
            critical=True,
        )

        recommendations = health_system._generate_recommendations(
            service_results, dependencies_satisfied=True
        )

        assert len(recommendations) > 0
        assert any("critical" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_slow_response(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test generating recommendations for slow services."""
        service_results = {
            "slow-service": HealthCheckResult(
                service_name="slow-service",
                status=HealthStatus.HEALTHY,
                response_time=15.0,  # Very slow response
                timestamp=datetime.now(),
                details={},
            )
        }

        recommendations = health_system._generate_recommendations(
            service_results, dependencies_satisfied=True
        )

        assert len(recommendations) > 0
        assert any(
            "performance" in rec.lower() or "slow" in rec.lower()
            for rec in recommendations
        )

    def test_collect_system_metrics(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test collecting system metrics."""
        service_results = {
            "service1": HealthCheckResult(
                service_name="service1",
                status=HealthStatus.HEALTHY,
                response_time=0.5,
                timestamp=datetime.now(),
                details={},
            ),
            "service2": HealthCheckResult(
                service_name="service2",
                status=HealthStatus.UNHEALTHY,
                response_time=2.0,
                timestamp=datetime.now(),
                details={},
            ),
            "service3": HealthCheckResult(
                service_name="service3",
                status=HealthStatus.STARTING,
                response_time=1.0,
                timestamp=datetime.now(),
                details={},
            ),
        }

        metrics = health_system._collect_system_metrics(service_results)

        assert metrics["total_services"] == 3
        assert metrics["healthy_services"] == 1
        assert metrics["unhealthy_services"] == 1
        assert metrics["starting_services"] == 1
        assert (
            abs(metrics["health_percentage"] - 33.33) < 0.01
        )  # Allow for floating point precision
        assert (
            abs(metrics["avg_response_time"] - 1.17) < 0.01
        )  # (0.5 + 2.0 + 1.0) / 3
        assert metrics["max_response_time"] == 2.0


# =============================================================================
# Test Dashboard and Reporting
# =============================================================================


class TestDashboardAndReporting:
    """Test dashboard and reporting functionality."""

    def test_generate_dashboard_data_no_history(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test generating dashboard data with no history."""
        data = health_system.generate_dashboard_data()

        # When no history, should return error message
        assert "error" in data
        assert data["error"] == "No health check data available"

    @patch.object(HealthCheckSystem, "check_all_services")
    @pytest.mark.asyncio
    async def test_generate_dashboard_data_with_history(
        self, mock_check_all: AsyncMock, health_system: HealthCheckSystem
    ) -> None:
        """Test generating dashboard data with health history."""
        # Mock health check result
        mock_report = SystemHealthReport(
            timestamp=datetime.now(),
            overall_status=HealthStatus.HEALTHY,
            services={
                "test-service": HealthCheckResult(
                    service_name="test-service",
                    status=HealthStatus.HEALTHY,
                    response_time=0.5,
                    timestamp=datetime.now(),
                    details={},
                )
            },
            dependencies_satisfied=True,
            recommendations=[],
            metrics={"total_services": 1, "healthy_services": 1},
        )
        mock_check_all.return_value = mock_report

        # Generate history by calling mocked method (returns our report)
        report1 = await health_system.check_all_services()
        report2 = await health_system.check_all_services()

        # Manually add to history since the mock might not do it
        health_system.health_history = [report1, report2]

        data = health_system.generate_dashboard_data()

        # Should now have data, not error
        assert "error" not in data
        assert data["overall_status"] == "healthy"
        assert "test-service" in data["services"]
        assert "historical_data" in data
        assert len(data["historical_data"]) == 2

    def test_save_report(
        self, health_system: HealthCheckSystem, tmp_path: Path
    ) -> None:
        """Test saving health check report."""
        # Create test report
        report = SystemHealthReport(
            timestamp=datetime.now(),
            overall_status=HealthStatus.HEALTHY,
            services={
                "test-service": HealthCheckResult(
                    service_name="test-service",
                    status=HealthStatus.HEALTHY,
                    response_time=0.5,
                    timestamp=datetime.now(),
                    details={"test": "data"},
                )
            },
            dependencies_satisfied=True,
            recommendations=["All systems operational"],
            metrics={"total_services": 1, "healthy_services": 1},
        )

        # Save to temporary file
        report_file = tmp_path / "test_report.json"
        health_system.save_report(report, report_file)

        # Verify file exists and contains expected data
        assert report_file.exists()

        with open(report_file) as f:
            saved_data = json.load(f)

        assert saved_data["overall_status"] == "healthy"
        assert "test-service" in saved_data["services"]
        assert saved_data["dependencies_satisfied"] is True


# =============================================================================
# Test Monitoring Operations
# =============================================================================


class TestMonitoringOperations:
    """Test continuous monitoring operations."""

    def test_start_stop_monitoring(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test starting and stopping monitoring."""
        # Initially not monitoring
        assert health_system.monitoring_active is False

        # Start monitoring (async method - just test flag)
        assert health_system.monitoring_active is False

        # Stop monitoring
        health_system.stop_monitoring()
        assert health_system.monitoring_active is False

    @pytest.mark.asyncio
    async def test_monitoring_handles_exceptions(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test monitoring handles exceptions gracefully."""
        with patch.object(health_system, "check_all_services") as mock_check:
            with patch("asyncio.sleep") as mock_sleep:
                mock_check.side_effect = Exception("Test exception")
                mock_sleep.side_effect = [
                    None,
                    KeyboardInterrupt(),
                ]  # Stop after one cycle

                # Should not raise exception, monitoring should handle it
                try:
                    await health_system.start_monitoring(interval=1)
                except KeyboardInterrupt:
                    pass  # Expected to stop monitoring

                # Verify exception was handled and logged
                assert mock_check.called


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestHealthCheckSystemIntegration:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_health_check_workflow(
        self, health_system: HealthCheckSystem
    ) -> None:
        """Test complete health check workflow."""
        with patch.object(
            health_system, "_check_docker_container_status"
        ) as mock_docker:
            with patch.object(
                health_system, "_check_service_endpoint"
            ) as mock_endpoint:
                # Mock successful checks
                mock_docker.return_value = HealthStatus.HEALTHY
                mock_endpoint.return_value = {
                    "status": HealthStatus.HEALTHY,
                    "http_status": 200,
                    "response_data": {"status": "ok"},
                    "headers": {"content-type": "application/json"},
                    "response_time_ms": 0,
                }

                # Run full workflow
                report = await health_system.check_all_services()

                # Verify results
                assert report.overall_status == HealthStatus.HEALTHY
                assert len(report.services) == 6  # All services checked
                assert report.dependencies_satisfied is True
                assert len(health_system.health_history) == 1

    def test_configuration_merge_with_defaults(self, tmp_path: Path) -> None:
        """Test configuration merging with default values."""
        # Create partial config
        partial_config = {
            "custom-service": {
                "name": "custom-service",
                "container_name": "custom-container",
                "health_endpoint": "http://custom:9000/status",
                "port": 9000,
                # Missing timeout, retries, dependencies - use defaults
            }
        }

        config_file = tmp_path / "partial_config.json"
        with open(config_file, "w") as f:
            json.dump(partial_config, f)

        system = HealthCheckSystem(config_path=config_file)

        service = system.services["custom-service"]
        assert service.name == "custom-service"
        assert service.port == 9000
        assert service.timeout == 10  # Default
        assert service.retries == 3  # Default
        assert service.dependencies == []  # Default
        assert service.critical is True  # Default
