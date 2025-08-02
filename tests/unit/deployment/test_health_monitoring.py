"""Unit tests for health monitoring system."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.crackseg.utils.deployment.health_monitoring import (
    DefaultHealthChecker,
    DefaultResourceMonitor,
    DeploymentHealthMonitor,
    HealthCheckResult,
    ResourceMetrics,
)


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_health_check_result_creation(self) -> None:
        """Test creating HealthCheckResult instances."""
        result = HealthCheckResult(
            success=True,
            status="healthy",
            response_time_ms=150.5,
            timestamp=time.time(),
            details={"status_code": 200},
        )

        assert result.success is True
        assert result.status == "healthy"
        assert result.response_time_ms == 150.5
        assert result.details == {"status_code": 200}
        assert result.error_message is None

    def test_health_check_result_with_error(self) -> None:
        """Test HealthCheckResult with error information."""
        result = HealthCheckResult(
            success=False,
            status="unhealthy",
            response_time_ms=5000.0,
            timestamp=time.time(),
            error_message="Connection timeout",
        )

        assert result.success is False
        assert result.status == "unhealthy"
        assert result.response_time_ms == 5000.0
        assert result.error_message == "Connection timeout"


class TestResourceMetrics:
    """Test ResourceMetrics dataclass."""

    def test_resource_metrics_creation(self) -> None:
        """Test creating ResourceMetrics instances."""
        metrics = ResourceMetrics(
            cpu_usage_percent=45.2,
            memory_usage_mb=1024.5,
            disk_usage_percent=67.8,
            network_io_mbps=12.3,
            timestamp=time.time(),
        )

        assert metrics.cpu_usage_percent == 45.2
        assert metrics.memory_usage_mb == 1024.5
        assert metrics.disk_usage_percent == 67.8
        assert metrics.network_io_mbps == 12.3


class TestDefaultHealthChecker:
    """Test DefaultHealthChecker implementation."""

    def test_health_checker_initialization(self) -> None:
        """Test DefaultHealthChecker initialization."""
        checker = DefaultHealthChecker()
        assert checker.session is not None
        assert "User-Agent" in checker.session.headers

    @patch("src.crackseg.utils.deployment.health_monitoring.requests.Session")
    def test_check_health_success(self, mock_session: Mock) -> None:
        """Test successful health check."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"OK"
        mock_session.return_value.get.return_value = mock_response

        checker = DefaultHealthChecker()
        result = checker.check_health("http://localhost:8080/health")

        assert result.success is True
        assert result.status == "healthy"
        assert result.response_time_ms >= 0
        assert result.details == {"status_code": 200, "content_length": 2}

    @patch("src.crackseg.utils.deployment.health_monitoring.requests.Session")
    def test_check_health_failure(self, mock_session: Mock) -> None:
        """Test failed health check."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_session.return_value.get.return_value = mock_response

        checker = DefaultHealthChecker()
        result = checker.check_health("http://localhost:8080/health")

        assert result.success is False
        assert result.status == "unhealthy"
        assert result.error_message == "HTTP 500"
        assert result.details == {"status_code": 500}

    @patch("src.crackseg.utils.deployment.health_monitoring.requests.Session")
    def test_check_health_connection_error(self, mock_session: Mock) -> None:
        """Test health check with connection error."""
        # Mock connection error
        mock_session.return_value.get.side_effect = Exception(
            "Connection failed"
        )

        checker = DefaultHealthChecker()
        result = checker.check_health("http://localhost:8080/health")

        assert result.success is False
        assert result.status == "unhealthy"
        assert result.error_message == "Connection failed"

    @patch("src.crackseg.utils.deployment.health_monitoring.requests.Session")
    def test_wait_for_healthy_success(self, mock_session: Mock) -> None:
        """Test waiting for healthy service."""
        # Mock successful response after first call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.return_value.get.return_value = mock_response

        checker = DefaultHealthChecker()
        result = checker.wait_for_healthy(
            "http://localhost:8080/health", max_wait=10
        )

        assert result is True

    @patch("src.crackseg.utils.deployment.health_monitoring.requests.Session")
    def test_wait_for_healthy_timeout(self, mock_session: Mock) -> None:
        """Test waiting for healthy service with timeout."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_session.return_value.get.return_value = mock_response

        checker = DefaultHealthChecker()
        result = checker.wait_for_healthy(
            "http://localhost:8080/health", max_wait=1
        )

        assert result is False


class TestDefaultResourceMonitor:
    """Test DefaultResourceMonitor implementation."""

    @patch("src.crackseg.utils.deployment.health_monitoring.psutil")
    def test_get_system_metrics(self, mock_psutil: Mock) -> None:
        """Test getting system metrics."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value = Mock(
            used=1024 * 1024 * 512
        )  # 512MB
        mock_psutil.disk_usage.return_value = Mock(percent=45.2)
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1024 * 1024, bytes_recv=2048 * 1024
        )

        monitor = DefaultResourceMonitor()
        metrics = monitor.get_system_metrics()

        assert metrics.cpu_usage_percent == 25.5
        assert metrics.memory_usage_mb == 512.0
        assert metrics.disk_usage_percent == 45.2
        assert metrics.network_io_mbps > 0

    @patch("src.crackseg.utils.deployment.health_monitoring.psutil")
    def test_get_process_metrics_found(self, mock_psutil: Mock) -> None:
        """Test getting process metrics when process is found."""
        # Mock process iteration
        mock_process = Mock()
        mock_process.info = {
            "pid": 12345,
            "name": "python.exe",
            "cpu_percent": 15.2,
            "memory_info": Mock(rss=1024 * 1024 * 256),  # 256MB
        }
        mock_psutil.process_iter.return_value = [mock_process]

        monitor = DefaultResourceMonitor()
        metrics = monitor.get_process_metrics("python")

        assert metrics.cpu_usage_percent == 15.2
        assert metrics.memory_usage_mb == 256.0

    @patch("src.crackseg.utils.deployment.health_monitoring.psutil")
    def test_get_process_metrics_not_found(self, mock_psutil: Mock) -> None:
        """Test getting process metrics when process is not found."""
        # Mock empty process iteration
        mock_psutil.process_iter.return_value = []

        monitor = DefaultResourceMonitor()
        metrics = monitor.get_process_metrics("nonexistent")

        assert metrics.cpu_usage_percent == 0.0
        assert metrics.memory_usage_mb == 0.0

    @patch("src.crackseg.utils.deployment.health_monitoring.psutil")
    def test_get_system_metrics_exception(self, mock_psutil: Mock) -> None:
        """Test getting system metrics with exception."""
        # Mock exception
        mock_psutil.cpu_percent.side_effect = Exception("CPU error")

        monitor = DefaultResourceMonitor()
        metrics = monitor.get_system_metrics()

        assert metrics.cpu_usage_percent == 0.0
        assert metrics.memory_usage_mb == 0.0


class TestDeploymentHealthMonitor:
    """Test DeploymentHealthMonitor implementation."""

    def test_health_monitor_initialization(self) -> None:
        """Test DeploymentHealthMonitor initialization."""
        monitor = DeploymentHealthMonitor()
        assert monitor.monitoring_active is False
        assert monitor.monitored_deployments == {}
        assert "response_time_ms" in monitor.alert_thresholds

    def test_add_deployment_monitoring(self) -> None:
        """Test adding deployment to monitoring."""
        monitor = DeploymentHealthMonitor()
        monitor.add_deployment_monitoring(
            deployment_id="test-deployment",
            health_check_url="http://localhost:8080/health",
            process_name="test-process",
            check_interval=30,
        )

        assert "test-deployment" in monitor.monitored_deployments
        config = monitor.monitored_deployments["test-deployment"]
        assert config["health_check_url"] == "http://localhost:8080/health"
        assert config["process_name"] == "test-process"
        assert config["check_interval"] == 30

    def test_remove_deployment_monitoring(self) -> None:
        """Test removing deployment from monitoring."""
        monitor = DeploymentHealthMonitor()
        monitor.add_deployment_monitoring(
            deployment_id="test-deployment",
            health_check_url="http://localhost:8080/health",
        )

        assert "test-deployment" in monitor.monitored_deployments

        monitor.remove_deployment_monitoring("test-deployment")
        assert "test-deployment" not in monitor.monitored_deployments

    def test_remove_nonexistent_deployment(self) -> None:
        """Test removing nonexistent deployment."""
        monitor = DeploymentHealthMonitor()
        # Should not raise exception
        monitor.remove_deployment_monitoring("nonexistent")

    def test_get_deployment_status_not_found(self) -> None:
        """Test getting status of nonexistent deployment."""
        monitor = DeploymentHealthMonitor()
        status = monitor.get_deployment_status("nonexistent")
        assert status["error"] == "Deployment not found"

    def test_get_deployment_status_no_checks(self) -> None:
        """Test getting status when no health checks performed."""
        monitor = DeploymentHealthMonitor()
        monitor.add_deployment_monitoring(
            deployment_id="test-deployment",
            health_check_url="http://localhost:8080/health",
        )

        status = monitor.get_deployment_status("test-deployment")
        assert status["status"] == "unknown"
        assert "No health checks performed" in status["message"]

    def test_get_all_deployment_statuses(self) -> None:
        """Test getting all deployment statuses."""
        monitor = DeploymentHealthMonitor()
        monitor.add_deployment_monitoring(
            deployment_id="deployment-1",
            health_check_url="http://localhost:8080/health",
        )
        monitor.add_deployment_monitoring(
            deployment_id="deployment-2",
            health_check_url="http://localhost:8081/health",
        )

        all_statuses = monitor.get_all_deployment_statuses()
        assert len(all_statuses) == 2
        assert "deployment-1" in all_statuses
        assert "deployment-2" in all_statuses

    @patch(
        "src.crackseg.utils.deployment.health_monitoring.asyncio.create_task"
    )
    def test_start_monitoring(self, mock_create_task: Mock) -> None:
        """Test starting health monitoring."""
        monitor = DeploymentHealthMonitor()
        monitor.start_monitoring()

        assert monitor.monitoring_active is True
        mock_create_task.assert_called_once()

    def test_stop_monitoring(self) -> None:
        """Test stopping health monitoring."""
        monitor = DeploymentHealthMonitor()
        monitor.monitoring_active = True
        monitor.stop_monitoring()

        assert monitor.monitoring_active is False

    def test_check_alerts(self) -> None:
        """Test alert checking functionality."""
        monitor = DeploymentHealthMonitor()

        # Test with healthy result
        healthy_result = HealthCheckResult(
            success=True,
            status="healthy",
            response_time_ms=100.0,
            timestamp=time.time(),
        )
        resource_metrics = ResourceMetrics(
            cpu_usage_percent=50.0,
            memory_usage_mb=512.0,
            disk_usage_percent=30.0,
            network_io_mbps=5.0,
            timestamp=time.time(),
        )

        # Should not generate alerts
        with patch.object(monitor.logger, "warning") as mock_warning:
            monitor._check_alerts(
                "test-deployment", healthy_result, resource_metrics
            )
            mock_warning.assert_not_called()

        # Test with unhealthy result
        unhealthy_result = HealthCheckResult(
            success=False,
            status="unhealthy",
            response_time_ms=2000.0,
            timestamp=time.time(),
            error_message="Connection failed",
        )

        # Should generate alerts
        with patch.object(monitor.logger, "warning") as mock_warning:
            monitor._check_alerts(
                "test-deployment", unhealthy_result, resource_metrics
            )
            mock_warning.assert_called()

    def test_export_monitoring_data(self, tmp_path: Path) -> None:
        """Test exporting monitoring data."""
        monitor = DeploymentHealthMonitor()
        monitor.add_deployment_monitoring(
            deployment_id="test-deployment",
            health_check_url="http://localhost:8080/health",
        )

        export_path = tmp_path / "monitoring_data.json"
        monitor.export_monitoring_data(export_path)

        assert export_path.exists()

        # Verify JSON content
        with open(export_path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "monitored_deployments" in data
        assert "test-deployment" in data["monitored_deployments"]

    def test_export_monitoring_data_error(self, tmp_path: Path) -> None:
        """Test exporting monitoring data with error."""
        monitor = DeploymentHealthMonitor()

        # Try to export to invalid path
        invalid_path = tmp_path / "nonexistent" / "data.json"

        with patch.object(monitor.logger, "error") as mock_error:
            monitor.export_monitoring_data(invalid_path)
            mock_error.assert_called_once()


@pytest.mark.asyncio
class TestDeploymentHealthMonitorAsync:
    """Test async functionality of DeploymentHealthMonitor."""

    async def test_monitoring_loop_basic(self) -> None:
        """Test basic monitoring loop functionality."""
        monitor = DeploymentHealthMonitor()
        monitor.add_deployment_monitoring(
            deployment_id="test-deployment",
            health_check_url="http://localhost:8080/health",
            check_interval=1,  # Short interval for testing
        )

        # Start monitoring
        monitor.start_monitoring()

        # Let it run for a short time
        await asyncio.sleep(0.1)

        # Stop monitoring
        monitor.stop_monitoring()

        # Verify monitoring was active
        assert monitor.monitoring_active is False

    async def test_monitoring_loop_with_exception(self) -> None:
        """Test monitoring loop handles exceptions gracefully."""
        monitor = DeploymentHealthMonitor()

        # Mock health checker to raise exception
        mock_checker = Mock()
        mock_checker.check_health.side_effect = Exception(
            "Health check failed"
        )
        monitor.health_checker = mock_checker

        monitor.add_deployment_monitoring(
            deployment_id="test-deployment",
            health_check_url="http://localhost:8080/health",
            check_interval=1,
        )

        # Start monitoring
        monitor.start_monitoring()

        # Let it run for a short time
        await asyncio.sleep(0.1)

        # Stop monitoring
        monitor.stop_monitoring()

        # Should not crash and should stop gracefully
        assert monitor.monitoring_active is False


class TestHealthMonitoringIntegration:
    """Test integration between health monitoring components."""

    def test_health_monitor_with_custom_checker(self) -> None:
        """Test health monitor with custom health checker."""
        # Create custom health checker
        custom_checker = Mock()
        custom_checker.check_health.return_value = HealthCheckResult(
            success=True,
            status="healthy",
            response_time_ms=50.0,
            timestamp=time.time(),
        )

        # Create health monitor with custom checker
        monitor = DeploymentHealthMonitor(health_checker=custom_checker)
        monitor.add_deployment_monitoring(
            deployment_id="test-deployment",
            health_check_url="http://localhost:8080/health",
        )

        # Verify custom checker is used
        assert monitor.health_checker is custom_checker

    def test_health_monitor_with_custom_resource_monitor(self) -> None:
        """Test health monitor with custom resource monitor."""
        # Create custom resource monitor
        custom_monitor = Mock()
        custom_monitor.get_system_metrics.return_value = ResourceMetrics(
            cpu_usage_percent=25.0,
            memory_usage_mb=1024.0,
            disk_usage_percent=50.0,
            network_io_mbps=10.0,
            timestamp=time.time(),
        )

        # Create health monitor with custom resource monitor
        monitor = DeploymentHealthMonitor(resource_monitor=custom_monitor)
        monitor.add_deployment_monitoring(
            deployment_id="test-deployment",
            health_check_url="http://localhost:8080/health",
        )

        # Verify custom monitor is used
        assert monitor.resource_monitor is custom_monitor
