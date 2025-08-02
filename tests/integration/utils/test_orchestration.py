"""Integration tests for deployment orchestration system.

This module tests the deployment orchestration capabilities including
blue-green deployments, canary releases, rolling updates, and rollback
mechanisms.
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.crackseg.utils.deployment import (
    DefaultHealthChecker,
    DeploymentConfig,
    DeploymentMetadata,
    DeploymentOrchestrator,
    DeploymentState,
    DeploymentStrategy,
    EmailAlertHandler,
    LoggingAlertHandler,
    PerformanceMonitor,
    SlackAlertHandler,
)


class TestDeploymentOrchestrator:
    """Test deployment orchestrator functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.orchestrator = DeploymentOrchestrator()
        self.config = DeploymentConfig(
            artifact_id="test-model-v1",
            target_environment="production",
            deployment_type="container",
            enable_quantization=True,
            target_format="onnx",
        )

    def test_blue_green_deployment_success(self) -> None:
        """Test successful blue-green deployment."""
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

        assert result.success
        assert "test-model-v1" in result.deployment_id
        assert result.artifact_id == "test-model-v1"
        assert result.target_environment == "production"

    def test_blue_green_deployment_health_check_failure(self) -> None:
        """Test blue-green deployment with health check failure."""
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=False,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

        assert not result.success
        assert result.error_message is not None
        assert "Health check failed" in result.error_message

    def test_canary_deployment_success(self) -> None:
        """Test successful canary deployment."""
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            with patch.object(
                self.orchestrator.health_checker,
                "check_health",
                return_value=True,
            ):
                with patch.object(
                    self.orchestrator,
                    "_monitor_canary_performance",
                    return_value=True,
                ):
                    result = self.orchestrator.deploy_with_strategy(
                        self.config,
                        DeploymentStrategy.CANARY,
                        mock_deployment_func,
                    )

        assert result.success
        assert result.artifact_id == "test-model-v1"

    def test_rolling_deployment_success(self) -> None:
        """Test successful rolling deployment."""
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            with patch.object(
                self.orchestrator, "_get_current_replicas", return_value=3
            ):
                result = self.orchestrator.deploy_with_strategy(
                    self.config,
                    DeploymentStrategy.ROLLING,
                    mock_deployment_func,
                )

        assert result.success
        assert result.artifact_id == "test-model-v1"

    def test_recreate_deployment_success(self) -> None:
        """Test successful recreate deployment."""
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.RECREATE,
                mock_deployment_func,
            )

        assert result.success
        assert result.artifact_id == "test-model-v1"

    def test_deployment_failure_with_rollback(self) -> None:
        """Test deployment failure with automatic rollback."""
        mock_deployment_func = Mock()
        mock_deployment_func.side_effect = RuntimeError("Deployment failed")

        # Mock previous deployment for rollback
        with patch.object(
            self.orchestrator,
            "_find_current_deployment",
            return_value="previous-deployment",
        ):
            with patch.object(
                self.orchestrator.health_checker,
                "wait_for_healthy",
                return_value=True,
            ):
                result = self.orchestrator.deploy_with_strategy(
                    self.config,
                    DeploymentStrategy.BLUE_GREEN,
                    mock_deployment_func,
                )

        assert not result.success
        assert result.error_message is not None
        assert "Deployment failed" in result.error_message

        # Check that rollback was attempted
        deployment_id = result.deployment_id
        status = self.orchestrator.get_deployment_status(deployment_id)
        assert status["state"] == DeploymentState.ROLLED_BACK.value

    def test_manual_rollback(self) -> None:
        """Test manual rollback functionality."""
        # Create a deployment first
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

        deployment_id = result.deployment_id

        # Mock previous deployment for rollback
        with patch.object(
            self.orchestrator,
            "_find_current_deployment",
            return_value="previous-deployment",
        ):
            with patch.object(
                self.orchestrator.health_checker,
                "wait_for_healthy",
                return_value=True,
            ):
                rollback_success = self.orchestrator.manual_rollback(
                    deployment_id
                )

        assert rollback_success

    def test_deployment_status_tracking(self) -> None:
        """Test deployment status tracking."""
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

        deployment_id = result.deployment_id
        status = self.orchestrator.get_deployment_status(deployment_id)

        assert status["deployment_id"] == deployment_id
        assert status["artifact_id"] == "test-model-v1"
        assert status["strategy"] == DeploymentStrategy.BLUE_GREEN.value
        assert status["state"] == DeploymentState.SUCCESS.value
        assert status["start_time"] > 0
        assert status["end_time"] > status["start_time"]
        assert status["duration"] > 0

    def test_deployment_history(self) -> None:
        """Test deployment history functionality."""
        # Create multiple deployments
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            # First deployment
            self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

            # Second deployment with different artifact
            config2 = DeploymentConfig(
                artifact_id="test-model-v2",
                target_environment="production",
                deployment_type="container",
            )
            self.orchestrator.deploy_with_strategy(
                config2,
                DeploymentStrategy.CANARY,
                mock_deployment_func,
            )

        # Get all deployment history
        history = self.orchestrator.get_deployment_history()
        assert len(history) >= 2

        # Get filtered history
        filtered_history = self.orchestrator.get_deployment_history(
            "test-model-v1"
        )
        assert len(filtered_history) >= 1
        assert all(
            deployment["artifact_id"] == "test-model-v1"
            for deployment in filtered_history
        )

    def test_health_checker_functionality(self) -> None:
        """Test health checker functionality."""
        health_checker = DefaultHealthChecker()

        # Test health check with mock requests
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            assert health_checker.check_health("http://localhost:8501")

            mock_get.return_value.status_code = 500
            assert not health_checker.check_health("http://localhost:8501")

            mock_get.side_effect = Exception("Connection failed")
            assert not health_checker.check_health("http://localhost:8501")

    def test_deployment_metadata_creation(self) -> None:
        """Test deployment metadata creation and tracking."""
        deployment_id = "test-deployment-123"
        metadata = self.orchestrator.deployment_history.get(deployment_id)

        # Create a deployment to generate metadata
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

        deployment_id = result.deployment_id
        metadata = self.orchestrator.deployment_history.get(deployment_id)

        assert metadata is not None
        assert metadata.deployment_id == deployment_id
        assert metadata.artifact_id == "test-model-v1"
        assert metadata.strategy == DeploymentStrategy.BLUE_GREEN
        assert metadata.state == DeploymentState.SUCCESS
        assert metadata.start_time > 0
        assert metadata.end_time is not None
        assert metadata.end_time > metadata.start_time


class TestDeploymentStrategies:
    """Test different deployment strategies."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.orchestrator = DeploymentOrchestrator()
        self.config = DeploymentConfig(
            artifact_id="test-model",
            target_environment="production",
            deployment_type="container",
        )

    def test_blue_green_strategy_characteristics(self) -> None:
        """Test blue-green deployment strategy characteristics."""
        assert DeploymentStrategy.BLUE_GREEN.value == "blue-green"

        # Blue-green should have zero downtime
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

        assert result.success

    def test_canary_strategy_characteristics(self) -> None:
        """Test canary deployment strategy characteristics."""
        assert DeploymentStrategy.CANARY.value == "canary"

        # Canary should gradually increase traffic
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            with patch.object(
                self.orchestrator.health_checker,
                "check_health",
                return_value=True,
            ):
                with patch.object(
                    self.orchestrator,
                    "_monitor_canary_performance",
                    return_value=True,
                ):
                    result = self.orchestrator.deploy_with_strategy(
                        self.config,
                        DeploymentStrategy.CANARY,
                        mock_deployment_func,
                    )

        assert result.success

    def test_rolling_strategy_characteristics(self) -> None:
        """Test rolling deployment strategy characteristics."""
        assert DeploymentStrategy.ROLLING.value == "rolling"

        # Rolling should update replicas one by one
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            with patch.object(
                self.orchestrator, "_get_current_replicas", return_value=3
            ):
                result = self.orchestrator.deploy_with_strategy(
                    self.config,
                    DeploymentStrategy.ROLLING,
                    mock_deployment_func,
                )

        assert result.success

    def test_recreate_strategy_characteristics(self) -> None:
        """Test recreate deployment strategy characteristics."""
        assert DeploymentStrategy.RECREATE.value == "recreate"

        # Recreate should remove old deployment before creating new one
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.RECREATE,
                mock_deployment_func,
            )

        assert result.success


class TestDeploymentStates:
    """Test deployment state management."""

    def test_deployment_state_transitions(self) -> None:
        """Test deployment state transitions."""
        orchestrator = DeploymentOrchestrator()
        config = DeploymentConfig(
            artifact_id="test-model",
            target_environment="production",
            deployment_type="container",
        )

        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
        )

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            result = orchestrator.deploy_with_strategy(
                config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

        deployment_id = result.deployment_id
        status = orchestrator.get_deployment_status(deployment_id)

        # Verify state transitions
        assert status["state"] == DeploymentState.SUCCESS.value

    def test_failed_deployment_state(self) -> None:
        """Test failed deployment state."""
        orchestrator = DeploymentOrchestrator()
        config = DeploymentConfig(
            artifact_id="test-model",
            target_environment="production",
            deployment_type="container",
        )

        mock_deployment_func = Mock()
        mock_deployment_func.side_effect = RuntimeError("Deployment failed")

        result = orchestrator.deploy_with_strategy(
            config,
            DeploymentStrategy.BLUE_GREEN,
            mock_deployment_func,
        )

        deployment_id = result.deployment_id
        status = orchestrator.get_deployment_status(deployment_id)

        # Should be either FAILED or ROLLED_BACK
        assert status["state"] in [
            DeploymentState.FAILED.value,
            DeploymentState.ROLLED_BACK.value,
        ]


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor("http://localhost:8501")

    def test_performance_monitor_initialization(self) -> None:
        """Test performance monitor initialization."""
        assert self.monitor.deployment_url == "http://localhost:8501"
        assert not self.monitor.monitoring
        assert self.monitor.metrics["response_time"] == []
        assert self.monitor.metrics["throughput"] == []

    def test_metrics_collection(self) -> None:
        """Test metrics collection functionality."""
        # Simulate metrics collection
        self.monitor.metrics["response_time"].extend([100, 110, 105, 95, 115])
        self.monitor.metrics["throughput"].extend([10, 9, 9.5, 10.5, 8.7])
        self.monitor.metrics["error_rate"].extend(
            [0.01, 0.02, 0.01, 0.01, 0.03]
        )
        self.monitor.metrics["memory_usage"].extend([512, 520, 515, 510, 525])
        self.monitor.metrics["cpu_usage"].extend([25, 28, 26, 24, 30])

        metrics = self.monitor.get_metrics()

        assert "current" in metrics
        assert "average" in metrics
        assert "trends" in metrics

        current = metrics["current"]
        assert current["response_time_ms"] == 115
        assert current["throughput_rps"] == 8.7
        assert current["error_rate"] == 0.03
        assert current["memory_usage_mb"] == 525
        assert current["cpu_usage_percent"] == 30

    def test_trend_calculation(self) -> None:
        """Test trend calculation functionality."""
        # Test improving trend
        improving_values = [200, 190, 180, 170, 160, 150, 140, 130, 120, 110]
        trend = self.monitor._calculate_trend(improving_values)
        assert trend == "improving"

        # Test degrading trend
        degrading_values = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        trend = self.monitor._calculate_trend(degrading_values)
        assert trend == "degrading"

        # Test stable trend
        stable_values = [100, 105, 98, 102, 99, 101, 103, 97, 104, 100]
        trend = self.monitor._calculate_trend(stable_values)
        assert trend == "stable"

    def test_monitoring_start_stop(self) -> None:
        """Test monitoring start and stop functionality."""
        assert not self.monitor.monitoring

        # Start monitoring
        self.monitor.start_monitoring("test-deployment")
        assert self.monitor.monitoring
        assert self.monitor.monitor_thread is not None

        # Stop monitoring
        self.monitor.stop_monitoring()
        assert not self.monitor.monitoring


class TestAlertHandlers:
    """Test alert handler functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.metadata = DeploymentMetadata(
            deployment_id="test-deployment-123",
            artifact_id="test-model-v1",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.SUCCESS,
            start_time=time.time() - 60,
            end_time=time.time(),
        )

    def test_logging_alert_handler(self) -> None:
        """Test logging alert handler."""
        handler = LoggingAlertHandler()

        # Test success alert
        with patch.object(handler.logger, "info") as mock_info:
            handler.send_alert("deployment_success", self.metadata)
            mock_info.assert_called_once()

        # Test failure alert
        with patch.object(handler.logger, "error") as mock_error:
            handler.send_alert(
                "deployment_failure", self.metadata, error="Test error"
            )
            mock_error.assert_called_once()

    def test_email_alert_handler(self) -> None:
        """Test email alert handler."""
        handler = EmailAlertHandler(
            smtp_server="localhost",
            smtp_port=587,
            username="test@example.com",
            password="password",
        )

        # Test alert sending (should fail in test environment)
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server

            handler.send_alert("deployment_success", self.metadata)

            mock_smtp.assert_called_once_with("localhost", 587)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with(
                "test@example.com", "password"
            )

    def test_slack_alert_handler(self) -> None:
        """Test Slack alert handler."""
        handler = SlackAlertHandler("https://hooks.slack.com/test")

        # Test alert sending
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            handler.send_alert("deployment_success", self.metadata)

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert (
                call_args[1]["headers"]["Content-Type"] == "application/json"
            )


class TestAdvancedOrchestration:
    """Test advanced orchestration features."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.orchestrator = DeploymentOrchestrator()
        self.config = DeploymentConfig(
            artifact_id="test-model-v1",
            target_environment="production",
            deployment_type="container",
            enable_quantization=True,
            target_format="onnx",
        )

    def test_alert_handler_registration(self) -> None:
        """Test alert handler registration."""
        handler = LoggingAlertHandler()
        self.orchestrator.add_alert_handler(handler)

        assert len(self.orchestrator.alert_handlers) == 1
        assert self.orchestrator.alert_handlers[0] == handler

    def test_performance_monitor_registration(self) -> None:
        """Test performance monitor registration."""
        monitor = PerformanceMonitor("http://localhost:8501")
        self.orchestrator.add_performance_monitor("test-deployment", monitor)

        assert "test-deployment" in self.orchestrator.performance_monitors
        assert (
            self.orchestrator.performance_monitors["test-deployment"]
            == monitor
        )

    def test_performance_metrics_retrieval(self) -> None:
        """Test performance metrics retrieval."""
        monitor = PerformanceMonitor("http://localhost:8501")
        monitor.metrics["response_time"] = [100, 110, 105]
        monitor.metrics["throughput"] = [10, 9, 9.5]

        self.orchestrator.add_performance_monitor("test-deployment", monitor)

        metrics = self.orchestrator.get_performance_metrics("test-deployment")
        assert "current" in metrics
        assert "average" in metrics
        assert "trends" in metrics

    def test_performance_monitoring_stop(self) -> None:
        """Test stopping performance monitoring."""
        monitor = PerformanceMonitor("http://localhost:8501")
        self.orchestrator.add_performance_monitor("test-deployment", monitor)

        # Start monitoring
        monitor.start_monitoring("test-deployment")
        assert monitor.monitoring

        # Stop monitoring
        self.orchestrator.stop_performance_monitoring("test-deployment")
        assert not monitor.monitoring
        assert "test-deployment" not in self.orchestrator.performance_monitors

    def test_deployment_with_monitoring(self) -> None:
        """Test deployment with performance monitoring."""
        mock_deployment_func = Mock()
        mock_deployment_func.return_value = Mock(
            success=True,
            deployment_url="http://localhost:8501",
            health_check_url="http://localhost:8501/healthz",
        )

        with patch.object(
            self.orchestrator.health_checker,
            "wait_for_healthy",
            return_value=True,
        ):
            result = self.orchestrator.deploy_with_strategy(
                self.config,
                DeploymentStrategy.BLUE_GREEN,
                mock_deployment_func,
            )

        assert result.success
        # Check that performance monitoring was started
        assert result.deployment_id in self.orchestrator.performance_monitors


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
