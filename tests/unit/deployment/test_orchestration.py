"""Unit tests for deployment orchestration and rollback mechanisms.

This module provides comprehensive unit tests for the DeploymentOrchestrator
class, covering all deployment strategies, rollback mechanisms, and edge cases
as required by subtask 5.5.
"""

import time
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

from crackseg.utils.deployment.config import DeploymentConfig
from crackseg.utils.deployment.orchestration import (
    DefaultHealthChecker,
    DeploymentMetadata,
    DeploymentOrchestrator,
    DeploymentState,
    DeploymentStrategy,
)


@dataclass
class MockDeploymentResult:
    """Mock deployment result for testing."""

    success: bool = True
    deployment_id: str = "test-deployment-001"
    artifact_id: str = "test-artifact"
    target_environment: str = "production"
    health_check_url: str | None = "http://localhost:8501/healthz"
    deployment_url: str | None = "http://localhost:8501"
    error_message: str | None = None


class TestDeploymentOrchestrator:
    """Test DeploymentOrchestrator core functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        return DeploymentOrchestrator()

    @pytest.fixture
    def config(self):
        """Create deployment config for testing."""
        return DeploymentConfig(
            artifact_id="test-model-v1",
            target_environment="production",
            deployment_type="container",
            enable_quantization=True,
            target_format="onnx",
        )

    @pytest.fixture
    def mock_deployment_func(self):
        """Create mock deployment function."""
        return Mock()

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert hasattr(orchestrator, "health_checker")
        assert hasattr(orchestrator, "deployment_history")
        assert hasattr(orchestrator, "logger")

    def test_deploy_with_strategy_success(
        self, orchestrator, config, mock_deployment_func
    ):
        """Test successful deployment with strategy."""
        mock_deployment_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            result = orchestrator.deploy_with_strategy(
                config, DeploymentStrategy.BLUE_GREEN, mock_deployment_func
            )

        assert result.success
        assert "test-model-v1" in result.deployment_id
        assert result.artifact_id == "test-model-v1"

    def test_deploy_with_strategy_failure(
        self, orchestrator, config, mock_deployment_func
    ):
        """Test deployment failure with automatic rollback."""
        mock_deployment_func.side_effect = RuntimeError("Deployment failed")

        # Mock previous deployment for rollback
        with patch.object(
            orchestrator,
            "_find_current_deployment",
            return_value="prev-deployment",
        ):
            with patch.object(
                orchestrator.health_checker,
                "wait_for_healthy",
                return_value=True,
            ):
                result = orchestrator.deploy_with_strategy(
                    config, DeploymentStrategy.BLUE_GREEN, mock_deployment_func
                )

        assert not result.success
        assert "Deployment failed" in result.error_message

        # Verify rollback was attempted
        deployment_id = result.deployment_id
        status = orchestrator.get_deployment_status(deployment_id)
        assert status["state"] == DeploymentState.ROLLED_BACK.value

    def test_blue_green_deployment_success(
        self, orchestrator, config, mock_deployment_func
    ):
        """Test blue-green deployment success path."""
        mock_deployment_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            with patch.object(
                orchestrator, "_find_current_deployment", return_value=None
            ):
                with patch.object(
                    orchestrator, "_switch_traffic"
                ) as mock_switch:
                    result = orchestrator._blue_green_deploy(
                        config, mock_deployment_func, Mock()
                    )

        assert result.success
        mock_switch.assert_called_once()

    def test_blue_green_deployment_health_check_failure(
        self, orchestrator, config, mock_deployment_func
    ):
        """Test blue-green deployment with health check failure."""
        mock_deployment_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=False
        ):
            with pytest.raises(RuntimeError, match="Health check failed"):
                orchestrator._blue_green_deploy(
                    config, mock_deployment_func, Mock()
                )

    def test_canary_deployment_success(
        self, orchestrator, config, mock_deployment_func
    ):
        """Test canary deployment success path."""
        mock_deployment_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            with patch.object(
                orchestrator, "_monitor_canary_performance", return_value=True
            ):
                with patch.object(
                    orchestrator, "_update_traffic_split"
                ) as mock_update:
                    result = orchestrator._canary_deploy(
                        config, mock_deployment_func, Mock()
                    )

        assert result.success
        mock_update.assert_called()

    def test_rolling_deployment_success(
        self, orchestrator, config, mock_deployment_func
    ):
        """Test rolling deployment success path."""
        mock_deployment_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            with patch.object(
                orchestrator, "_get_current_replicas", return_value=3
            ):
                with patch.object(
                    orchestrator, "_remove_old_replica"
                ) as mock_remove:
                    result = orchestrator._rolling_deploy(
                        config, mock_deployment_func, Mock()
                    )

        assert result.success
        mock_remove.assert_called()

    def test_recreate_deployment_success(
        self, orchestrator, config, mock_deployment_func
    ):
        """Test recreate deployment success path."""
        mock_deployment_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            with patch.object(
                orchestrator, "_remove_current_deployment"
            ) as mock_remove:
                result = orchestrator._recreate_deploy(
                    config, mock_deployment_func, Mock()
                )

        assert result.success
        mock_remove.assert_called_once()

    def test_attempt_rollback_success(self, orchestrator):
        """Test successful rollback attempt."""
        metadata = DeploymentMetadata(
            deployment_id="test-deployment",
            artifact_id="test-artifact",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.FAILED,
            start_time=time.time(),
            previous_deployment_id="prev-deployment",
        )

        with patch.object(orchestrator, "_switch_traffic"):
            with patch.object(
                orchestrator.health_checker,
                "wait_for_healthy",
                return_value=True,
            ):
                with patch.object(
                    orchestrator,
                    "deployment_history",
                    {"prev-deployment": metadata},
                ):
                    result = orchestrator._attempt_rollback(
                        metadata, Exception("Test error")
                    )

        assert result is True

    def test_attempt_rollback_no_previous_deployment(self, orchestrator):
        """Test rollback attempt with no previous deployment."""
        metadata = DeploymentMetadata(
            deployment_id="test-deployment",
            artifact_id="test-artifact",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.FAILED,
            start_time=time.time(),
            previous_deployment_id=None,
        )

        result = orchestrator._attempt_rollback(
            metadata, Exception("Test error")
        )
        assert result is False

    def test_attempt_rollback_health_check_failure(self, orchestrator):
        """Test rollback attempt with health check failure."""
        metadata = DeploymentMetadata(
            deployment_id="test-deployment",
            artifact_id="test-artifact",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.FAILED,
            start_time=time.time(),
            previous_deployment_id="prev-deployment",
        )

        with patch.object(orchestrator, "_switch_traffic"):
            with patch.object(
                orchestrator.health_checker,
                "wait_for_healthy",
                return_value=False,
            ):
                with patch.object(
                    orchestrator,
                    "deployment_history",
                    {"prev-deployment": metadata},
                ):
                    result = orchestrator._attempt_rollback(
                        metadata, Exception("Test error")
                    )

        assert result is False

    def test_manual_rollback_success(self, orchestrator):
        """Test successful manual rollback."""
        metadata = DeploymentMetadata(
            deployment_id="test-deployment",
            artifact_id="test-artifact",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.SUCCESS,
            start_time=time.time(),
            previous_deployment_id="prev-deployment",
        )

        orchestrator.deployment_history["test-deployment"] = metadata

        with patch.object(
            orchestrator, "_attempt_rollback", return_value=True
        ):
            result = orchestrator.manual_rollback("test-deployment")

        assert result is True

    def test_manual_rollback_deployment_not_found(self, orchestrator):
        """Test manual rollback with non-existent deployment."""
        result = orchestrator.manual_rollback("non-existent-deployment")
        assert result is False

    def test_get_deployment_status_success(self, orchestrator):
        """Test getting deployment status."""
        metadata = DeploymentMetadata(
            deployment_id="test-deployment",
            artifact_id="test-artifact",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.SUCCESS,
            start_time=time.time(),
            end_time=time.time() + 100,
            previous_deployment_id="prev-deployment",
            health_check_url="http://localhost:8501/healthz",
        )

        orchestrator.deployment_history["test-deployment"] = metadata

        status = orchestrator.get_deployment_status("test-deployment")

        assert status["deployment_id"] == "test-deployment"
        assert status["artifact_id"] == "test-artifact"
        assert status["strategy"] == DeploymentStrategy.BLUE_GREEN.value
        assert status["state"] == DeploymentState.SUCCESS.value
        assert status["duration"] > 0

    def test_get_deployment_status_not_found(self, orchestrator):
        """Test getting status for non-existent deployment."""
        status = orchestrator.get_deployment_status("non-existent-deployment")
        assert "error" in status
        assert status["error"] == "Deployment not found"

    def test_get_deployment_history_empty(self, orchestrator):
        """Test getting deployment history when empty."""
        history = orchestrator.get_deployment_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_deployment_history_with_deployments(self, orchestrator):
        """Test getting deployment history with deployments."""
        metadata1 = DeploymentMetadata(
            deployment_id="deployment-1",
            artifact_id="artifact-1",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.SUCCESS,
            start_time=time.time(),
            end_time=time.time() + 100,
        )

        metadata2 = DeploymentMetadata(
            deployment_id="deployment-2",
            artifact_id="artifact-2",
            strategy=DeploymentStrategy.CANARY,
            state=DeploymentState.FAILED,
            start_time=time.time(),
            end_time=time.time() + 50,
        )

        orchestrator.deployment_history["deployment-1"] = metadata1
        orchestrator.deployment_history["deployment-2"] = metadata2

        history = orchestrator.get_deployment_history()
        assert len(history) == 2

        # Test filtering by artifact_id
        filtered_history = orchestrator.get_deployment_history(
            artifact_id="artifact-1"
        )
        assert len(filtered_history) == 1
        assert filtered_history[0]["artifact_id"] == "artifact-1"


class TestDeploymentStrategies:
    """Test deployment strategy implementations."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return DeploymentOrchestrator()

    @pytest.fixture
    def config(self):
        """Create deployment config."""
        return DeploymentConfig(
            artifact_id="test-model",
            target_environment="production",
            deployment_type="container",
        )

    def test_blue_green_strategy_characteristics(self, orchestrator, config):
        """Test blue-green deployment strategy characteristics."""
        mock_func = Mock()
        mock_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            with patch.object(
                orchestrator, "_find_current_deployment", return_value=None
            ):
                with patch.object(
                    orchestrator, "_switch_traffic"
                ) as mock_switch:
                    result = orchestrator.deploy_with_strategy(
                        config, DeploymentStrategy.BLUE_GREEN, mock_func
                    )

        assert result.success
        mock_switch.assert_called()  # Should switch traffic

    def test_canary_strategy_characteristics(self, orchestrator, config):
        """Test canary deployment strategy characteristics."""
        mock_func = Mock()
        mock_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            with patch.object(
                orchestrator, "_monitor_canary_performance", return_value=True
            ):
                with patch.object(
                    orchestrator, "_update_traffic_split"
                ) as mock_update:
                    result = orchestrator.deploy_with_strategy(
                        config, DeploymentStrategy.CANARY, mock_func
                    )

        assert result.success
        mock_update.assert_called()  # Should update traffic split

    def test_rolling_strategy_characteristics(self, orchestrator, config):
        """Test rolling deployment strategy characteristics."""
        mock_func = Mock()
        mock_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            with patch.object(
                orchestrator, "_get_current_replicas", return_value=3
            ):
                with patch.object(
                    orchestrator, "_remove_old_replica"
                ) as mock_remove:
                    result = orchestrator.deploy_with_strategy(
                        config, DeploymentStrategy.ROLLING, mock_func
                    )

        assert result.success
        mock_remove.assert_called()  # Should remove old replicas

    def test_recreate_strategy_characteristics(self, orchestrator, config):
        """Test recreate deployment strategy characteristics."""
        mock_func = Mock()
        mock_func.return_value = MockDeploymentResult()

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            with patch.object(
                orchestrator, "_remove_current_deployment"
            ) as mock_remove:
                result = orchestrator.deploy_with_strategy(
                    config, DeploymentStrategy.RECREATE, mock_func
                )

        assert result.success
        mock_remove.assert_called_once()  # Should remove current deployment


class TestDefaultHealthChecker:
    """Test DefaultHealthChecker implementation."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker instance."""
        return DefaultHealthChecker()

    def test_initialization(self, health_checker):
        """Test health checker initialization."""
        assert health_checker is not None
        assert hasattr(health_checker, "logger")

    @patch("requests.get")
    def test_check_health_success(self, mock_get, health_checker):
        """Test successful health check."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status.return_value = None

        result = health_checker.check_health("http://localhost:8501/healthz")
        assert result is True

    @patch("requests.get")
    def test_check_health_failure(self, mock_get, health_checker):
        """Test failed health check."""
        mock_get.side_effect = Exception("Connection failed")

        result = health_checker.check_health("http://localhost:8501/healthz")
        assert result is False

    @patch("requests.get")
    def test_wait_for_healthy_success(self, mock_get, health_checker):
        """Test successful wait for healthy."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status.return_value = None

        result = health_checker.wait_for_healthy(
            "http://localhost:8501/healthz", max_wait=5
        )
        assert result is True

    @patch("requests.get")
    def test_wait_for_healthy_timeout(self, mock_get, health_checker):
        """Test wait for healthy timeout."""
        mock_get.side_effect = Exception("Connection failed")

        result = health_checker.wait_for_healthy(
            "http://localhost:8501/healthz", max_wait=1
        )
        assert result is False


class TestDeploymentStates:
    """Test deployment state management."""

    def test_deployment_state_enum_values(self):
        """Test deployment state enum values."""
        assert DeploymentState.PENDING.value == "pending"
        assert DeploymentState.IN_PROGRESS.value == "in-progress"
        assert DeploymentState.HEALTH_CHECKING.value == "health-checking"
        assert DeploymentState.SUCCESS.value == "success"
        assert DeploymentState.FAILED.value == "failed"
        assert DeploymentState.ROLLING_BACK.value == "rolling-back"
        assert DeploymentState.ROLLED_BACK.value == "rolled-back"

    def test_deployment_strategy_enum_values(self):
        """Test deployment strategy enum values."""
        assert DeploymentStrategy.BLUE_GREEN.value == "blue-green"
        assert DeploymentStrategy.CANARY.value == "canary"
        assert DeploymentStrategy.ROLLING.value == "rolling"
        assert DeploymentStrategy.RECREATE.value == "recreate"


class TestDeploymentMetadata:
    """Test deployment metadata functionality."""

    def test_deployment_metadata_creation(self):
        """Test deployment metadata creation."""
        metadata = DeploymentMetadata(
            deployment_id="test-deployment",
            artifact_id="test-artifact",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.PENDING,
            start_time=time.time(),
        )

        assert metadata.deployment_id == "test-deployment"
        assert metadata.artifact_id == "test-artifact"
        assert metadata.strategy == DeploymentStrategy.BLUE_GREEN
        assert metadata.state == DeploymentState.PENDING
        assert metadata.start_time > 0
        assert metadata.end_time is None
        assert metadata.previous_deployment_id is None
        assert metadata.rollback_reason is None

    def test_deployment_metadata_with_optional_fields(self):
        """Test deployment metadata with optional fields."""
        metadata = DeploymentMetadata(
            deployment_id="test-deployment",
            artifact_id="test-artifact",
            strategy=DeploymentStrategy.BLUE_GREEN,
            state=DeploymentState.SUCCESS,
            start_time=time.time(),
            end_time=time.time() + 100,
            previous_deployment_id="prev-deployment",
            rollback_reason="Health check failed",
            health_check_url="http://localhost:8501/healthz",
            metrics_url="http://localhost:8501/metrics",
        )

        assert metadata.end_time is not None
        assert metadata.previous_deployment_id == "prev-deployment"
        assert metadata.rollback_reason == "Health check failed"
        assert metadata.health_check_url == "http://localhost:8501/healthz"
        assert metadata.metrics_url == "http://localhost:8501/metrics"
