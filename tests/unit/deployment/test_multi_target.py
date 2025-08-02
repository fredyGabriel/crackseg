"""Unit tests for multi-target deployment system."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.crackseg.utils.deployment.config import (
    DeploymentConfig,
    DeploymentResult,
)
from src.crackseg.utils.deployment.multi_target import (
    EnvironmentConfig,
    MultiTargetDeploymentManager,
    TargetEnvironment,
)
from src.crackseg.utils.deployment.orchestration import DeploymentStrategy


class TestTargetEnvironment:
    """Test TargetEnvironment enum."""

    def test_environment_values(self) -> None:
        """Test that all environment values are valid."""
        environments = list(TargetEnvironment)
        assert len(environments) == 5

        expected_values = [
            "development",
            "staging",
            "production",
            "testing",
            "demo",
        ]
        for env in environments:
            assert env.value in expected_values


class TestEnvironmentConfig:
    """Test EnvironmentConfig dataclass."""

    def test_environment_config_creation(self) -> None:
        """Test creating EnvironmentConfig with default values."""
        config = EnvironmentConfig(
            name=TargetEnvironment.DEVELOPMENT,
            deployment_strategy=DeploymentStrategy.RECREATE,
        )

        assert config.name == TargetEnvironment.DEVELOPMENT
        assert config.deployment_strategy == DeploymentStrategy.RECREATE
        assert config.health_check_timeout == 30
        assert config.max_retries == 3
        assert config.auto_rollback is True
        assert config.performance_thresholds is None
        assert config.resource_limits is None
        assert config.security_requirements is None
        assert config.monitoring_config is None

    def test_environment_config_with_custom_values(self) -> None:
        """Test creating EnvironmentConfig with custom values."""
        performance_thresholds = {"response_time_ms": 500.0}
        resource_limits = {"memory_mb": 2048}
        security_requirements = {"ssl_required": True}
        monitoring_config = {"check_interval": 30}

        config = EnvironmentConfig(
            name=TargetEnvironment.PRODUCTION,
            deployment_strategy=DeploymentStrategy.CANARY,
            health_check_timeout=60,
            max_retries=5,
            auto_rollback=False,
            performance_thresholds=performance_thresholds,
            resource_limits=resource_limits,
            security_requirements=security_requirements,
            monitoring_config=monitoring_config,
        )

        assert config.name == TargetEnvironment.PRODUCTION
        assert config.deployment_strategy == DeploymentStrategy.CANARY
        assert config.health_check_timeout == 60
        assert config.max_retries == 5
        assert config.auto_rollback is False
        assert config.performance_thresholds == performance_thresholds
        assert config.resource_limits == resource_limits
        assert config.security_requirements == security_requirements
        assert config.monitoring_config == monitoring_config


class TestMultiTargetDeploymentManager:
    """Test MultiTargetDeploymentManager class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = MultiTargetDeploymentManager()

    def test_initialization(self) -> None:
        """Test manager initialization."""
        assert len(self.manager.environment_configs) == 5
        assert len(self.manager.orchestrators) == 0

        # Check that all environments have configs
        for environment in TargetEnvironment:
            assert environment in self.manager.environment_configs

    def test_get_environment_config(self) -> None:
        """Test getting environment configuration."""
        for environment in TargetEnvironment:
            config = self.manager.get_environment_config(environment)
            assert isinstance(config, EnvironmentConfig)
            assert config.name == environment

    def test_get_environment_config_invalid(self) -> None:
        """Test getting configuration for invalid environment."""
        with pytest.raises(ValueError, match="Unknown environment"):
            # Create a mock environment that doesn't exist
            mock_env = Mock()
            mock_env.value = "invalid"
            self.manager.get_environment_config(mock_env)

    def test_get_orchestrator(self) -> None:
        """Test getting orchestrator for environment."""
        environment = TargetEnvironment.DEVELOPMENT

        # First call should create orchestrator
        orchestrator1 = self.manager.get_orchestrator(environment)
        assert environment in self.manager.orchestrators
        assert orchestrator1 is self.manager.orchestrators[environment]

        # Second call should return same orchestrator
        orchestrator2 = self.manager.get_orchestrator(environment)
        assert orchestrator1 is orchestrator2

    def test_deploy_to_environment(self) -> None:
        """Test deploying to specific environment."""
        config = DeploymentConfig(
            artifact_id="test-model",
            target_environment="development",
        )

        def mock_deployment_func(
            config: DeploymentConfig, **kwargs
        ) -> DeploymentResult:
            return DeploymentResult(
                success=True,
                deployment_id="test-deploy-123",
                artifact_id=config.artifact_id,
                target_environment=config.target_environment,
            )

        result = self.manager.deploy_to_environment(
            config=config,
            environment=TargetEnvironment.DEVELOPMENT,
            deployment_func=mock_deployment_func,
        )

        assert result.success
        assert result.deployment_id == "test-deploy-123"
        assert result.artifact_id == "test-model"
        assert result.target_environment == "development"

    def test_deploy_to_multiple_environments(self) -> None:
        """Test deploying to multiple environments."""
        config = DeploymentConfig(
            artifact_id="test-model",
            target_environment="development",
        )

        def mock_deployment_func(
            config: DeploymentConfig, **kwargs
        ) -> DeploymentResult:
            return DeploymentResult(
                success=True,
                deployment_id=f"deploy-{config.target_environment}",
                artifact_id=config.artifact_id,
                target_environment=config.target_environment,
            )

        environments = [
            TargetEnvironment.DEVELOPMENT,
            TargetEnvironment.STAGING,
        ]
        results = self.manager.deploy_to_multiple_environments(
            config=config,
            environments=environments,
            deployment_func=mock_deployment_func,
        )

        assert len(results) == 2
        for environment in environments:
            assert environment in results
            assert results[environment].success
            assert results[environment].target_environment == environment.value

    def test_deploy_to_multiple_environments_with_failure(self) -> None:
        """Test deploying to multiple environments with some failures."""
        config = DeploymentConfig(
            artifact_id="test-model",
            target_environment="development",
        )

        def mock_deployment_func(
            config: DeploymentConfig, **kwargs
        ) -> DeploymentResult:
            # Simulate failure for staging
            if config.target_environment == "staging":
                return DeploymentResult(
                    success=False,
                    deployment_id="failed-staging",
                    artifact_id=config.artifact_id,
                    target_environment=config.target_environment,
                    error_message="Mock failure",
                )
            else:
                return DeploymentResult(
                    success=True,
                    deployment_id=f"deploy-{config.target_environment}",
                    artifact_id=config.artifact_id,
                    target_environment=config.target_environment,
                )

        environments = [
            TargetEnvironment.DEVELOPMENT,
            TargetEnvironment.STAGING,
        ]
        results = self.manager.deploy_to_multiple_environments(
            config=config,
            environments=environments,
            deployment_func=mock_deployment_func,
        )

        assert len(results) == 2
        assert results[TargetEnvironment.DEVELOPMENT].success
        assert not results[TargetEnvironment.STAGING].success
        assert (
            results[TargetEnvironment.STAGING].error_message == "Mock failure"
        )

    @patch("src.crackseg.utils.deployment.multi_target.psutil")
    def test_validate_environment_readiness_with_psutil(
        self, mock_psutil
    ) -> None:
        """Test environment validation with psutil available."""
        # Mock psutil responses
        mock_psutil.virtual_memory.return_value.total = (
            16 * 1024 * 1024 * 1024
        )  # 16GB
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.disk_usage.return_value.free = (
            100 * 1024 * 1024 * 1024
        )  # 100GB

        validation = self.manager.validate_environment_readiness(
            TargetEnvironment.DEVELOPMENT
        )

        assert validation["environment"] == "development"
        assert validation["ready"] is True
        assert len(validation["issues"]) == 0

    def test_validate_environment_readiness_without_psutil(self) -> None:
        """Test environment validation without psutil."""
        with patch("src.crackseg.utils.deployment.multi_target.psutil", None):
            validation = self.manager.validate_environment_readiness(
                TargetEnvironment.DEVELOPMENT
            )

            assert validation["environment"] == "development"
            assert validation["ready"] is True
            assert len(validation["warnings"]) > 0
            assert any(
                "psutil not available" in warning
                for warning in validation["warnings"]
            )

    def test_get_deployment_status_across_environments(self) -> None:
        """Test getting deployment status across environments."""
        # Mock orchestrator responses
        mock_orchestrator = Mock()
        mock_orchestrator.get_deployment_status.return_value = {
            "status": "success"
        }

        self.manager.orchestrators[TargetEnvironment.DEVELOPMENT] = (
            mock_orchestrator
        )

        statuses = self.manager.get_deployment_status_across_environments(
            "test-deploy-123"
        )

        assert TargetEnvironment.DEVELOPMENT in statuses
        assert statuses[TargetEnvironment.DEVELOPMENT]["status"] == "success"

    def test_rollback_across_environments(self) -> None:
        """Test rollback across environments."""
        # Mock orchestrator responses
        mock_orchestrator = Mock()
        mock_orchestrator.manual_rollback.return_value = True

        self.manager.orchestrators[TargetEnvironment.DEVELOPMENT] = (
            mock_orchestrator
        )

        results = self.manager.rollback_across_environments(
            "test-deploy-123", [TargetEnvironment.DEVELOPMENT]
        )

        assert TargetEnvironment.DEVELOPMENT in results
        assert results[TargetEnvironment.DEVELOPMENT] is True

    def test_export_environment_configs(self) -> None:
        """Test exporting environment configurations."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as tmp_file:
            export_path = Path(tmp_file.name)

        try:
            self.manager.export_environment_configs(export_path)

            assert export_path.exists()

            # Verify exported content
            with open(export_path) as f:
                configs = json.load(f)

            assert "development" in configs
            assert "production" in configs
            assert "staging" in configs

            # Check structure of exported config
            dev_config = configs["development"]
            assert "deployment_strategy" in dev_config
            assert "health_check_timeout" in dev_config
            assert "max_retries" in dev_config
            assert "auto_rollback" in dev_config

        finally:
            # Cleanup
            if export_path.exists():
                export_path.unlink()

    def test_environment_configurations(self) -> None:
        """Test that all environments have appropriate configurations."""
        for environment in TargetEnvironment:
            config = self.manager.get_environment_config(environment)

            # Check that each environment has a valid configuration
            assert config.name == environment
            assert isinstance(config.deployment_strategy, DeploymentStrategy)
            assert config.health_check_timeout > 0
            assert config.max_retries >= 0
            assert isinstance(config.auto_rollback, bool)

            # Check environment-specific configurations
            if environment == TargetEnvironment.PRODUCTION:
                assert config.deployment_strategy == DeploymentStrategy.CANARY
                assert config.auto_rollback is True
                assert config.max_retries >= 3
            elif environment == TargetEnvironment.DEVELOPMENT:
                assert (
                    config.deployment_strategy == DeploymentStrategy.RECREATE
                )
                assert config.auto_rollback is False
                assert config.max_retries <= 1

    def test_performance_thresholds(self) -> None:
        """Test that performance thresholds are reasonable."""
        for environment in TargetEnvironment:
            config = self.manager.get_environment_config(environment)

            if config.performance_thresholds:
                thresholds = config.performance_thresholds

                # Check that thresholds are positive
                for metric, threshold in thresholds.items():
                    assert (
                        threshold > 0
                    ), f"Threshold for {metric} should be positive"

                # Check specific thresholds
                if "response_time_ms" in thresholds:
                    assert thresholds["response_time_ms"] > 0
                    assert (
                        thresholds["response_time_ms"] <= 5000
                    )  # Max 5 seconds

                if "memory_usage_mb" in thresholds:
                    assert thresholds["memory_usage_mb"] > 0
                    assert thresholds["memory_usage_mb"] <= 16384  # Max 16GB

                if "cpu_usage_percent" in thresholds:
                    assert thresholds["cpu_usage_percent"] > 0
                    assert thresholds["cpu_usage_percent"] <= 100

    def test_resource_limits(self) -> None:
        """Test that resource limits are reasonable."""
        for environment in TargetEnvironment:
            config = self.manager.get_environment_config(environment)

            if config.resource_limits:
                limits = config.resource_limits

                # Check that limits are positive
                for resource, limit in limits.items():
                    assert (
                        limit > 0
                    ), f"Limit for {resource} should be positive"

                # Check specific resource limits
                if "memory_mb" in limits:
                    assert limits["memory_mb"] > 0
                    assert limits["memory_mb"] <= 32768  # Max 32GB

                if "cpu_cores" in limits:
                    assert limits["cpu_cores"] > 0
                    assert limits["cpu_cores"] <= 32  # Max 32 cores

                if "disk_gb" in limits:
                    assert limits["disk_gb"] > 0
                    assert limits["disk_gb"] <= 1000  # Max 1TB


class TestMultiTargetDeploymentIntegration:
    """Integration tests for multi-target deployment system."""

    def test_full_deployment_workflow(self) -> None:
        """Test complete deployment workflow across environments."""
        manager = MultiTargetDeploymentManager()

        # Create deployment config
        config = DeploymentConfig(
            artifact_id="integration-test-model",
            target_environment="development",
        )

        # Mock deployment function
        deployment_results = []

        def mock_deployment_func(
            config: DeploymentConfig, **kwargs
        ) -> DeploymentResult:
            result = DeploymentResult(
                success=True,
                deployment_id=f"integration-{config.target_environment}-{len(deployment_results)}",
                artifact_id=config.artifact_id,
                target_environment=config.target_environment,
                deployment_url=f"http://{config.target_environment}.crackseg.com",
                health_check_url=f"http://{config.target_environment}.crackseg.com/health",
            )
            deployment_results.append(result)
            return result

        # Deploy to multiple environments
        environments = [
            TargetEnvironment.DEVELOPMENT,
            TargetEnvironment.STAGING,
        ]
        results = manager.deploy_to_multiple_environments(
            config=config,
            environments=environments,
            deployment_func=mock_deployment_func,
        )

        # Verify results
        assert len(results) == 2
        assert all(result.success for result in results.values())
        assert len(deployment_results) == 2

        # Verify environment-specific configurations were applied
        for environment, result in results.items():
            assert result.target_environment == environment.value

            # Check that environment-specific strategy was used
            # (This would be verified in the actual deployment orchestration)

    def test_environment_validation_integration(self) -> None:
        """Test integration of environment validation with deployment."""
        manager = MultiTargetDeploymentManager()

        # Validate all environments
        validation_results = {}
        for environment in TargetEnvironment:
            validation = manager.validate_environment_readiness(environment)
            validation_results[environment] = validation

        # Check that all environments have validation results
        assert len(validation_results) == len(TargetEnvironment)

        # Check that validation results have expected structure
        for environment, validation in validation_results.items():
            assert "environment" in validation
            assert "ready" in validation
            assert "issues" in validation
            assert "warnings" in validation
            assert validation["environment"] == environment.value
            assert isinstance(validation["ready"], bool)
            assert isinstance(validation["issues"], list)
            assert isinstance(validation["warnings"], list)
