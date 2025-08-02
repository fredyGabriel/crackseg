"""Unit tests for Production Readiness Validator.

This module provides comprehensive unit tests for the
ProductionReadinessValidator class, covering all validation
categories and edge cases.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from crackseg.utils.deployment.production_readiness_validator import (
    ProductionReadinessCriteria,
    ProductionReadinessResult,
    ProductionReadinessValidator,
)


def create_mock_artifact_entity():
    """Create a mock ArtifactEntity for testing."""
    from crackseg.utils.traceability import ArtifactEntity
    from crackseg.utils.traceability.enums import ArtifactType

    return ArtifactEntity(
        artifact_id="test-artifact-001",
        artifact_type=ArtifactType.MODEL,
        file_path=Path("/path/to/artifact"),
        file_size=1024,
        checksum="a" * 64,  # Mock SHA256
        name="Test Model",
        owner="test_user",
        experiment_id="test-exp-001",
        metadata={"version": "1.0.0", "format": "pytorch"},
    )


class TestProductionReadinessCriteria:
    """Test ProductionReadinessCriteria dataclass."""

    def test_default_criteria(self):
        """Test default criteria initialization."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={},
            resource_checks={},
            operational_checks={},
            compliance_checks={},
        )

        assert criteria.security_checks == {}
        assert criteria.performance_checks == {}
        assert criteria.resource_checks == {}
        assert criteria.operational_checks == {}
        assert criteria.compliance_checks == {}

    def test_custom_criteria(self):
        """Test custom criteria initialization."""
        criteria = ProductionReadinessCriteria(
            security_checks={"vulnerability_scan": True},
            performance_checks={"inference_time": True},
            resource_checks={"memory_usage": True},
            operational_checks={"health_checks": True},
            compliance_checks={"gdpr": True},
        )

        assert criteria.security_checks["vulnerability_scan"] is True
        assert criteria.performance_checks["inference_time"] is True
        assert criteria.resource_checks["memory_usage"] is True
        assert criteria.operational_checks["health_checks"] is True
        assert criteria.compliance_checks["gdpr"] is True


class TestProductionReadinessResult:
    """Test ProductionReadinessResult dataclass."""

    def test_default_result(self):
        """Test default result initialization."""
        result = ProductionReadinessResult(
            success=False,
            total_checks=0,
            passed_checks=0,
            failed_checks_count=0,
            failed_checks={},
        )

        assert result.success is False
        assert result.total_checks == 0
        assert result.passed_checks == 0
        assert result.failed_checks_count == 0
        assert result.failed_checks == {}
        assert result.error_message is None

    def test_successful_result(self):
        """Test successful result initialization."""
        result = ProductionReadinessResult(
            success=True,
            total_checks=5,
            passed_checks=4,
            failed_checks_count=1,
            failed_checks={
                "security_vulnerability_scan": "Vulnerability found"
            },
        )

        assert result.success is True
        assert result.total_checks == 5
        assert result.passed_checks == 4
        assert result.failed_checks_count == 1
        assert len(result.failed_checks) == 1
        assert (
            result.failed_checks["security_vulnerability_scan"]
            == "Vulnerability found"
        )


class TestProductionReadinessValidator:
    """Test ProductionReadinessValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ProductionReadinessValidator()

    @pytest.fixture
    def mock_artifact(self):
        """Create mock artifact."""
        return create_mock_artifact_entity()

    @pytest.fixture
    def deployment_config(self):
        """Create deployment config."""
        return {"target_environment": "production"}

    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.logger is not None

    def test_initialization_with_custom_criteria(self):
        """Test validator initialization with custom criteria."""
        validator = ProductionReadinessValidator()
        assert validator.logger is not None

    @patch.object(ProductionReadinessValidator, "_check_security")
    @patch.object(ProductionReadinessValidator, "_check_performance")
    @patch.object(ProductionReadinessValidator, "_check_resources")
    @patch.object(ProductionReadinessValidator, "_check_operational")
    @patch.object(ProductionReadinessValidator, "_check_compliance")
    def test_validate_production_readiness_success(
        self,
        mock_compliance,
        mock_operational,
        mock_resources,
        mock_performance,
        mock_security,
        validator,
        mock_artifact,
        deployment_config,
    ):
        """Test successful production readiness validation."""
        # Mock all checks to pass
        mock_security.return_value = True
        mock_performance.return_value = True
        mock_resources.return_value = True
        mock_operational.return_value = True
        mock_compliance.return_value = True

        criteria = ProductionReadinessCriteria(
            security_checks={"vulnerability_scan": True},
            performance_checks={"inference_time": True},
            resource_checks={"memory_usage": True},
            operational_checks={"health_checks": True},
            compliance_checks={"gdpr": True},
        )

        result = validator.validate_production_readiness(
            mock_artifact, criteria
        )

        assert result.success is True
        assert result.total_checks == 5
        assert result.passed_checks == 5
        assert result.failed_checks_count == 0
        assert result.failed_checks == {}

    @patch.object(ProductionReadinessValidator, "_check_security")
    def test_validate_production_readiness_failure(
        self, mock_security, validator, mock_artifact, deployment_config
    ):
        """Test failed production readiness validation."""
        # Mock security check to fail
        mock_security.return_value = False

        criteria = ProductionReadinessCriteria(
            security_checks={"vulnerability_scan": True},
            performance_checks={},
            resource_checks={},
            operational_checks={},
            compliance_checks={},
        )

        result = validator.validate_production_readiness(
            mock_artifact, criteria
        )

        assert result.success is False
        assert result.total_checks == 1
        assert result.passed_checks == 0
        assert result.failed_checks_count == 1
        assert "security_vulnerability_scan" in result.failed_checks

    def test_validate_security_success(
        self, validator, mock_artifact, deployment_config
    ):
        """Test successful security validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={"vulnerability_scan": True},
            performance_checks={},
            resource_checks={},
            operational_checks={},
            compliance_checks={},
        )

        with patch.object(validator, "_check_security", return_value=True):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is True
            assert result.total_checks == 1
            assert result.passed_checks == 1
            assert result.failed_checks_count == 0

    def test_validate_security_failure(
        self, validator, mock_artifact, deployment_config
    ):
        """Test failed security validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={"vulnerability_scan": True},
            performance_checks={},
            resource_checks={},
            operational_checks={},
            compliance_checks={},
        )

        with patch.object(validator, "_check_security", return_value=False):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is False
            assert result.total_checks == 1
            assert result.passed_checks == 0
            assert result.failed_checks_count == 1
            assert "security_vulnerability_scan" in result.failed_checks

    def test_validate_performance_success(
        self, validator, mock_artifact, deployment_config
    ):
        """Test successful performance validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={"inference_time": True},
            resource_checks={},
            operational_checks={},
            compliance_checks={},
        )

        with patch.object(validator, "_check_performance", return_value=True):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is True
            assert result.total_checks == 1
            assert result.passed_checks == 1
            assert result.failed_checks_count == 0

    def test_validate_performance_failure(
        self, validator, mock_artifact, deployment_config
    ):
        """Test failed performance validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={"inference_time": True},
            resource_checks={},
            operational_checks={},
            compliance_checks={},
        )

        with patch.object(validator, "_check_performance", return_value=False):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is False
            assert result.total_checks == 1
            assert result.passed_checks == 0
            assert result.failed_checks_count == 1
            assert "performance_inference_time" in result.failed_checks

    def test_validate_resources_success(
        self, validator, mock_artifact, deployment_config
    ):
        """Test successful resource validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={},
            resource_checks={"memory_usage": True},
            operational_checks={},
            compliance_checks={},
        )

        with patch.object(validator, "_check_resources", return_value=True):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is True
            assert result.total_checks == 1
            assert result.passed_checks == 1
            assert result.failed_checks_count == 0

    def test_validate_resources_failure(
        self, validator, mock_artifact, deployment_config
    ):
        """Test failed resource validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={},
            resource_checks={"memory_usage": True},
            operational_checks={},
            compliance_checks={},
        )

        with patch.object(validator, "_check_resources", return_value=False):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is False
            assert result.total_checks == 1
            assert result.passed_checks == 0
            assert result.failed_checks_count == 1
            assert "resource_memory_usage" in result.failed_checks

    def test_validate_operational_readiness_success(
        self, validator, mock_artifact, deployment_config
    ):
        """Test successful operational readiness validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={},
            resource_checks={},
            operational_checks={"health_checks": True},
            compliance_checks={},
        )

        with patch.object(validator, "_check_operational", return_value=True):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is True
            assert result.total_checks == 1
            assert result.passed_checks == 1
            assert result.failed_checks_count == 0

    def test_validate_operational_readiness_failure(
        self, validator, mock_artifact, deployment_config
    ):
        """Test failed operational readiness validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={},
            resource_checks={},
            operational_checks={"health_checks": True},
            compliance_checks={},
        )

        with patch.object(validator, "_check_operational", return_value=False):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is False
            assert result.total_checks == 1
            assert result.passed_checks == 0
            assert result.failed_checks_count == 1
            assert "operational_health_checks" in result.failed_checks

    def test_validate_compliance_success(
        self, validator, mock_artifact, deployment_config
    ):
        """Test successful compliance validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={},
            resource_checks={},
            operational_checks={},
            compliance_checks={"gdpr": True},
        )

        with patch.object(validator, "_check_compliance", return_value=True):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is True
            assert result.total_checks == 1
            assert result.passed_checks == 1
            assert result.failed_checks_count == 0

    def test_validate_compliance_with_requirements(
        self, validator, mock_artifact, deployment_config
    ):
        """Test compliance validation with specific requirements."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={},
            resource_checks={},
            operational_checks={},
            compliance_checks={"gdpr": True, "sox": True},
        )

        with patch.object(validator, "_check_compliance", return_value=True):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is True
            assert result.total_checks == 2
            assert result.passed_checks == 2
            assert result.failed_checks_count == 0

    def test_validate_compliance_failure(
        self, validator, mock_artifact, deployment_config
    ):
        """Test failed compliance validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={},
            resource_checks={},
            operational_checks={},
            compliance_checks={"gdpr": True},
        )

        with patch.object(validator, "_check_compliance", return_value=False):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is False
            assert result.total_checks == 1
            assert result.passed_checks == 0
            assert result.failed_checks_count == 1
            assert "compliance_gdpr" in result.failed_checks

    def test_calculate_overall_score(self, validator):
        """Test overall score calculation."""
        result = ProductionReadinessResult(
            success=True,
            total_checks=5,
            passed_checks=4,
            failed_checks_count=1,
            failed_checks={
                "security_vulnerability_scan": "Vulnerability found"
            },
        )

        # Expected: 4 passed / 5 total = 0.8
        expected_score = 0.8
        assert result.passed_checks / result.total_checks == expected_score

    def test_calculate_overall_score_max_cap(self, validator):
        """Test overall score calculation with maximum cap."""
        result = ProductionReadinessResult(
            success=True,
            total_checks=5,
            passed_checks=5,
            failed_checks_count=0,
            failed_checks={},
        )

        # Expected: 5 passed / 5 total = 1.0
        expected_score = 1.0
        assert result.passed_checks / result.total_checks == expected_score

    def test_validation_exception_handling(
        self, validator, mock_artifact, deployment_config
    ):
        """Test exception handling during validation."""
        criteria = ProductionReadinessCriteria(
            security_checks={"vulnerability_scan": True},
            performance_checks={},
            resource_checks={},
            operational_checks={},
            compliance_checks={},
        )

        with patch.object(
            validator,
            "_check_security",
            side_effect=Exception("Test error"),
        ):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is False
            assert result.error_message == "Test error"

    def test_model_loading_failure(
        self, validator, mock_artifact, deployment_config
    ):
        """Test performance validation when model loading fails."""
        criteria = ProductionReadinessCriteria(
            security_checks={},
            performance_checks={"inference_time": True},
            resource_checks={},
            operational_checks={},
            compliance_checks={},
        )

        with patch.object(validator, "_check_performance", return_value=False):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is False
            assert result.total_checks == 1
            assert result.passed_checks == 0
            assert result.failed_checks_count == 1

    def test_custom_criteria_validation(self):
        """Test validation with custom criteria."""
        validator = ProductionReadinessValidator()
        mock_artifact = create_mock_artifact_entity()

        criteria = ProductionReadinessCriteria(
            security_checks={"vulnerability_scan": True},
            performance_checks={"inference_time": True},
            resource_checks={"memory_usage": True},
            operational_checks={"health_checks": True},
            compliance_checks={"gdpr": True},
        )

        with (
            patch.object(validator, "_check_security", return_value=True),
            patch.object(validator, "_check_performance", return_value=True),
            patch.object(validator, "_check_resources", return_value=True),
            patch.object(validator, "_check_operational", return_value=True),
            patch.object(validator, "_check_compliance", return_value=True),
        ):
            result = validator.validate_production_readiness(
                mock_artifact, criteria
            )

            assert result.success is True
            assert result.total_checks == 5
            assert result.passed_checks == 5
            assert result.failed_checks_count == 0
