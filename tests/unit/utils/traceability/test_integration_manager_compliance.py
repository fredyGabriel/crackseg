"""
Unit tests for TraceabilityIntegrationManager - Compliance and audit
functionality.

This module tests the compliance validation and audit trail functionality
of the TraceabilityIntegrationManager class.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.crackseg.utils.traceability import (
    TraceabilityIntegrationManager,
    TraceabilityStorage,
)


class TestTraceabilityIntegrationManagerCompliance:
    """
    Test compliance and audit functionality of TraceabilityIntegrationManager.
    """

    @pytest.fixture
    def mock_storage(self) -> Mock:
        """Create mock storage for testing."""
        storage = Mock(spec=TraceabilityStorage)
        storage._load_artifacts.return_value = []
        storage._load_experiments.return_value = []
        return storage

    @pytest.fixture
    def integration_manager(
        self, mock_storage: Mock
    ) -> TraceabilityIntegrationManager:
        """Create integration manager with mock storage."""
        manager = TraceabilityIntegrationManager(mock_storage)
        # Mock the individual managers
        manager.metadata_manager = Mock()
        manager.access_control = Mock()
        return manager

    def test_validate_compliance_with_access_success(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test compliance validation with access control success."""
        # Mock access control to grant access
        access_control = integration_manager.access_control
        access_control.check_artifact_access.return_value = True

        # Mock compliance enforcement
        access_control.enforce_compliance_policy.return_value = {
            "valid": True,
            "compliance_level": "standard",
            "issues": [],
        }

        # Mock metadata completeness
        metadata_manager = integration_manager.metadata_manager
        metadata_manager.validate_metadata_completeness.return_value = {
            "artifact_completeness": {},
            "experiment_completeness": {},
        }

        result = integration_manager.validate_compliance_with_access(
            "artifact", "test-artifact-1", "test_user"
        )

        assert result["success"] is True
        assert result["access_granted"] is True
        assert result["compliance_result"]["valid"] is True

    def test_validate_compliance_with_access_denied(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test compliance validation with access denied."""
        # Mock access control to deny access
        access_control = integration_manager.access_control
        access_control.check_artifact_access.return_value = False

        result = integration_manager.validate_compliance_with_access(
            "artifact", "test-artifact-1", "test_user"
        )

        assert result["success"] is False
        assert "Access denied" in result["error"]

    def test_audit_trace_with_access_control(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test audit trail with access control."""
        # Mock access log
        integration_manager.access_control.get_access_log.return_value = [
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": "test_user",
                "entity_type": "artifact",
                "entity_id": "test-artifact-1",
                "action": "read",
                "result": "granted",
            },
        ]

        # Mock user permissions
        access_control = integration_manager.access_control
        access_control.get_user_permissions.return_value = {
            "user_id": "test_user",
            "accessible_artifacts": 2,
        }

        result = integration_manager.audit_trace_with_access_control(
            "test_user"
        )

        assert result["success"] is True
        assert result["user_id"] == "test_user"
        assert result["access_log_entries"] == 1
        assert len(result["access_log"]) == 1

    def test_audit_trace_with_entity_type_filter(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test audit trail with entity type filter."""
        # Mock access log with mixed entity types
        integration_manager.access_control.get_access_log.return_value = [
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": "test_user",
                "entity_type": "artifact",
                "entity_id": "test-artifact-1",
                "action": "read",
                "result": "granted",
            },
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": "test_user",
                "entity_type": "experiment",
                "entity_id": "test-exp-1",
                "action": "write",
                "result": "granted",
            },
        ]

        # Mock user permissions
        access_control = integration_manager.access_control
        access_control.get_user_permissions.return_value = {
            "user_id": "test_user",
        }

        result = integration_manager.audit_trace_with_access_control(
            "test_user", entity_type="artifact"
        )

        assert result["success"] is True
        assert result["entity_type_filter"] == "artifact"
        assert result["access_log_entries"] == 1
        assert all(
            entry["entity_type"] == "artifact"
            for entry in result["access_log"]
        )
