"""
Unit tests for TraceabilityIntegrationManager - Bulk operations functionality.

This module tests the bulk metadata operations functionality of the
TraceabilityIntegrationManager class.
"""

from unittest.mock import Mock

import pytest

from src.crackseg.utils.traceability import (
    TraceabilityIntegrationManager,
    TraceabilityStorage,
)


class TestTraceabilityIntegrationManagerBulk:
    """Test bulk operations functionality of TraceabilityIntegrationManager."""

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

    def test_bulk_metadata_operation_with_access_success(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test bulk metadata operation with access control success."""
        # Mock access control to grant access
        access_control = integration_manager.access_control
        access_control.check_artifact_access.return_value = True

        # Mock the enrich_artifact_with_access_control method
        enrich_method = integration_manager.enrich_artifact_with_access_control
        enrich_method = Mock()
        enrich_method.return_value = {
            "success": True,
            "artifact_id": "test-artifact-1",
        }

        result = integration_manager.bulk_metadata_operation_with_access(
            "enrich",
            ["test-artifact-1", "test-artifact-2"],
            "test_user",
            {"new_metric": 0.98},
            "artifact",
        )

        assert result["success"] is True
        assert result["operation"] == "enrich"
        assert result["total_entities"] == 2
        assert result["processed_entities"] == 2
        assert result["successful_operations"] == 2
        assert result["failed_operations"] == 0
        assert result["access_denied"] == 0

        # Verify the method was called twice (for both artifacts)
        assert (
            integration_manager.enrich_artifact_with_access_control.call_count
            == 2
        )
        integration_manager.enrich_artifact_with_access_control.assert_any_call(
            "test-artifact-1", "test_user", {"new_metric": 0.98}
        )
        integration_manager.enrich_artifact_with_access_control.assert_any_call(
            "test-artifact-2", "test_user", {"new_metric": 0.98}
        )

    def test_bulk_metadata_operation_with_access_denied(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test bulk metadata operation with access denied."""
        # Mock access control to deny access
        access_control = integration_manager.access_control
        access_control.check_artifact_access.return_value = False

        result = integration_manager.bulk_metadata_operation_with_access(
            "enrich",
            ["test-artifact-1", "test-artifact-2"],
            "test_user",
            {"new_metric": 0.98},
            "artifact",
        )

        assert result["success"] is True
        assert result["total_entities"] == 2
        assert result["processed_entities"] == 0
        assert result["successful_operations"] == 0
        assert result["failed_operations"] == 0
        assert result["access_denied"] == 2

    def test_bulk_metadata_operation_with_mixed_results(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test bulk metadata operation with mixed access results."""
        # Mock access control to grant access for first, deny for second
        access_control = integration_manager.access_control
        access_control.check_artifact_access.side_effect = [
            True,  # First artifact accessible
            False,  # Second artifact not accessible
        ]

        # Mock the enrich_artifact_with_access_control method
        enrich_method = integration_manager.enrich_artifact_with_access_control
        enrich_method = Mock()
        enrich_method.return_value = {
            "success": True,
            "artifact_id": "test-artifact-1",
        }

        result = integration_manager.bulk_metadata_operation_with_access(
            "enrich",
            ["test-artifact-1", "test-artifact-2"],
            "test_user",
            {"new_metric": 0.98},
            "artifact",
        )

        assert result["success"] is True
        assert result["total_entities"] == 2
        assert result["processed_entities"] == 1  # Only first entity processed
        assert result["successful_operations"] == 1
        assert result["failed_operations"] == 0
        assert result["access_denied"] == 1  # Second entity denied access

        # Verify the method was called only once (for the first artifact)
        integration_manager.enrich_artifact_with_access_control.assert_called_once_with(
            "test-artifact-1", "test_user", {"new_metric": 0.98}
        )

    def test_bulk_metadata_operation_unknown_operation(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test bulk metadata operation with unknown operation."""
        # Mock access control to grant access
        access_control = integration_manager.access_control
        access_control.check_artifact_access.return_value = True

        result = integration_manager.bulk_metadata_operation_with_access(
            "unknown_operation",
            ["test-artifact-1"],
            "test_user",
            {"new_metric": 0.98},
            "artifact",
        )

        assert result["success"] is True
        assert result["processed_entities"] == 1
        assert result["successful_operations"] == 0
        assert result["failed_operations"] == 1
        assert "Unknown operation" in result["results"][0]["result"]["error"]
