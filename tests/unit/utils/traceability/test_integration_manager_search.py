"""
Unit tests for TraceabilityIntegrationManager - Search functionality.

This module tests the search and query functionality of the
TraceabilityIntegrationManager class.
"""

from unittest.mock import Mock

import pytest

from src.crackseg.utils.traceability import (
    TraceabilityIntegrationManager,
    TraceabilityStorage,
)


class TestTraceabilityIntegrationManagerSearch:
    """Test search functionality of TraceabilityIntegrationManager."""

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

    def test_search_with_access_control_artifacts(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test search with access control for artifacts."""
        # Mock metadata manager search
        metadata_manager = integration_manager.metadata_manager
        metadata_manager.search_by_metadata.return_value = [
            {"artifact_id": "test-artifact-1", "metadata": {"accuracy": 0.95}},
            {"artifact_id": "test-artifact-2", "metadata": {"accuracy": 0.92}},
        ]

        # Mock access control checks
        access_control = integration_manager.access_control
        access_control.check_artifact_access.side_effect = [
            True,  # First artifact accessible
            False,  # Second artifact not accessible
        ]

        result = integration_manager.search_with_access_control(
            "accuracy", 0.95, "test_user", "artifact"
        )

        assert result["success"] is True
        assert result["total_matches"] == 2
        assert result["accessible_matches"] == 1
        assert len(result["accessible_entities"]) == 1

    def test_search_with_access_control_experiments(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test search with access control for experiments."""
        # Mock metadata manager search
        metadata_manager = integration_manager.metadata_manager
        metadata_manager.search_by_metadata.return_value = [
            {
                "experiment_id": "test-exp-1",
                "metadata": {"objective": "crack_detection"},
            },
        ]

        # Mock access control checks
        access_control = integration_manager.access_control
        access_control.check_experiment_access.return_value = True

        result = integration_manager.search_with_access_control(
            "objective", "crack_detection", "test_user", "experiment"
        )

        assert result["success"] is True
        assert result["total_matches"] == 1
        assert result["accessible_matches"] == 1
        assert len(result["accessible_entities"]) == 1

    def test_get_metadata_statistics_with_access(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test getting metadata statistics with access control."""
        # Mock user permissions
        access_control = integration_manager.access_control
        access_control.get_user_permissions.return_value = {
            "user_id": "test_user",
            "accessible_artifacts": 2,
            "accessible_experiments": 1,
            "owned_artifacts": 1,
            "owned_experiments": 1,
            "can_create_artifacts": True,
            "can_create_experiments": True,
            "can_access_public_data": True,
        }

        # Mock metadata statistics
        metadata_manager = integration_manager.metadata_manager
        metadata_manager.get_metadata_statistics.return_value = {
            "total_artifacts": 2,
            "total_experiments": 1,
            "artifact_metadata_keys": ["accuracy", "model_type"],
        }

        result = integration_manager.get_metadata_statistics_with_access(
            "test_user"
        )

        assert result["success"] is True
        assert result["user_permissions"]["user_id"] == "test_user"
        assert (
            result["accessible_statistics"]["total_accessible_artifacts"] == 2
        )
        assert (
            result["accessible_statistics"]["total_accessible_experiments"]
            == 1
        )
