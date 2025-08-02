"""
Unit tests for TraceabilityIntegrationManager - Core functionality.

This module tests the core integration functionality of the
TraceabilityIntegrationManager class, including initialization and
basic operations.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.crackseg.utils.traceability import (
    ArtifactEntity,
    TraceabilityIntegrationManager,
    TraceabilityStorage,
)
from src.crackseg.utils.traceability.enums import ArtifactType


class TestTraceabilityIntegrationManagerCore:
    """Test core functionality of TraceabilityIntegrationManager."""

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
        return TraceabilityIntegrationManager(mock_storage)

    def test_initialization(self, mock_storage: Mock) -> None:
        """Test proper initialization of integration manager."""
        manager = TraceabilityIntegrationManager(mock_storage)
        assert manager.storage == mock_storage

    def test_enrich_artifact_with_access_control_success(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test successful artifact enrichment with access control."""
        # Mock the storage to return artifact data
        integration_manager.storage._load_artifacts.return_value = [
            {
                "artifact_id": "test-artifact",
                "artifact_type": "model",
                "file_path": "/test/path",
                "file_size": 1000,
                "checksum": "a" * 64,
                "name": "Test Artifact",
                "owner": "test_user",
                "experiment_id": "test-exp",
            }
        ]

        # Mock the access control check
        integration_manager.access_control.check_artifact_access = Mock(
            return_value=True
        )
        integration_manager.metadata_manager.enrich_artifact_metadata = Mock(
            return_value=ArtifactEntity(
                artifact_id="test-artifact",
                artifact_type=ArtifactType.MODEL,
                file_path=Path("/test/path"),
                file_size=1000,
                checksum="a" * 64,
                name="Test Artifact",
                owner="test_user",
                experiment_id="test-exp",
            )
        )

        result = integration_manager.enrich_artifact_with_access_control(
            "test-artifact", "test_user", {"new_metric": 0.95}
        )

        assert result["success"] is True
        assert result["access_granted"] is True
        assert result["artifact_id"] == "test-artifact"

    def test_enrich_artifact_with_access_control_denied(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test artifact enrichment with access denied."""
        # Mock the access control check to deny access
        integration_manager.access_control.check_artifact_access = Mock(
            return_value=False
        )

        with pytest.raises(RuntimeError, match="Access denied"):
            integration_manager.enrich_artifact_with_access_control(
                "test-artifact", "unauthorized_user", {"new_metric": 0.95}
            )

    def test_enrich_artifact_with_access_control_not_found(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test artifact enrichment with non-existent artifact."""
        # Mock the access control check to pass but artifact not found
        integration_manager.access_control.check_artifact_access = Mock(
            return_value=True
        )
        integration_manager.metadata_manager.enrich_artifact_metadata = Mock(
            side_effect=RuntimeError("Artifact test-artifact not found")
        )

        with pytest.raises(
            RuntimeError, match="Artifact test-artifact not found"
        ):
            integration_manager.enrich_artifact_with_access_control(
                "test-artifact", "test_user", {"new_metric": 0.95}
            )
