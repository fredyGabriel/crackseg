"""
Integration tests for CrackSeg traceability system - Bulk operations.

Tests bulk operations, metadata statistics, error handling scenarios,
and multi-user collaboration workflows.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.crackseg.utils.traceability import (
    ArtifactEntity,
    TraceabilityIntegrationManager,
    TraceabilityStorage,
)
from src.crackseg.utils.traceability.enums import ArtifactType, ComplianceLevel


class TestTraceabilityBulkOperations:
    """Integration tests for bulk operations and error handling."""

    @pytest.fixture
    def temp_storage_dir(self) -> Path:
        """Create temporary directory for test storage."""
        temp_dir = tempfile.mkdtemp()
        return Path(temp_dir)

    @pytest.fixture
    def traceability_storage(
        self, temp_storage_dir: Path
    ) -> TraceabilityStorage:
        """Create traceability storage for testing."""
        return TraceabilityStorage(storage_path=temp_storage_dir)

    @pytest.fixture
    def integration_manager(
        self, traceability_storage: TraceabilityStorage
    ) -> TraceabilityIntegrationManager:
        """Create integration manager for testing."""
        return TraceabilityIntegrationManager(traceability_storage)

    def test_metadata_statistics_workflow(
        self,
        traceability_storage: TraceabilityStorage,
        integration_manager: TraceabilityIntegrationManager,
    ) -> None:
        """Test metadata statistics generation with access control."""
        # 1. Create multiple artifacts with different owners
        artifacts_data = [
            {
                "artifact_id": "stats-model-1",
                "owner": "user1",
                "compliance_level": ComplianceLevel.BASIC,
                "metadata": {"accuracy": 0.89, "category": "baseline"},
            },
            {
                "artifact_id": "stats-model-2",
                "owner": "user1",
                "compliance_level": ComplianceLevel.STANDARD,
                "metadata": {"accuracy": 0.92, "category": "improved"},
            },
            {
                "artifact_id": "stats-model-3",
                "owner": "user2",
                "compliance_level": ComplianceLevel.STANDARD,
                "metadata": {"accuracy": 0.94, "category": "advanced"},
            },
            {
                "artifact_id": "stats-model-4",
                "owner": "admin_user",
                "compliance_level": ComplianceLevel.COMPREHENSIVE,
                "metadata": {"accuracy": 0.98, "category": "production"},
            },
        ]

        for i, artifact_data in enumerate(artifacts_data):
            artifact = ArtifactEntity(
                artifact_id=artifact_data["artifact_id"],
                artifact_type=ArtifactType.MODEL,
                file_path=Path(f"/models/stats_{i + 1}.pth"),
                file_size=200000 + i * 50000,
                checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                name=f"Stats Model {i + 1}",
                owner=artifact_data["owner"],
                compliance_level=artifact_data["compliance_level"],
                experiment_id="test-exp",
                metadata=artifact_data["metadata"],
            )
            traceability_storage.save_artifact(artifact)

        # 2. Get metadata statistics for different users
        user1_stats = integration_manager.get_metadata_statistics_with_access(
            "user1"
        )
        assert user1_stats["success"] is True
        assert user1_stats["accessible_statistics"]["user_id"] == "user1"
        assert "metadata_statistics" in user1_stats
        assert "accessible_statistics" in user1_stats

        user2_stats = integration_manager.get_metadata_statistics_with_access(
            "user2"
        )
        assert user2_stats["success"] is True
        assert user2_stats["accessible_statistics"]["user_id"] == "user2"

        admin_stats = integration_manager.get_metadata_statistics_with_access(
            "admin_user"
        )
        assert admin_stats["success"] is True
        assert admin_stats["accessible_statistics"]["user_id"] == "admin_user"

    def test_bulk_operations_workflow(
        self,
        traceability_storage: TraceabilityStorage,
        integration_manager: TraceabilityIntegrationManager,
    ) -> None:
        """Test bulk operations with mixed access scenarios."""
        # 1. Create artifacts with different access permissions
        bulk_artifacts = []
        for i in range(5):
            owner = "user1" if i < 2 else "user2" if i < 4 else "admin_user"
            artifact = ArtifactEntity(
                artifact_id=f"bulk-{i + 1}",
                artifact_type=ArtifactType.MODEL,
                file_path=Path(f"/models/bulk_{i + 1}.pth"),
                file_size=150000 + i * 25000,
                checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                name=f"Bulk Model {i + 1}",
                owner=owner,
                compliance_level=ComplianceLevel.STANDARD,
                experiment_id="test-exp",
                metadata={"accuracy": 0.90 + i * 0.01, "batch": "bulk_test"},
            )
            traceability_storage.save_artifact(artifact)
            bulk_artifacts.append(artifact.artifact_id)

        # 2. Test bulk enrichment with user1 (affects their artifacts only)
        user1_bulk_result = (
            integration_manager.bulk_metadata_operation_with_access(
                "enrich",
                bulk_artifacts,
                "user1",
                {
                    "enriched_by": "user1",
                    "timestamp": datetime.now().isoformat(),
                },
                "artifact",
            )
        )
        assert user1_bulk_result["success"] is True
        assert user1_bulk_result["total_entities"] == 5
        # Should only process user1's artifacts (2 out of 5)
        assert user1_bulk_result["processed_entities"] == 2
        assert user1_bulk_result["successful_operations"] == 2
        assert user1_bulk_result["access_denied"] == 3

        # 3. Test bulk validation with admin (should affect all artifacts)
        admin_bulk_result = (
            integration_manager.bulk_metadata_operation_with_access(
                "validate",
                bulk_artifacts,
                "admin_user",
                {},
                "artifact",
            )
        )
        assert admin_bulk_result["success"] is True
        assert admin_bulk_result["total_entities"] == 5
        assert admin_bulk_result["processed_entities"] == 5
        assert admin_bulk_result["successful_operations"] == 5
        assert admin_bulk_result["access_denied"] == 0

    def test_error_handling_workflow(
        self, integration_manager: TraceabilityIntegrationManager
    ) -> None:
        """Test comprehensive error handling across the traceability system."""
        # Create a test artifact for error handling tests
        test_artifact = ArtifactEntity(
            artifact_id="error-test-artifact",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/error_test.pth"),
            file_size=300000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Error Test Model",
            owner="test_user",
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="test-exp",
            metadata={"accuracy": 0.92},
        )
        # Save artifact to storage using the integration_manager's storage
        integration_manager.storage.save_artifact(test_artifact)

        # 1. Test access to non-existent artifacts
        with pytest.raises(
            RuntimeError,
            match="Access denied: User test_user cannot write artifact "
            "non-existent",
        ):
            integration_manager.enrich_artifact_with_access_control(
                "non-existent", "test_user", {"test": "data"}
            )

        # 2. Test invalid operation types
        invalid_result = (
            integration_manager.bulk_metadata_operation_with_access(
                "invalid_operation",
                ["error-test-artifact"],  # Use the artifact we just created
                "test_user",
                {"test": "data"},
                "artifact",
            )
        )
        assert invalid_result["success"] is True
        assert invalid_result["processed_entities"] == 1
        assert invalid_result["failed_operations"] == 1
        assert "Unknown operation" in str(invalid_result["results"])

        # 3. Test search with invalid parameters
        search_result = integration_manager.search_with_access_control(
            "non_existent_key", "non_existent_value", "test_user", "artifact"
        )
        assert search_result["success"] is True
        assert search_result["total_matches"] == 0
        assert search_result["accessible_matches"] == 0

        # 4. Test compliance validation with non-existent entity
        compliance_result = (
            integration_manager.validate_compliance_with_access(
                "artifact", "non-existent", "test_user"
            )
        )
        # Should handle gracefully without raising exception
        assert isinstance(compliance_result, dict)

    def test_multi_user_collaboration_workflow(
        self,
        traceability_storage: TraceabilityStorage,
        integration_manager: TraceabilityIntegrationManager,
    ) -> None:
        """Test multi-user collaboration scenarios with access control."""
        # 1. Create collaborative experiment artifacts
        shared_artifact = ArtifactEntity(
            artifact_id="shared-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/shared.pth"),
            file_size=400000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Shared Collaborative Model",
            owner="team_lead",
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="collab-exp",
            metadata={
                "accuracy": 0.94,
                "collaborators": ["user1", "user2", "team_lead"],
                "status": "in_review",
            },
        )
        traceability_storage.save_artifact(shared_artifact)

        # 2. Test team lead operations
        team_lead_result = (
            integration_manager.enrich_artifact_with_access_control(
                "shared-model", "team_lead", {"review_status": "approved"}
            )
        )
        assert team_lead_result["success"] is True
        assert team_lead_result["access_granted"] is True

        # 3. Test team member operations (should be denied for write)
        with pytest.raises(RuntimeError, match="Access denied"):
            integration_manager.enrich_artifact_with_access_control(
                "shared-model", "user1", {"review_status": "approved"}
            )

        # 4. Test read access for team members
        user1_search = integration_manager.search_with_access_control(
            "collaborators", "user1", "user1", "artifact"
        )
        assert user1_search["success"] is True

        # 5. Test audit trail for collaboration
        team_audit = integration_manager.audit_trace_with_access_control(
            "team_lead", "artifact"
        )
        assert team_audit["success"] is True
        assert team_audit["entity_type_filter"] == "artifact"
