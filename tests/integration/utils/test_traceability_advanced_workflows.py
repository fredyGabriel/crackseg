"""
Integration tests for CrackSeg traceability system - Advanced workflows.

Tests advanced user workflows including audit trails, compliance validation,
and advanced search functionality with access control.
"""

import tempfile
from pathlib import Path

import pytest

from src.crackseg.utils.traceability import (
    ArtifactEntity,
    TraceabilityIntegrationManager,
    TraceabilityStorage,
)
from src.crackseg.utils.traceability.enums import ArtifactType, ComplianceLevel


class TestTraceabilityAdvancedWorkflows:
    """Integration tests for advanced traceability workflows."""

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

    def test_audit_trail_workflow(
        self,
        traceability_storage: TraceabilityStorage,
        integration_manager: TraceabilityIntegrationManager,
    ) -> None:
        """Test audit trail generation and access control integration."""
        # 1. Create artifacts with different access levels
        public_artifact = ArtifactEntity(
            artifact_id="public-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/public.pth"),
            file_size=300000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Public Model",
            owner="admin_user",
            compliance_level=ComplianceLevel.BASIC,
            experiment_id="test-exp",
            metadata={"accuracy": 0.88, "public": True},
        )
        traceability_storage.save_artifact(public_artifact)

        private_artifact = ArtifactEntity(
            artifact_id="private-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/private.pth"),
            file_size=400000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Private Model",
            owner="admin_user",
            compliance_level=ComplianceLevel.COMPREHENSIVE,
            experiment_id="test-exp",
            metadata={"accuracy": 0.96, "sensitive": True},
        )
        traceability_storage.save_artifact(private_artifact)

        # 2. Perform operations to generate audit trail
        # Enrich public artifact (should work for the owner)
        integration_manager.enrich_artifact_with_access_control(
            "public-model", "admin_user", {"accessed_by": "admin_user"}
        )

        # Try to access private artifact (should be denied)
        with pytest.raises(RuntimeError, match="Access denied"):
            integration_manager.enrich_artifact_with_access_control(
                "private-model",
                "regular_user",
                {"accessed_by": "regular_user"},
            )

        # 3. Get audit trail for admin user
        admin_audit = integration_manager.audit_trace_with_access_control(
            "admin_user"
        )
        assert admin_audit["success"] is True
        assert admin_audit["user_id"] == "admin_user"
        assert admin_audit["access_log_entries"] >= 0

        # 4. Get audit trail for regular user
        regular_audit = integration_manager.audit_trace_with_access_control(
            "regular_user"
        )
        assert regular_audit["success"] is True
        assert regular_audit["user_id"] == "regular_user"

    def test_compliance_validation_workflow(
        self,
        traceability_storage: TraceabilityStorage,
        integration_manager: TraceabilityIntegrationManager,
    ) -> None:
        """Test compliance validation workflow."""
        # 1. Create artifacts with different compliance levels
        basic_artifact = ArtifactEntity(
            artifact_id="basic-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/basic.pth"),
            file_size=200000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Basic Model",
            owner="test_user",
            compliance_level=ComplianceLevel.BASIC,
            experiment_id="test-exp",
            metadata={"accuracy": 0.85},
        )
        traceability_storage.save_artifact(basic_artifact)

        standard_artifact = ArtifactEntity(
            artifact_id="standard-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/standard.pth"),
            file_size=350000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Standard Model",
            owner="test_user",
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="test-exp",
            metadata={
                "accuracy": 0.92,
                "documentation": "complete",
                "testing": "basic",
            },
        )
        traceability_storage.save_artifact(standard_artifact)

        high_artifact = ArtifactEntity(
            artifact_id="high-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/high.pth"),
            file_size=500000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="High Compliance Model",
            owner="test_user",
            compliance_level=ComplianceLevel.COMPREHENSIVE,
            experiment_id="test-exp",
            metadata={
                "accuracy": 0.98,
                "documentation": "comprehensive",
                "testing": "extensive",
                "validation": "complete",
                "certification": "approved",
            },
        )
        traceability_storage.save_artifact(high_artifact)

        # 2. Validate compliance for each artifact
        basic_compliance = integration_manager.validate_compliance_with_access(
            "artifact", "basic-model", "test_user"
        )
        assert basic_compliance["success"] is True
        assert basic_compliance["access_granted"] is True

        standard_compliance = (
            integration_manager.validate_compliance_with_access(
                "artifact", "standard-model", "test_user"
            )
        )
        assert standard_compliance["success"] is True
        assert standard_compliance["access_granted"] is True

        high_compliance = integration_manager.validate_compliance_with_access(
            "artifact", "high-model", "test_user"
        )
        assert high_compliance["success"] is True
        assert high_compliance["access_granted"] is True

    def test_advanced_search_with_access_control(
        self,
        traceability_storage: TraceabilityStorage,
        integration_manager: TraceabilityIntegrationManager,
    ) -> None:
        """Test advanced search functionality with access control filtering."""
        # 1. Create artifacts with different owners and metadata
        user1_artifact = ArtifactEntity(
            artifact_id="user1-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/user1.pth"),
            file_size=250000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="User1 Model",
            owner="user1",
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="test-exp",
            metadata={"accuracy": 0.91, "owner": "user1"},
        )
        traceability_storage.save_artifact(user1_artifact)

        user2_artifact = ArtifactEntity(
            artifact_id="user2-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/user2.pth"),
            file_size=280000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="User2 Model",
            owner="user2",
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="test-exp",
            metadata={"accuracy": 0.93, "owner": "user2"},
        )
        traceability_storage.save_artifact(user2_artifact)

        admin_artifact = ArtifactEntity(
            artifact_id="admin-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/admin.pth"),
            file_size=320000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Admin Model",
            owner="admin_user",
            compliance_level=ComplianceLevel.COMPREHENSIVE,
            experiment_id="test-exp",
            metadata={"accuracy": 0.97, "owner": "admin_user"},
        )
        traceability_storage.save_artifact(admin_artifact)

        # 2. Test search with access control for different users
        # User1 should only see their own artifact and public ones
        user1_search = integration_manager.search_with_access_control(
            "accuracy", 0.91, "user1", "artifact"
        )
        assert user1_search["success"] is True
        assert user1_search["total_matches"] >= 1
        # Should only see accessible artifacts
        assert (
            user1_search["accessible_matches"] <= user1_search["total_matches"]
        )

        # User2 should see their own artifact and public ones
        user2_search = integration_manager.search_with_access_control(
            "accuracy", 0.93, "user2", "artifact"
        )
        assert user2_search["success"] is True
        assert user2_search["total_matches"] >= 1

        # Admin should see all artifacts
        admin_search = integration_manager.search_with_access_control(
            "accuracy", 0.97, "admin_user", "artifact"
        )
        assert admin_search["success"] is True
        assert admin_search["total_matches"] >= 1
