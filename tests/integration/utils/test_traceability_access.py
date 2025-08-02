"""
Integration tests for CrackSeg traceability system - Access control and
compliance.

Tests access control integration and compliance validation scenarios
in the traceability system.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.crackseg.utils.traceability import (
    AccessControl,
    ArtifactEntity,
    MetadataManager,
    TraceabilityIntegrationManager,
    TraceabilityStorage,
)
from src.crackseg.utils.traceability.enums import ArtifactType, ComplianceLevel


class TestTraceabilitySystemAccess:
    """Integration tests for access control and compliance."""

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
    def complete_traceability_system(
        self, traceability_storage: TraceabilityStorage
    ) -> dict[str, Any]:
        """Create a complete traceability system with all components."""
        return {
            "storage": traceability_storage,
            "metadata_manager": MetadataManager(traceability_storage),
            "access_control": AccessControl(traceability_storage),
            "integration_manager": TraceabilityIntegrationManager(
                traceability_storage
            ),
        }

    def test_access_control_integration(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test access control integration with metadata operations."""
        storage = complete_traceability_system["storage"]
        integration_manager = complete_traceability_system[
            "integration_manager"
        ]

        # 1. Create artifact with access control
        artifact_data = ArtifactEntity(
            artifact_id="restricted-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/restricted.pth"),
            file_size=500000,
            checksum="d" * 64,
            name="Restricted Model",
            owner="admin_user",
            compliance_level=ComplianceLevel.HIGH,
            experiment_id="test-exp",
            metadata={"sensitivity": "high", "accuracy": 0.98},
        )
        storage.save_artifact(artifact_data)

        # 2. Test access control with metadata enrichment
        # This should work for the owner
        result = integration_manager.enrich_artifact_with_access_control(
            "restricted-model", "admin_user", {"new_metric": 0.99}
        )
        assert result["success"] is True
        assert result["access_granted"] is True

        # 3. Test access denied for non-owner
        with pytest.raises(RuntimeError, match="Access denied"):
            integration_manager.enrich_artifact_with_access_control(
                "restricted-model", "regular_user", {"new_metric": 0.99}
            )

    def test_compliance_validation_integration(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test compliance validation integration with metadata."""
        storage = complete_traceability_system["storage"]
        integration_manager = complete_traceability_system[
            "integration_manager"
        ]

        # 1. Create artifact with compliance requirements
        compliant_artifact = ArtifactEntity(
            artifact_id="compliant-model",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/compliant.pth"),
            file_size=800000,
            checksum="e" * 64,
            name="Compliant Model",
            owner="test_user",
            compliance_level=ComplianceLevel.HIGH,
            experiment_id="test-exp",
            metadata={
                "accuracy": 0.96,
                "documentation": "complete",
                "testing": "comprehensive",
            },
        )
        storage.save_artifact(compliant_artifact)

        # 2. Validate compliance
        compliance_result = (
            integration_manager.validate_compliance_with_access(
                "artifact", "compliant-model", "test_user"
            )
        )
        assert compliance_result["success"] is True
        assert compliance_result["access_granted"] is True
