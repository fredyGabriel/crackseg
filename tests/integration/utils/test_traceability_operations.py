"""
Integration tests for CrackSeg traceability system - Bulk operations and
performance.

Tests bulk operations, performance scenarios, and data persistence
in the traceability system.
"""

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.crackseg.utils.traceability import (
    ArtifactEntity,
    ExperimentEntity,
    MetadataManager,
    TraceabilityIntegrationManager,
    TraceabilityQuery,
    TraceabilityQueryInterface,
    TraceabilityStorage,
)
from src.crackseg.utils.traceability.enums import (
    ArtifactType,
    ComplianceLevel,
    ExperimentStatus,
)


class TestTraceabilitySystemOperations:
    """Integration tests for bulk operations and performance."""

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
            "query_interface": TraceabilityQueryInterface(
                traceability_storage
            ),
            "integration_manager": TraceabilityIntegrationManager(
                traceability_storage
            ),
        }

    def test_bulk_operations_integration(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test bulk operations with multiple artifacts."""
        storage = complete_traceability_system["storage"]
        integration_manager = complete_traceability_system[
            "integration_manager"
        ]

        # 1. Create multiple artifacts
        artifacts = []
        for i in range(3):
            artifact = ArtifactEntity(
                artifact_id=f"bulk-artifact-{i}",
                artifact_type=ArtifactType.MODEL,
                file_path=Path(f"/models/bulk_{i}.pth"),
                file_size=500000 + i * 100000,
                checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                name=f"Bulk Model {i}",
                owner="test_user",
                compliance_level=ComplianceLevel.STANDARD,
                experiment_id="test-exp",
                metadata={"accuracy": 0.90 + i * 0.02},
            )
            storage.save_artifact(artifact)
            artifacts.append(artifact.artifact_id)

        # 2. Perform bulk metadata enrichment
        bulk_result = integration_manager.bulk_metadata_operation_with_access(
            "enrich",
            artifacts,
            "test_user",
            {"bulk_operation": True, "timestamp": datetime.now().isoformat()},
            "artifact",
        )

        assert bulk_result["success"] is True
        assert bulk_result["total_entities"] == 3
        assert bulk_result["processed_entities"] == 3
        assert bulk_result["successful_operations"] == 3

    def test_query_interface_integration(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test query interface with real data."""
        storage = complete_traceability_system["storage"]
        query_interface = complete_traceability_system["query_interface"]

        # 1. Create test data
        experiment_data = ExperimentEntity(
            experiment_id="query-test-exp",
            experiment_name="Query Test Experiment",
            username="test_user",
            status=ExperimentStatus.COMPLETED,
            config_hash="test_hash",
            python_version="3.12",
            pytorch_version="2.0",
            platform="win32",
            hostname="test-host",
            memory_gb=8.0,
            cuda_version=None,
            git_commit=None,
            git_branch=None,
            started_at=None,
            completed_at=None,
            metadata={"objective": "testing", "model_type": "test"},
        )
        storage.save_experiment(experiment_data)

        artifact_data = ArtifactEntity(
            artifact_id="query-test-artifact",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/query_test.pth"),
            file_size=300000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Query Test Model",
            owner="test_user",
            compliance_level=ComplianceLevel.BASIC,
            experiment_id="query-test-exp",
            metadata={"accuracy": 0.92, "test_data": True},
        )
        storage.save_artifact(artifact_data)

        # 2. Test metadata search

        # Create a query to search for artifacts with specific metadata
        query = TraceabilityQuery(
            artifact_types=[ArtifactType.MODEL],
            limit=10,
            offset=0,
            created_after=None,
            created_before=None,
        )

        search_result = query_interface.search_artifacts(query)

        assert search_result.total_count >= 1
        assert len(search_result.artifacts) >= 1
        assert any(
            artifact.artifact_id == "query-test-artifact"
            for artifact in search_result.artifacts
        )

    def test_error_handling_integration(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test error handling across components."""
        storage = complete_traceability_system["storage"]
        integration_manager = complete_traceability_system[
            "integration_manager"
        ]

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
        storage.save_artifact(test_artifact)

        # Test with non-existent artifact
        with pytest.raises(
            RuntimeError,
            match="Access denied: User test_user cannot write artifact "
            "non-existent",
        ):
            integration_manager.enrich_artifact_with_access_control(
                "non-existent", "test_user", {"test": "data"}
            )

        # Test with invalid operation
        result = integration_manager.bulk_metadata_operation_with_access(
            "invalid_operation",
            ["error-test-artifact"],  # Use the artifact we just created
            "test_user",
            {"test": "data"},
            "artifact",
        )
        assert result["success"] is True
        assert result["processed_entities"] == 1
        assert result["failed_operations"] == 1
        assert "Unknown operation" in str(result["results"])

        # Test with non-existent experiment -
        # using validate_compliance_with_access instead
        compliance_result = (
            integration_manager.validate_compliance_with_access(
                "experiment", "non-existent", "test_user"
            )
        )
        assert compliance_result["success"] is False
        assert "Access denied" in compliance_result["error"]

    def test_performance_integration(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test performance with larger datasets."""
        storage = complete_traceability_system["storage"]
        query_interface = complete_traceability_system["query_interface"]

        # Create multiple artifacts for performance testing
        artifacts = []
        for i in range(10):
            artifact = ArtifactEntity(
                artifact_id=f"perf-artifact-{i}",
                artifact_type=ArtifactType.MODEL,
                file_path=Path(f"/models/perf_{i}.pth"),
                file_size=400000,
                checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                name=f"Performance Model {i}",
                owner="test_user",
                compliance_level=ComplianceLevel.STANDARD,
                experiment_id="test-exp",
                metadata={
                    "accuracy": 0.90 + i * 0.01,
                    "performance_test": True,
                },
            )
            storage.save_artifact(artifact)
            artifacts.append(artifact.artifact_id)

        # Test query performance
        start_time = time.time()

        query = TraceabilityQuery(
            artifact_types=[ArtifactType.MODEL],
            limit=20,
            offset=0,
            created_after=None,
            created_before=None,
        )

        search_result = query_interface.search_artifacts(query)

        end_time = time.time()
        query_time = end_time - start_time

        assert search_result.total_count >= 10
        assert len(search_result.artifacts) >= 10
        assert query_time < 1.0  # Should complete within 1 second

    def test_data_persistence_integration(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test that data persists correctly across operations."""
        storage = complete_traceability_system["storage"]
        metadata_manager = complete_traceability_system["metadata_manager"]

        # 1. Create and save artifact
        artifact_data = ArtifactEntity(
            artifact_id="persistence-test",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/persistence.pth"),
            file_size=600000,
            checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            name="Persistence Test Model",
            owner="test_user",
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="test-exp",
            metadata={"accuracy": 0.94},
        )
        storage.save_artifact(artifact_data)

        # 2. Enrich with additional metadata
        metadata_manager.enrich_artifact_metadata(
            artifact_data, {"additional_metric": 0.96}
        )

        # 3. Verify data persistence by reloading
        artifacts = storage._load_artifacts()
        persisted_artifact = next(
            (a for a in artifacts if a["artifact_id"] == "persistence-test"),
            None,
        )

        assert persisted_artifact is not None
        assert persisted_artifact["metadata"]["accuracy"] == 0.94
        assert persisted_artifact["metadata"]["additional_metric"] == 0.96
