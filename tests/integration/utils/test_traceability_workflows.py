"""
Integration tests for CrackSeg traceability system - Basic workflows.

Tests real user workflows and component interactions to ensure the traceability
system works correctly as a complete unit. Focuses on end-to-end scenarios
that users would actually perform.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.crackseg.utils.traceability import (
    AccessControl,
    ArtifactEntity,
    ExperimentEntity,
    LineageManager,
    MetadataManager,
    TraceabilityIntegrationManager,
    TraceabilityQueryInterface,
    TraceabilityStorage,
)
from src.crackseg.utils.traceability.enums import (
    ArtifactType,
    ComplianceLevel,
    ExperimentStatus,
)


class TestTraceabilitySystemWorkflows:
    """Integration tests for basic traceability workflows."""

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
            "lineage_manager": LineageManager(traceability_storage),
            "query_interface": TraceabilityQueryInterface(
                traceability_storage
            ),
            "integration_manager": TraceabilityIntegrationManager(
                traceability_storage
            ),
        }

    def test_experiment_to_artifact_workflow(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test complete workflow from experiment to artifact tracking."""
        # Extract components
        storage = complete_traceability_system["storage"]
        metadata_manager = complete_traceability_system["metadata_manager"]
        lineage_manager = complete_traceability_system["lineage_manager"]
        query_interface = complete_traceability_system["query_interface"]

        # 1. Create an experiment
        experiment_data = ExperimentEntity(
            experiment_id="test-exp-001",
            experiment_name="Test Experiment",
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
            metadata={
                "objective": "crack_detection",
                "model_type": "unet",
                "dataset": "crack_dataset_v1",
            },
        )
        storage.save_experiment(experiment_data)

        # 2. Create artifacts linked to the experiment
        model_artifact = ArtifactEntity(
            artifact_id="model-unet-001",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/unet_crack_detection.pth"),
            file_size=1024000,
            checksum="a" * 64,
            name="UNet Crack Detection Model",
            owner="test_user",
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="test-exp-001",
            metadata={
                "accuracy": 0.95,
                "model_type": "unet",
                "input_channels": 3,
                "output_channels": 2,
            },
        )
        storage.save_artifact(model_artifact)

        # 3. Create a source artifact for lineage (simulating input data)
        source_artifact = ArtifactEntity(
            artifact_id="input-data-001",
            artifact_type=ArtifactType.DATASET,
            file_path=Path("/data/crack_dataset.pth"),
            file_size=500000,
            checksum="b" * 64,
            name="Crack Dataset",
            owner="test_user",
            compliance_level=ComplianceLevel.BASIC,
            experiment_id="test-exp-001",
            metadata={"dataset_type": "crack_images", "size": 1000},
        )
        storage.save_artifact(source_artifact)

        # 4. Create lineage relationship (input data -> model)
        lineage_manager.create_lineage(
            source_artifact_id="input-data-001",
            target_artifact_id="model-unet-001",
            relationship_type="derived_from",
            metadata={"timestamp": datetime.now().isoformat()},
        )

        # 5. Enrich artifact with metadata
        metadata_manager.enrich_artifact_metadata(
            model_artifact, {"training_epochs": 100, "loss": 0.05}
        )

        # 6. Query the complete lineage
        lineage_result = query_interface.get_artifact_lineage("model-unet-001")

        # Verify the complete workflow worked
        assert len(lineage_result) > 0
        assert any(
            lineage.source_artifact_id == "input-data-001"
            for lineage in lineage_result
        )
        assert any(
            lineage.target_artifact_id == "model-unet-001"
            for lineage in lineage_result
        )

    def test_artifact_versioning_workflow(
        self, complete_traceability_system: dict[str, Any]
    ) -> None:
        """Test artifact versioning and evolution tracking."""
        storage = complete_traceability_system["storage"]
        lineage_manager = complete_traceability_system["lineage_manager"]

        # 1. Create initial artifact version
        initial_artifact = ArtifactEntity(
            artifact_id="model-v1",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/model_v1.pth"),
            file_size=1000000,
            checksum="b" * 64,
            name="Crack Detection Model v1",
            owner="test_user",
            compliance_level=ComplianceLevel.BASIC,
            experiment_id="test-exp",
            metadata={"accuracy": 0.90, "version": "1.0"},
        )
        storage.save_artifact(initial_artifact)

        # 2. Create improved version
        improved_artifact = ArtifactEntity(
            artifact_id="model-v2",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/models/model_v2.pth"),
            file_size=1100000,
            checksum="c" * 64,
            name="Crack Detection Model v2",
            owner="test_user",
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="test-exp",
            metadata={"accuracy": 0.95, "version": "2.0"},
        )
        storage.save_artifact(improved_artifact)

        # 3. Create version lineage
        lineage_manager.create_lineage(
            source_artifact_id="model-v1",
            target_artifact_id="model-v2",
            relationship_type="evolves_to",
            metadata={
                "improvement_type": "accuracy",
                "timestamp": datetime.now().isoformat(),
            },
        )

        # 8. Verify lineage tree structure
        lineage_tree = lineage_manager.get_lineage_tree("model-v2")
        assert lineage_tree["artifact_id"] == "model-v2"
        assert len(lineage_tree["parents"]) > 0
        assert any(
            parent["artifact_id"] == "model-v1"
            for parent in lineage_tree["parents"]
        )
