"""
Tests for lineage management module.

This module tests the advanced lineage management functionality including
lineage creation, validation, analysis, and integrity checking.
"""

from datetime import datetime
from pathlib import Path

import pytest

from crackseg.utils.traceability.lineage_manager import LineageManager
from crackseg.utils.traceability.models import (
    ArtifactEntity,
    ArtifactType,
    ComplianceLevel,
    VerificationStatus,
)
from crackseg.utils.traceability.storage import TraceabilityStorage


class TestLineageManager:
    """Test lineage management functionality."""

    @pytest.fixture
    def temp_storage(self, tmp_path: Path) -> Path:
        """Create temporary storage directory."""
        storage_path = tmp_path / "lineage_storage"
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    @pytest.fixture
    def storage(self, temp_storage: Path) -> TraceabilityStorage:
        """Create storage instance with temporary directory."""
        return TraceabilityStorage(temp_storage)

    @pytest.fixture
    def lineage_manager(self, storage: TraceabilityStorage) -> LineageManager:
        """Create lineage manager instance."""
        return LineageManager(storage)

    @pytest.fixture
    def sample_artifact_1(self) -> ArtifactEntity:
        """Create first sample artifact."""
        return ArtifactEntity(
            artifact_id="artifact-001",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/path/to/model1.pth"),
            file_size=1024,
            checksum=(
                "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
            ),
            name="Model 1",
            description="First test model",
            owner="user1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            verification_status=VerificationStatus.VERIFIED,
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="exp-001",
            version="1.0.0",
            tags=["test", "model"],
            metadata={"accuracy": 0.95},
        )

    @pytest.fixture
    def sample_artifact_2(self) -> ArtifactEntity:
        """Create second sample artifact."""
        return ArtifactEntity(
            artifact_id="artifact-002",
            artifact_type=ArtifactType.MODEL,
            file_path=Path("/path/to/model2.pth"),
            file_size=2048,
            checksum=(
                "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
            ),
            name="Model 2",
            description="Second test model",
            owner="user1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            verification_status=VerificationStatus.VERIFIED,
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="exp-001",
            version="1.0.0",
            tags=["test", "model"],
            metadata={"accuracy": 0.97},
        )

    @pytest.fixture
    def sample_artifact_3(self) -> ArtifactEntity:
        """Create third sample artifact."""
        return ArtifactEntity(
            artifact_id="artifact-003",
            artifact_type=ArtifactType.DATASET,
            file_path=Path("/path/to/dataset.pth"),
            file_size=512,
            checksum=(
                "3c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
            ),
            name="Dataset",
            description="Test dataset",
            owner="user1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            verification_status=VerificationStatus.VERIFIED,
            compliance_level=ComplianceLevel.STANDARD,
            experiment_id="exp-001",
            version="1.0.0",
            tags=["test", "dataset"],
            metadata={"samples": 1000},
        )

    def test_initialization(self, lineage_manager: LineageManager) -> None:
        """Test lineage manager initialization."""
        assert lineage_manager is not None
        assert hasattr(lineage_manager, "storage")

    def test_create_lineage_success(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test successful lineage creation."""
        # Save artifacts first
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        # Create lineage
        lineage = lineage_manager.create_lineage(
            source_artifact_id="artifact-001",
            target_artifact_id="artifact-002",
            relationship_type="evolves_to",
            relationship_description="Model 2 evolved from Model 1",
            confidence=0.95,
        )

        assert lineage is not None
        assert lineage.source_artifact_id == "artifact-001"
        assert lineage.target_artifact_id == "artifact-002"
        assert lineage.relationship_type == "evolves_to"
        assert lineage.confidence == 0.95

    def test_create_lineage_missing_artifacts(
        self, lineage_manager: LineageManager
    ) -> None:
        """Test lineage creation with missing artifacts."""
        with pytest.raises(
            ValueError, match="Source or target artifact not found"
        ):
            lineage_manager.create_lineage(
                source_artifact_id="nonexistent-1",
                target_artifact_id="nonexistent-2",
                relationship_type="evolves_to",
            )

    def test_create_lineage_invalid_relationship_type(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test lineage creation with invalid relationship type."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        with pytest.raises(ValueError, match="Invalid relationship type"):
            lineage_manager.create_lineage(
                source_artifact_id="artifact-001",
                target_artifact_id="artifact-002",
                relationship_type="invalid_type",
            )

    def test_create_lineage_circular_dependency(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test lineage creation that would create circular dependency."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        # Create first lineage
        lineage_manager.create_lineage(
            source_artifact_id="artifact-001",
            target_artifact_id="artifact-002",
            relationship_type="evolves_to",
        )

        # Try to create reverse lineage (should fail)
        with pytest.raises(
            ValueError, match="Lineage would create circular dependency"
        ):
            lineage_manager.create_lineage(
                source_artifact_id="artifact-002",
                target_artifact_id="artifact-001",
                relationship_type="evolves_to",
            )

    def test_update_lineage_success(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test successful lineage update."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        # Create lineage
        lineage = lineage_manager.create_lineage(
            source_artifact_id="artifact-001",
            target_artifact_id="artifact-002",
            relationship_type="evolves_to",
        )

        # Update lineage
        updated_lineage = lineage_manager.update_lineage(
            lineage.lineage_id,
            relationship_description="Updated description",
            confidence=0.98,
        )

        assert updated_lineage is not None
        assert (
            updated_lineage.relationship_description == "Updated description"
        )
        assert updated_lineage.confidence == 0.98

    def test_update_lineage_not_found(
        self, lineage_manager: LineageManager
    ) -> None:
        """Test lineage update with non-existent lineage."""
        result = lineage_manager.update_lineage(
            "nonexistent-lineage",
            relationship_description="Updated description",
        )
        assert result is None

    def test_delete_lineage_success(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test successful lineage deletion."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        # Create lineage
        lineage = lineage_manager.create_lineage(
            source_artifact_id="artifact-001",
            target_artifact_id="artifact-002",
            relationship_type="evolves_to",
        )

        # Delete lineage
        result = lineage_manager.delete_lineage(lineage.lineage_id)
        assert result is True

    def test_delete_lineage_not_found(
        self, lineage_manager: LineageManager
    ) -> None:
        """Test lineage deletion with non-existent lineage."""
        result = lineage_manager.delete_lineage("nonexistent-lineage")
        assert result is False

    def test_get_lineage_path_success(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
        sample_artifact_3: ArtifactEntity,
    ) -> None:
        """Test successful lineage path retrieval."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)
        storage.save_artifact(sample_artifact_3)

        # Create lineage chain: 1 -> 2 -> 3
        lineage_manager.create_lineage(
            source_artifact_id="artifact-001",
            target_artifact_id="artifact-002",
            relationship_type="evolves_to",
        )
        lineage_manager.create_lineage(
            source_artifact_id="artifact-002",
            target_artifact_id="artifact-003",
            relationship_type="evolves_to",
        )

        # Get path from 1 to 3
        path = lineage_manager.get_lineage_path("artifact-001", "artifact-003")
        assert len(path) == 2
        assert path[0].source_artifact_id == "artifact-001"
        assert path[0].target_artifact_id == "artifact-002"
        assert path[1].source_artifact_id == "artifact-002"
        assert path[1].target_artifact_id == "artifact-003"

    def test_get_lineage_path_no_path(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test lineage path retrieval when no path exists."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        # No lineage created
        path = lineage_manager.get_lineage_path("artifact-001", "artifact-002")
        assert len(path) == 0

    def test_get_lineage_tree_success(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
        sample_artifact_3: ArtifactEntity,
    ) -> None:
        """Test successful lineage tree retrieval."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)
        storage.save_artifact(sample_artifact_3)

        # Create lineage relationships
        lineage_manager.create_lineage(
            source_artifact_id="artifact-001",
            target_artifact_id="artifact-002",
            relationship_type="evolves_to",
        )
        lineage_manager.create_lineage(
            source_artifact_id="artifact-002",
            target_artifact_id="artifact-003",
            relationship_type="evolves_to",
        )

        # Get tree for artifact-002
        tree = lineage_manager.get_lineage_tree("artifact-002")
        assert tree["artifact_id"] == "artifact-002"
        assert len(tree["children"]) == 1  # artifact-003
        assert len(tree["parents"]) == 1  # artifact-001

    def test_analyze_lineage_impact_success(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test successful lineage impact analysis."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        # Create lineage
        lineage_manager.create_lineage(
            source_artifact_id="artifact-001",
            target_artifact_id="artifact-002",
            relationship_type="evolves_to",
            confidence=0.95,
        )

        # Analyze impact
        analysis = lineage_manager.analyze_lineage_impact("artifact-001")
        assert analysis["artifact_id"] == "artifact-001"
        assert analysis["direct_relationships"] == 1
        assert "evolves_to" in analysis["relationship_types"]
        assert analysis["average_confidence"] == 0.95

    def test_validate_lineage_integrity_success(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test successful lineage integrity validation."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        # Create valid lineage
        lineage_manager.create_lineage(
            source_artifact_id="artifact-001",
            target_artifact_id="artifact-002",
            relationship_type="evolves_to",
        )

        # Validate integrity
        validation = lineage_manager.validate_lineage_integrity()
        assert validation["total_lineage"] == 1
        assert validation["valid_lineage"] == 1
        assert validation["integrity_score"] == 1.0
        assert len(validation["issues"]) == 0

    def test_validate_lineage_integrity_issues(
        self, lineage_manager: LineageManager, storage: TraceabilityStorage
    ) -> None:
        """Test lineage integrity validation with issues."""
        # Create lineage with non-existent artifacts
        lineage_data = [
            {
                "lineage_id": "lineage-001",
                "source_artifact_id": "nonexistent-1",
                "target_artifact_id": "nonexistent-2",
                "relationship_type": "evolves_to",
                "relationship_description": "Test",
                "confidence": 1.0,
                "created_at": datetime.now().isoformat(),
                "metadata": {},
            }
        ]

        storage._save_lineage(lineage_data)

        # Validate integrity
        validation = lineage_manager.validate_lineage_integrity()
        assert validation["total_lineage"] == 1
        assert validation["valid_lineage"] == 0
        assert validation["integrity_score"] == 0.0
        assert len(validation["issues"]) == 2  # Missing source and target

    def test_validate_relationship_invalid_type(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_2: ArtifactEntity,
    ) -> None:
        """Test relationship validation with invalid type."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_2)

        with pytest.raises(ValueError, match="Invalid relationship type"):
            lineage_manager.create_lineage(
                source_artifact_id="artifact-001",
                target_artifact_id="artifact-002",
                relationship_type="invalid_type",
            )

    def test_validate_relationship_artifact_type_compatibility(
        self,
        lineage_manager: LineageManager,
        storage: TraceabilityStorage,
        sample_artifact_1: ArtifactEntity,
        sample_artifact_3: ArtifactEntity,
    ) -> None:
        """Test relationship validation with incompatible artifact types."""
        storage.save_artifact(sample_artifact_1)
        storage.save_artifact(sample_artifact_3)

        # Try to create invalid relationship (model derived from dataset)
        with pytest.raises(
            ValueError, match="Model cannot be derived from dataset"
        ):
            lineage_manager.create_lineage(
                source_artifact_id="artifact-001",  # Model
                target_artifact_id="artifact-003",  # Dataset
                relationship_type="derived_from",
            )
