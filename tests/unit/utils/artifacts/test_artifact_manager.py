"""
Unit tests for ArtifactManager.

Tests the artifact management functionality including saving, loading,
validation, and metadata tracking.
"""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import yaml

from crackseg.utils.artifact_manager import (
    ArtifactManager,
    ArtifactManagerConfig,
    ArtifactMetadata,
)


class SimpleModel(nn.Module):
    """Simple test model for artifact testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestArtifactMetadata:
    """Test ArtifactMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test creating ArtifactMetadata with default values."""
        metadata = ArtifactMetadata(experiment_name="test_experiment")

        assert metadata.experiment_name == "test_experiment"
        assert metadata.artifact_type == ""
        assert metadata.file_path == ""
        assert metadata.file_size == 0
        assert metadata.checksum == ""
        assert metadata.description == ""
        assert metadata.tags == []
        assert metadata.dependencies == []

    def test_metadata_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        metadata = ArtifactMetadata(
            experiment_name="test_experiment",
            artifact_type="model",
            file_path="/path/to/file.pth",
            file_size=1024,
            checksum="abc123",
            description="Test model",
            tags=["model", "pytorch"],
            dependencies=["config.yaml"],
        )

        result = metadata.to_dict()

        assert result["experiment_name"] == "test_experiment"
        assert result["artifact_type"] == "model"
        assert result["file_path"] == "/path/to/file.pth"
        assert result["file_size"] == 1024
        assert result["checksum"] == "abc123"
        assert result["description"] == "Test model"
        assert result["tags"] == ["model", "pytorch"]
        assert result["dependencies"] == ["config.yaml"]


class TestArtifactManager:
    """Test ArtifactManager functionality."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def artifact_manager(self, temp_dir: Path) -> ArtifactManager:
        """Create ArtifactManager instance for testing."""
        config = ArtifactManagerConfig(
            base_path=str(temp_dir), experiment_name="test_experiment"
        )
        return ArtifactManager(config)

    def test_initialization(self, artifact_manager: ArtifactManager) -> None:
        """Test ArtifactManager initialization."""
        assert artifact_manager.experiment_name == "test_experiment"
        assert artifact_manager.metadata == []
        assert artifact_manager.metadata_file.exists()

    def test_directory_structure_creation(self, temp_dir: Path) -> None:
        """Test that directory structure is created correctly."""
        config = ArtifactManagerConfig(base_path=str(temp_dir))
        manager = ArtifactManager(config)
        experiment_dir = temp_dir / manager.experiment_name

        expected_dirs = [
            "models",
            "logs",
            "metrics",
            "visualizations",
            "predictions",
            "reports",
            "configs",
        ]

        for dir_name in expected_dirs:
            assert (experiment_dir / dir_name).exists()
            assert (experiment_dir / dir_name).is_dir()

    def test_save_model(self, artifact_manager: ArtifactManager) -> None:
        """Test saving a model with metadata."""
        model = SimpleModel()
        filename = "test_model.pth"

        # Use the storage component directly
        file_path, meta = artifact_manager.storage.save_model(
            model, filename, artifact_manager.experiment_name
        )
        artifact_manager.metadata.append(meta)
        artifact_manager._save_metadata()

        # Check file was created
        assert Path(file_path).exists()

        # Check metadata was created
        assert len(artifact_manager.metadata) == 1
        meta = artifact_manager.metadata[0]
        assert meta.artifact_type == "model"
        assert meta.file_path == file_path
        assert meta.description == f"Trained model: {filename}"
        assert "model" in meta.tags
        assert "pytorch" in meta.tags

    def test_save_metrics(self, artifact_manager: ArtifactManager) -> None:
        """Test saving metrics."""
        metrics = {"accuracy": 0.95, "loss": 0.1}
        filename = "test_metrics.json"

        # Use the storage component directly
        file_path, meta = artifact_manager.storage.save_metrics(
            metrics, filename, artifact_manager.experiment_name, "Test metrics"
        )
        artifact_manager.metadata.append(meta)
        artifact_manager._save_metadata()

        # Check file was created
        assert Path(file_path).exists()

        # Check file content
        with open(file_path) as f:
            saved_data = json.load(f)

        assert saved_data["accuracy"] == 0.95
        assert saved_data["loss"] == 0.1
        assert "timestamp" in saved_data
        assert saved_data["experiment_name"] == "test_experiment"

        # Check metadata
        assert len(artifact_manager.metadata) == 1
        meta = artifact_manager.metadata[0]
        assert meta.artifact_type == "metrics"
        assert meta.description == "Test metrics"

    def test_save_config(self, artifact_manager: ArtifactManager) -> None:
        """Test saving configuration."""
        config = {"learning_rate": 0.001, "batch_size": 32}
        filename = "test_config.yaml"

        # Use the storage component directly
        file_path, meta = artifact_manager.storage.save_config(
            config, filename, artifact_manager.experiment_name, "Test config"
        )
        artifact_manager.metadata.append(meta)
        artifact_manager._save_metadata()

        # Check file was created
        assert Path(file_path).exists()

        # Check file content
        with open(file_path) as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["learning_rate"] == 0.001
        assert saved_data["batch_size"] == 32
        assert "timestamp" in saved_data
        assert saved_data["experiment_name"] == "test_experiment"

        # Check metadata
        assert len(artifact_manager.metadata) == 1
        meta = artifact_manager.metadata[0]
        assert meta.artifact_type == "config"
        assert meta.description == "Test config"

    def test_save_visualization(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test saving visualization file."""
        # Create a temporary source file
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp_file:
            tmp_file.write(b"fake image data")
            source_path = Path(tmp_file.name)

        try:
            filename = "test_plot.png"

            # Use the storage component directly
            file_path, meta = artifact_manager.storage.save_visualization(
                source_path,
                filename,
                artifact_manager.experiment_name,
                "Test visualization",
            )
            artifact_manager.metadata.append(meta)
            artifact_manager._save_metadata()

            # Check file was copied
            assert Path(file_path).exists()

            # Check metadata
            assert len(artifact_manager.metadata) == 1
            meta = artifact_manager.metadata[0]
            assert meta.artifact_type == "visualization"
            assert meta.description == "Test visualization"
            assert "visualization" in meta.tags
            assert "png" in meta.tags

        finally:
            # Cleanup
            source_path.unlink(missing_ok=True)

    def test_validate_artifact(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test artifact validation."""
        # Create a test file
        test_file = artifact_manager.experiment_path / "test.txt"
        test_file.write_text("test content")

        # Add metadata for the file
        meta = ArtifactMetadata(
            experiment_name="test_experiment",
            artifact_type="test",
            file_path=str(test_file),
            file_size=test_file.stat().st_size,
            checksum=artifact_manager.validator._calculate_checksum(test_file),
            description="Test file",
        )
        artifact_manager.metadata.append(meta)

        # Test validation
        assert artifact_manager.validator.validate_artifact(
            test_file, artifact_manager.metadata
        )

        # Test with corrupted file
        test_file.write_text("corrupted content")
        assert not artifact_manager.validator.validate_artifact(
            test_file, artifact_manager.metadata
        )

    def test_validate_all_artifacts(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test validation of all artifacts."""
        # Create test files
        files = []
        for i in range(3):
            test_file = artifact_manager.experiment_path / f"test_{i}.txt"
            test_file.write_text(f"content {i}")

            meta = ArtifactMetadata(
                experiment_name="test_experiment",
                artifact_type="test",
                file_path=str(test_file),
                file_size=test_file.stat().st_size,
                checksum=artifact_manager.validator._calculate_checksum(
                    test_file
                ),
                description=f"Test file {i}",
            )
            artifact_manager.metadata.append(meta)
            files.append(test_file)

        # Test validation
        results = artifact_manager.validator.validate_all_artifacts(
            artifact_manager.metadata
        )
        assert len(results) == 3
        assert all(results.values())

        # Corrupt one file
        files[0].write_text("corrupted")
        results = artifact_manager.validator.validate_all_artifacts(
            artifact_manager.metadata
        )
        assert not results[str(files[0])]
        assert results[str(files[1])]
        assert results[str(files[2])]

    def test_repair_artifact_metadata(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test repairing artifact metadata."""
        # Create a test file
        test_file = artifact_manager.experiment_path / "test.txt"
        test_file.write_text("test content")

        # Add metadata with wrong checksum
        meta = ArtifactMetadata(
            experiment_name="test_experiment",
            artifact_type="test",
            file_path=str(test_file),
            file_size=test_file.stat().st_size,
            checksum="wrong_checksum",
            description="Test file",
        )
        artifact_manager.metadata.append(meta)

        # Test repair
        assert artifact_manager.validator.repair_artifact_metadata(
            test_file, artifact_manager.metadata
        )

        # Check that checksum was updated
        updated_meta = artifact_manager.metadata[0]
        correct_checksum = artifact_manager.validator._calculate_checksum(
            test_file
        )
        assert updated_meta.checksum == correct_checksum

    def test_list_artifacts(self, artifact_manager: ArtifactManager) -> None:
        """Test listing artifacts by type."""
        # Add different types of metadata
        for artifact_type in ["model", "metrics", "config"]:
            meta = ArtifactMetadata(
                experiment_name="test_experiment",
                artifact_type=artifact_type,
                file_path=f"/path/to/{artifact_type}.file",
                description=f"Test {artifact_type}",
            )
            artifact_manager.metadata.append(meta)

        # Test listing all artifacts
        all_artifacts = artifact_manager.list_artifacts()
        assert len(all_artifacts) == 3

        # Test filtering by type
        model_artifacts = artifact_manager.list_artifacts("model")
        assert len(model_artifacts) == 1
        assert model_artifacts[0].artifact_type == "model"

    def test_get_artifact_info(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test getting artifact information."""
        # Add metadata
        meta = ArtifactMetadata(
            experiment_name="test_experiment",
            artifact_type="model",
            file_path="/path/to/model.pth",
            description="Test model",
        )
        artifact_manager.metadata.append(meta)

        # Test getting info
        result = artifact_manager.get_artifact_info("/path/to/model.pth")
        assert result is not None
        assert result.artifact_type == "model"
        assert result.description == "Test model"

        # Test non-existent file
        result = artifact_manager.get_artifact_info("/path/to/nonexistent.pth")
        assert result is None

    def test_export_experiment_summary(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test exporting experiment summary."""
        # Add some metadata
        for i in range(3):
            meta = ArtifactMetadata(
                experiment_name="test_experiment",
                artifact_type=f"type_{i}",
                file_path=f"/path/to/file_{i}",
                file_size=100 + i,
                description=f"Test file {i}",
            )
            artifact_manager.metadata.append(meta)

        summary = artifact_manager.export_experiment_summary()

        assert summary["experiment_name"] == "test_experiment"
        assert summary["total_artifacts"] == 3
        # Check for expected fields in summary
        assert "experiment_name" in summary
        assert "total_artifacts" in summary
        assert "artifact_types" in summary
        assert "created_at" in summary

    def test_connect_with_experiment_manager(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test connecting with ExperimentManager."""
        # Mock ExperimentManager
        mock_experiment_manager = Mock()
        mock_experiment_manager.experiment_dir = Path("/mock/experiment/dir")
        mock_experiment_manager.base_dir = Path("/mock/base/dir")
        mock_experiment_manager.experiment_id = "mock_experiment_id"

        # The method is currently a placeholder, so it should not change any
        # attributes
        artifact_manager.connect_with_experiment_manager(
            mock_experiment_manager
        )

        # Verify that the method doesn't change the artifact manager's
        # attributes (since it's just a placeholder that logs a message)
        assert artifact_manager.experiment_path != Path("/mock/experiment/dir")
        assert artifact_manager.base_path != Path("/mock/base/dir")

    def test_error_handling_save_model(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test error handling in save_model."""
        # Test with invalid model - pass None directly
        with pytest.raises((ValueError, TypeError, AttributeError)):
            artifact_manager.storage.save_model(
                None,  # type: ignore
                "test.pth",
                "test_experiment",
            )

    def test_error_handling_save_metrics(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test error handling in save_metrics."""
        # Test with invalid metrics - pass None directly
        with pytest.raises((ValueError, TypeError)):
            artifact_manager.storage.save_metrics(
                None,  # type: ignore
                "test.json",
                "test_experiment",
            )

    def test_metadata_persistence(self, temp_dir: Path) -> None:
        """Test that metadata persists between ArtifactManager instances."""
        # Create first instance and add metadata
        config1 = ArtifactManagerConfig(base_path=str(temp_dir))
        manager1 = ArtifactManager(config1)
        model = SimpleModel()

        # Save model using storage component
        _, meta = manager1.storage.save_model(
            model, "test_model.pth", manager1.experiment_name
        )
        manager1.metadata.append(meta)
        manager1._save_metadata()

        # Create second instance and check metadata is loaded
        config2 = ArtifactManagerConfig(
            base_path=str(temp_dir), experiment_name=manager1.experiment_name
        )
        manager2 = ArtifactManager(config2)

        assert len(manager2.metadata) == 1
        assert manager2.metadata[0].artifact_type == "model"
