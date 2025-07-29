"""
Unit tests for ExperimentTracker component.

Tests experiment metadata tracking, initialization, and core functionality.
"""

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from crackseg.utils.experiment import ExperimentTracker
from crackseg.utils.experiment.metadata import ExperimentMetadata


class TestExperimentTracker:
    """Test suite for ExperimentTracker component."""

    @pytest.fixture
    def temp_experiment_dir(self) -> Iterator[Path]:
        """Create temporary experiment directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_config(self) -> DictConfig:
        """Create sample configuration for testing."""
        config_dict = {
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "learning_rate": 0.001,
                "optimizer": {"_target_": "torch.optim.Adam"},
                "loss": {"_target_": "src.training.losses.BCEDiceLoss"},
            },
            "model": {
                "_target_": "src.model.UNet",
                "encoder": {"_target_": "src.model.encoders.ResNetEncoder"},
                "decoder": {"_target_": "src.model.decoders.UNetDecoder"},
            },
            "data": {
                "dataset": "crack_dataset",
                "root_dir": "data/",
            },
        }
        return OmegaConf.create(config_dict)

    def test_tracker_initialization(self, temp_experiment_dir: Path) -> None:
        """Test ExperimentTracker initialization."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        assert tracker.experiment_id == "test-123"
        assert tracker.experiment_name == "test_experiment"
        assert tracker.experiment_dir == temp_experiment_dir
        assert (
            tracker.metadata_file
            == temp_experiment_dir / "experiment_tracker.json"
        )

    def test_tracker_with_config(
        self, temp_experiment_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test ExperimentTracker initialization with configuration."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
            config=sample_config,
        )

        assert tracker.config == sample_config
        assert tracker.metadata.config_hash != ""
        assert "epochs" in tracker.metadata.config_summary
        assert tracker.metadata.config_summary["epochs"] == 100

    def test_tracker_metadata_creation(
        self, temp_experiment_dir: Path
    ) -> None:
        """Test that metadata is created with environment information."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        metadata = tracker.metadata

        assert metadata.experiment_id == "test-123"
        assert metadata.experiment_name == "test_experiment"
        assert metadata.status == "created"
        assert metadata.python_version != ""
        assert metadata.pytorch_version != ""
        assert metadata.platform != ""

    def test_tracker_metadata_persistence(
        self, temp_experiment_dir: Path
    ) -> None:
        """Test that metadata is saved and loaded correctly."""
        # Create tracker
        tracker1 = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Modify metadata and save
        tracker1.metadata.description = "Test description"
        tracker1.metadata.tags = ["test", "unit"]
        tracker1._save_metadata()  # Explicitly save

        # Create new tracker instance (should load existing metadata)
        tracker2 = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        assert tracker2.metadata.description == "Test description"
        assert tracker2.metadata.tags == ["test", "unit"]

    def test_metadata_access(self, temp_experiment_dir: Path) -> None:
        """Test metadata access methods."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Test get_metadata
        metadata = tracker.get_metadata()
        assert isinstance(metadata, ExperimentMetadata)
        assert metadata.experiment_id == "test-123"

        # Test get_metadata_dict
        metadata_dict = tracker.get_metadata_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["experiment_id"] == "test-123"

    def test_description_update(self, temp_experiment_dir: Path) -> None:
        """Test experiment description updates."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        tracker.update_description("Updated description")
        assert tracker.metadata.description == "Updated description"

    def test_tags_management(self, temp_experiment_dir: Path) -> None:
        """Test tag management functionality."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Add tags
        tracker.add_tags(["tag1", "tag2"])
        assert "tag1" in tracker.metadata.tags
        assert "tag2" in tracker.metadata.tags

        # Add duplicate tag (should not add)
        tracker.add_tags(["tag1"])
        assert tracker.metadata.tags.count("tag1") == 1

        # Remove tags
        tracker.remove_tags(["tag1"])
        assert "tag1" not in tracker.metadata.tags
        assert "tag2" in tracker.metadata.tags

    def test_auto_save_disabled(self, temp_experiment_dir: Path) -> None:
        """Test behavior when auto_save is disabled."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
            auto_save=False,
        )

        # Modify metadata
        tracker.metadata.description = "Test description"

        # Check that file wasn't automatically saved (should not exist)
        metadata_file = temp_experiment_dir / "experiment_tracker.json"
        assert not metadata_file.exists()

        # Manual save should work
        tracker._save_metadata()
        assert metadata_file.exists()

        with open(metadata_file) as f:
            saved_data = json.load(f)

        assert saved_data["description"] == "Test description"

    @patch("subprocess.run")
    def test_git_metadata_collection(
        self, mock_run: Mock, temp_experiment_dir: Path
    ) -> None:
        """Test Git metadata collection."""
        # Mock successful git commands
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc123\n"

        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Git metadata should be collected
        assert tracker.metadata.git_commit == "abc123"

    def test_config_hash_calculation(
        self, temp_experiment_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test configuration hash calculation."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
            config=sample_config,
        )

        # Hash should be calculated
        assert tracker.metadata.config_hash != ""
        assert len(tracker.metadata.config_hash) == 64  # SHA-256 hash length

        # Same config should produce same hash
        tracker2 = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-456",
            experiment_name="test_experiment2",
            config=sample_config,
        )

        assert tracker.metadata.config_hash == tracker2.metadata.config_hash

    def test_config_summary_extraction(
        self, temp_experiment_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test configuration summary extraction."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
            config=sample_config,
        )

        summary = tracker.metadata.config_summary

        assert summary["epochs"] == 100
        assert summary["batch_size"] == 16
        assert summary["learning_rate"] == 0.001
        assert "Adam" in summary["optimizer"]
        assert "BCEDiceLoss" in summary["loss"]
        assert summary["model_type"] == "src.model.UNet"
        assert "ResNetEncoder" in summary["encoder"]
        assert "UNetDecoder" in summary["decoder"]
        assert summary["dataset"] == "crack_dataset"
        assert summary["data_root"] == "data/"

    def test_error_handling_metadata_load(
        self, temp_experiment_dir: Path
    ) -> None:
        """Test error handling when loading corrupted metadata."""
        # Create corrupted metadata file
        metadata_file = temp_experiment_dir / "experiment_tracker.json"
        with open(metadata_file, "w") as f:
            f.write("invalid json content")

        # Should handle corruption gracefully
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Should use initial metadata
        assert tracker.metadata.experiment_id == "test-123"
        assert tracker.metadata.experiment_name == "test_experiment"

    def test_error_handling_metadata_save(
        self, temp_experiment_dir: Path
    ) -> None:
        """Test error handling when saving metadata fails."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Make the directory read-only to cause save failure
        temp_experiment_dir.chmod(0o444)

        # Should handle save failure gracefully
        tracker.update_description("Test description")

        # Restore permissions
        temp_experiment_dir.chmod(0o755)

        # Should continue working
        assert tracker.metadata.description == "Test description"
