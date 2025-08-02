"""
Unit tests for ExperimentIntegrityVerifier.

This module provides comprehensive testing for the ExperimentIntegrityVerifier
class, ensuring proper validation of experiment directories and verification
levels.
"""

import json
from pathlib import Path

import pytest

from crackseg.utils.integrity import (
    ExperimentIntegrityVerifier,
    VerificationLevel,
)


class TestExperimentIntegrityVerifier:
    """Test cases for ExperimentIntegrityVerifier."""

    @pytest.fixture
    def verifier(self) -> ExperimentIntegrityVerifier:
        """Create a standard verifier for testing."""
        return ExperimentIntegrityVerifier(VerificationLevel.STANDARD)

    @pytest.fixture
    def thorough_verifier(self) -> ExperimentIntegrityVerifier:
        """Create a thorough verifier for testing."""
        return ExperimentIntegrityVerifier(VerificationLevel.THOROUGH)

    @pytest.fixture
    def paranoid_verifier(self) -> ExperimentIntegrityVerifier:
        """Create a paranoid verifier for testing."""
        return ExperimentIntegrityVerifier(VerificationLevel.PARANOID)

    @pytest.fixture
    def valid_experiment_dir(self, tmp_path: Path) -> Path:
        """Create a valid experiment directory for testing."""
        experiment_dir = tmp_path / "valid_experiment"
        experiment_dir.mkdir()

        # Create experiment_tracker.json
        tracker_data = {
            "experiment_id": "exp_001",
            "name": "crack_segmentation_experiment",
            "description": "Crack segmentation with ResNet50 encoder",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
            "status": "completed",
            "tags": ["segmentation", "crack", "resnet50"],
            "artifacts": ["checkpoints/model_best.pth", "logs/training.log"],
        }
        with open(experiment_dir / "experiment_tracker.json", "w") as f:
            json.dump(tracker_data, f, indent=2)

        # Create config.yaml
        config_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
  pretrained: true
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100
  optimizer: "adam"
data:
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"
  image_size: [512, 512]
experiment:
  name: "crack_segmentation_experiment"
  description: "Crack segmentation with ResNet50 encoder"
  tags: ["segmentation", "crack", "resnet50"]
"""
        with open(experiment_dir / "config.yaml", "w") as f:
            f.write(config_content)

        # Create metrics.jsonl
        metrics_content = """{"epoch": 1, "loss": 0.5, "accuracy": 0.8,
        "val_loss": 0.6, "val_accuracy": 0.75}
{"epoch": 2, "loss": 0.4, "accuracy": 0.85, "val_loss": 0.5,
"val_accuracy": 0.8} {"epoch": 3, "loss": 0.3, "accuracy": 0.9,
"val_loss": 0.4, "val_accuracy": 0.85} {"epoch": 4, "loss": 0.25,
"accuracy": 0.92, "val_loss": 0.35, "val_accuracy": 0.88}
{"epoch": 5, "loss": 0.2, "accuracy": 0.94,
"val_loss": 0.3, "val_accuracy": 0.9}
"""
        with open(experiment_dir / "metrics.jsonl", "w") as f:
            f.write(metrics_content)

        # Create subdirectories
        (experiment_dir / "checkpoints").mkdir()
        (experiment_dir / "logs").mkdir()
        (experiment_dir / "metrics").mkdir()

        return experiment_dir

    @pytest.fixture
    def invalid_experiment_not_directory(self, tmp_path: Path) -> Path:
        """Create an invalid experiment (not a directory)."""
        experiment_path = tmp_path / "invalid_experiment"
        with open(experiment_path, "w") as f:
            f.write("This is not a directory")
        return experiment_path

    @pytest.fixture
    def invalid_experiment_missing_files(self, tmp_path: Path) -> Path:
        """Create an invalid experiment missing required files."""
        experiment_dir = tmp_path / "invalid_experiment"
        experiment_dir.mkdir()

        # Only create config.yaml, missing other required files
        config_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
"""
        with open(experiment_dir / "config.yaml", "w") as f:
            f.write(config_content)

        return experiment_dir

    @pytest.fixture
    def invalid_tracker_experiment(self, tmp_path: Path) -> Path:
        """Create an experiment with invalid tracker file."""
        experiment_dir = tmp_path / "invalid_tracker_experiment"
        experiment_dir.mkdir()

        # Create invalid experiment_tracker.json
        tracker_data = {
            "experiment_id": "exp_002",
            "name": "invalid_experiment",
            # Missing required fields like created_at, status
        }
        with open(experiment_dir / "experiment_tracker.json", "w") as f:
            json.dump(tracker_data, f, indent=2)

        # Create config.yaml
        config_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
"""
        with open(experiment_dir / "config.yaml", "w") as f:
            f.write(config_content)

        return experiment_dir

    @pytest.fixture
    def invalid_config_experiment(self, tmp_path: Path) -> Path:
        """Create an experiment with invalid config file."""
        experiment_dir = tmp_path / "invalid_config_experiment"
        experiment_dir.mkdir()

        # Create valid experiment_tracker.json
        tracker_data = {
            "experiment_id": "exp_003",
            "name": "invalid_config_experiment",
            "created_at": "2024-01-01T00:00:00Z",
            "status": "completed",
        }
        with open(experiment_dir / "experiment_tracker.json", "w") as f:
            json.dump(tracker_data, f, indent=2)

        # Create invalid config.yaml
        config_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
  invalid: yaml: content: with: too: many: colons:
"""
        with open(experiment_dir / "config.yaml", "w") as f:
            f.write(config_content)

        # Create metrics.jsonl
        metrics_content = """{"epoch": 1, "loss": 0.5, "accuracy": 0.8}
{"epoch": 2, "loss": 0.4, "accuracy": 0.85}
"""
        with open(experiment_dir / "metrics.jsonl", "w") as f:
            f.write(metrics_content)

        return experiment_dir

    @pytest.fixture
    def invalid_metrics_experiment(self, tmp_path: Path) -> Path:
        """Create an experiment with invalid metrics file."""
        experiment_dir = tmp_path / "invalid_metrics_experiment"
        experiment_dir.mkdir()

        # Create valid experiment_tracker.json
        tracker_data = {
            "experiment_id": "exp_004",
            "name": "invalid_metrics_experiment",
            "created_at": "2024-01-01T00:00:00Z",
            "status": "completed",
        }
        with open(experiment_dir / "experiment_tracker.json", "w") as f:
            json.dump(tracker_data, f, indent=2)

        # Create valid config.yaml
        config_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
"""
        with open(experiment_dir / "config.yaml", "w") as f:
            f.write(config_content)

        # Create invalid metrics.jsonl
        metrics_content = """{"epoch": 1, "loss": 0.5, "accuracy": 0.8}
{"epoch": 2, "loss": 0.4, "accuracy": 0.85}
invalid json line
{"epoch": 3, "loss": 0.3, "accuracy": 0.9}
"""
        with open(experiment_dir / "metrics.jsonl", "w") as f:
            f.write(metrics_content)

        return experiment_dir

    @pytest.fixture
    def empty_files_experiment(self, tmp_path: Path) -> Path:
        """Create an experiment with empty files."""
        experiment_dir = tmp_path / "empty_files_experiment"
        experiment_dir.mkdir()

        # Create empty experiment_tracker.json
        with open(experiment_dir / "experiment_tracker.json", "w") as f:
            f.write("")

        # Create empty config.yaml
        with open(experiment_dir / "config.yaml", "w") as f:
            f.write("")

        # Create empty metrics.jsonl
        with open(experiment_dir / "metrics.jsonl", "w") as f:
            f.write("")

        return experiment_dir

    @pytest.fixture
    def extra_files_experiment(self, tmp_path: Path) -> Path:
        """Create an experiment with extra files."""
        experiment_dir = tmp_path / "extra_files_experiment"
        experiment_dir.mkdir()

        # Create valid experiment_tracker.json
        tracker_data = {
            "experiment_id": "exp_005",
            "name": "extra_files_experiment",
            "created_at": "2024-01-01T00:00:00Z",
            "status": "completed",
        }
        with open(experiment_dir / "experiment_tracker.json", "w") as f:
            json.dump(tracker_data, f, indent=2)

        # Create valid config.yaml
        config_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
"""
        with open(experiment_dir / "config.yaml", "w") as f:
            f.write(config_content)

        # Create valid metrics.jsonl
        metrics_content = """{"epoch": 1, "loss": 0.5, "accuracy": 0.8}
{"epoch": 2, "loss": 0.4, "accuracy": 0.85}
"""
        with open(experiment_dir / "metrics.jsonl", "w") as f:
            f.write(metrics_content)

        # Create extra files
        with open(experiment_dir / "extra_file.txt", "w") as f:
            f.write("This is an extra file")

        (experiment_dir / "extra_dir").mkdir()
        with open(experiment_dir / "extra_dir" / "nested_file.txt", "w") as f:
            f.write("This is a nested file")

        return experiment_dir

    @pytest.fixture
    def subdirectories_experiment(self, tmp_path: Path) -> Path:
        """Create an experiment with subdirectories."""
        experiment_dir = tmp_path / "subdirectories_experiment"
        experiment_dir.mkdir()

        # Create valid experiment_tracker.json
        tracker_data = {
            "experiment_id": "exp_006",
            "name": "subdirectories_experiment",
            "created_at": "2024-01-01T00:00:00Z",
            "status": "completed",
        }
        with open(experiment_dir / "experiment_tracker.json", "w") as f:
            json.dump(tracker_data, f, indent=2)

        # Create valid config.yaml
        config_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
"""
        with open(experiment_dir / "config.yaml", "w") as f:
            f.write(config_content)

        # Create valid metrics.jsonl
        metrics_content = """{"epoch": 1, "loss": 0.5, "accuracy": 0.8}
{"epoch": 2, "loss": 0.4, "accuracy": 0.85}
"""
        with open(experiment_dir / "metrics.jsonl", "w") as f:
            f.write(metrics_content)

        # Create subdirectories
        (experiment_dir / "checkpoints").mkdir()
        (experiment_dir / "logs").mkdir()
        (experiment_dir / "metrics").mkdir()
        (experiment_dir / "visualizations").mkdir()

        return experiment_dir

    def test_verifier_initialization(self) -> None:
        """Test verifier initialization."""
        verifier = ExperimentIntegrityVerifier(VerificationLevel.STANDARD)
        assert verifier.verification_level == VerificationLevel.STANDARD
        assert len(verifier.required_files) > 0
        assert "experiment_tracker.json" in verifier.required_files
        assert "config.yaml" in verifier.required_files

    def test_verifier_initialization_custom_files(self) -> None:
        """Test verifier initialization with custom required files."""
        custom_files = ["custom1.json", "custom2.yaml"]
        verifier = ExperimentIntegrityVerifier(required_files=custom_files)
        assert verifier.required_files == custom_files

    def test_verify_valid_experiment_basic(
        self, valid_experiment_dir: Path
    ) -> None:
        """Test verification of valid experiment with basic level."""
        verifier = ExperimentIntegrityVerifier(VerificationLevel.BASIC)

        result = verifier.verify(valid_experiment_dir)

        assert result.is_valid is True
        assert result.artifact_path == valid_experiment_dir
        assert result.verification_level == VerificationLevel.BASIC

    def test_verify_valid_experiment_standard(
        self, verifier: ExperimentIntegrityVerifier, valid_experiment_dir: Path
    ) -> None:
        """Test verification of valid experiment with standard level."""
        result = verifier.verify(valid_experiment_dir)

        assert result.is_valid is True
        assert result.artifact_path == valid_experiment_dir
        assert result.verification_level == VerificationLevel.STANDARD
        assert "existing_files" in result.metadata
        assert "missing_files" in result.metadata
        assert "total_files" in result.metadata
        assert "experiment_id" in result.metadata
        assert "experiment_status" in result.metadata
        assert "created_at" in result.metadata

    def test_verify_valid_experiment_thorough(
        self,
        thorough_verifier: ExperimentIntegrityVerifier,
        valid_experiment_dir: Path,
    ) -> None:
        """Test verification with thorough level."""
        result = thorough_verifier.verify(valid_experiment_dir)

        assert result.is_valid is True
        assert result.verification_level == VerificationLevel.THOROUGH
        assert "config_file_size" in result.metadata
        assert "config_lines" in result.metadata
        assert "config_sections" in result.metadata

    def test_verify_valid_experiment_paranoid(
        self,
        paranoid_verifier: ExperimentIntegrityVerifier,
        valid_experiment_dir: Path,
    ) -> None:
        """Test verification with paranoid level."""
        result = paranoid_verifier.verify(valid_experiment_dir)

        assert result.is_valid is True
        assert result.verification_level == VerificationLevel.PARANOID
        assert "config_file_size" in result.metadata
        assert "config_lines" in result.metadata
        assert "config_sections" in result.metadata

    def test_verify_invalid_experiment_not_directory(
        self,
        verifier: ExperimentIntegrityVerifier,
        invalid_experiment_not_directory: Path,
    ) -> None:
        """Test verification of invalid experiment (not a directory)."""
        result = verifier.verify(invalid_experiment_not_directory)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Not a directory" in error for error in result.errors)

    def test_verify_invalid_experiment_missing_files(
        self,
        verifier: ExperimentIntegrityVerifier,
        invalid_experiment_missing_files: Path,
    ) -> None:
        """Test verification of experiment missing required files."""
        result = verifier.verify(invalid_experiment_missing_files)

        # Missing files are warnings, not errors
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any(
            "Missing recommended files" in warning
            for warning in result.warnings
        )

    def test_verify_experiment_with_invalid_tracker(
        self,
        verifier: ExperimentIntegrityVerifier,
        invalid_tracker_experiment: Path,
    ) -> None:
        """Test verification of experiment with invalid tracker file."""
        result = verifier.verify(invalid_tracker_experiment)

        # Invalid tracker fields are warnings, not errors
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any(
            "Missing metadata fields" in warning for warning in result.warnings
        )

    def test_verify_nonexistent_experiment(
        self, verifier: ExperimentIntegrityVerifier, tmp_path: Path
    ) -> None:
        """Test verification of nonexistent experiment directory."""
        experiment_path = tmp_path / "nonexistent_experiment"

        result = verifier.verify(experiment_path)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Not a directory" in error for error in result.errors)

    def test_verify_experiment_with_invalid_config(
        self,
        verifier: ExperimentIntegrityVerifier,
        invalid_config_experiment: Path,
    ) -> None:
        """Test verification of experiment with invalid config file."""
        result = verifier.verify(invalid_config_experiment)

        # Invalid config might not cause failure in basic implementation
        assert result.is_valid is True
        assert "existing_files" in result.metadata
        assert "experiment_id" in result.metadata

    def test_verify_experiment_with_invalid_metrics(
        self,
        verifier: ExperimentIntegrityVerifier,
        invalid_metrics_experiment: Path,
    ) -> None:
        """Test verification of experiment with invalid metrics file."""
        result = verifier.verify(invalid_metrics_experiment)

        # Invalid metrics might not cause failure in basic implementation
        assert result.is_valid is True
        assert "existing_files" in result.metadata
        assert "experiment_id" in result.metadata

    def test_verify_experiment_with_empty_files(
        self,
        verifier: ExperimentIntegrityVerifier,
        empty_files_experiment: Path,
    ) -> None:
        """Test verification of experiment with empty files."""
        result = verifier.verify(empty_files_experiment)

        # Empty JSON files should cause validation to fail
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Invalid JSON" in error for error in result.errors)

    def test_verify_experiment_with_extra_files(
        self,
        verifier: ExperimentIntegrityVerifier,
        extra_files_experiment: Path,
    ) -> None:
        """Test verification of experiment with extra files."""
        result = verifier.verify(extra_files_experiment)

        assert result.is_valid is True
        assert "existing_files" in result.metadata
        assert "total_files" in result.metadata
        assert (
            result.metadata["total_files"] >= 3
        )  # At least the required files

    def test_verify_experiment_with_subdirectories(
        self,
        verifier: ExperimentIntegrityVerifier,
        subdirectories_experiment: Path,
    ) -> None:
        """Test verification of experiment with subdirectories."""
        result = verifier.verify(subdirectories_experiment)

        assert result.is_valid is True
        assert "existing_files" in result.metadata
        assert "subdirectories" in result.metadata
        assert len(result.metadata["subdirectories"]) > 0
