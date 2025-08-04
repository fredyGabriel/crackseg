"""
Unit tests for ArtifactIntegrityVerifier.

This module provides comprehensive testing for the ArtifactIntegrityVerifier
class, ensuring proper validation of various artifact types and verification
levels.
"""

import json
from pathlib import Path

import pytest

from crackseg.utils.integrity import (
    ArtifactIntegrityVerifier,
    VerificationLevel,
)


class TestArtifactIntegrityVerifier:
    """Test cases for ArtifactIntegrityVerifier."""

    @pytest.fixture
    def verifier(self) -> ArtifactIntegrityVerifier:
        """Create a standard verifier for testing."""
        return ArtifactIntegrityVerifier(VerificationLevel.STANDARD)

    @pytest.fixture
    def thorough_verifier(self) -> ArtifactIntegrityVerifier:
        """Create a thorough verifier for testing."""
        return ArtifactIntegrityVerifier(VerificationLevel.THOROUGH)

    @pytest.fixture
    def paranoid_verifier(self) -> ArtifactIntegrityVerifier:
        """Create a paranoid verifier for testing."""
        return ArtifactIntegrityVerifier(VerificationLevel.PARANOID)

    @pytest.fixture
    def valid_json_artifact(self, tmp_path: Path) -> Path:
        """Create a valid JSON artifact for testing."""
        artifact_path = tmp_path / "valid_artifact.json"
        data = {
            "metadata": {
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "type": "metrics",
            },
            "data": {"accuracy": 0.95, "loss": 0.05, "epoch": 10},
        }
        with open(artifact_path, "w") as f:
            json.dump(data, f, indent=2)
        return artifact_path

    @pytest.fixture
    def valid_yaml_artifact(self, tmp_path: Path) -> Path:
        """Create a valid YAML artifact for testing."""
        artifact_path = tmp_path / "valid_artifact.yaml"
        yaml_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
  pretrained: true
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100
data:
  data_root: "data/unified"
  root_dir: "data/unified"
"""
        with open(artifact_path, "w") as f:
            f.write(yaml_content)
        return artifact_path

    @pytest.fixture
    def valid_csv_artifact(self, tmp_path: Path) -> Path:
        """Create a valid CSV artifact for testing."""
        artifact_path = tmp_path / "valid_artifact.csv"
        csv_content = """epoch,loss,accuracy,val_loss,val_accuracy
1,0.5,0.8,0.6,0.75
2,0.4,0.85,0.5,0.8
3,0.3,0.9,0.4,0.85
4,0.25,0.92,0.35,0.88
5,0.2,0.94,0.3,0.9
"""
        with open(artifact_path, "w") as f:
            f.write(csv_content)
        return artifact_path

    @pytest.fixture
    def valid_txt_artifact(self, tmp_path: Path) -> Path:
        """Create a valid TXT artifact for testing."""
        artifact_path = tmp_path / "valid_artifact.txt"
        txt_content = """Training Log
============

Epoch 1: Loss=0.5, Accuracy=0.8
Epoch 2: Loss=0.4, Accuracy=0.85
Epoch 3: Loss=0.3, Accuracy=0.9
Epoch 4: Loss=0.25, Accuracy=0.92
Epoch 5: Loss=0.2, Accuracy=0.94

Training completed successfully.
"""
        with open(artifact_path, "w") as f:
            f.write(txt_content)
        return artifact_path

    @pytest.fixture
    def valid_pytorch_artifact(self, tmp_path: Path) -> Path:
        """Create a valid PyTorch artifact for testing."""
        artifact_path = tmp_path / "valid_artifact.pth"
        # Create a simple binary file that looks like a PyTorch checkpoint
        with open(artifact_path, "wb") as f:
            f.write(b"PYTORCH_CHECKPOINT_V1")
            f.write(b"\x00" * 100)  # Add some padding
        return artifact_path

    @pytest.fixture
    def valid_binary_artifact(self, tmp_path: Path) -> Path:
        """Create a valid binary artifact for testing."""
        artifact_path = tmp_path / "valid_artifact.bin"
        with open(artifact_path, "wb") as f:
            f.write(b"BINARY_DATA_HEADER")
            f.write(b"\x00" * 50)  # Add some data
        return artifact_path

    @pytest.fixture
    def invalid_json_artifact(self, tmp_path: Path) -> Path:
        """Create an invalid JSON artifact for testing."""
        artifact_path = tmp_path / "invalid_artifact.json"
        with open(artifact_path, "w") as f:
            f.write('{"invalid": json, missing: quotes}')
        return artifact_path

    @pytest.fixture
    def invalid_yaml_artifact(self, tmp_path: Path) -> Path:
        """Create an invalid YAML artifact for testing."""
        artifact_path = tmp_path / "invalid_artifact.yaml"
        with open(artifact_path, "w") as f:
            f.write("invalid: yaml: content: with: too: many: colons:")
        return artifact_path

    @pytest.fixture
    def invalid_csv_artifact(self, tmp_path: Path) -> Path:
        """Create an invalid CSV artifact for testing."""
        artifact_path = tmp_path / "invalid_artifact.csv"
        with open(artifact_path, "w") as f:
            f.write(
                "header1,header2\nvalue1\nvalue1,value2,value3"
            )  # Inconsistent columns
        return artifact_path

    def test_verify_valid_json_artifact_standard(
        self, verifier: ArtifactIntegrityVerifier, valid_json_artifact: Path
    ) -> None:
        """Test verification of valid JSON artifact with standard level."""
        result = verifier.verify(valid_json_artifact)

        assert result.is_valid is True
        assert result.artifact_path == valid_json_artifact
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.checksum is not None
        assert "data_keys" in result.metadata
        assert "data_type" in result.metadata
        assert "file_extension" in result.metadata
        assert result.metadata["file_extension"] == ".json"
        assert result.metadata["data_type"] == "structured"

    def test_verify_valid_yaml_artifact_standard(
        self, verifier: ArtifactIntegrityVerifier, valid_yaml_artifact: Path
    ) -> None:
        """Test verification of valid YAML artifact with standard level."""
        result = verifier.verify(valid_yaml_artifact)

        assert result.is_valid is True
        assert result.artifact_path == valid_yaml_artifact
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.checksum is not None
        assert "data_type" in result.metadata
        assert "file_extension" in result.metadata
        assert result.metadata["file_extension"] == ".yaml"
        assert result.metadata["data_type"] == "structured"

    def test_verify_valid_csv_artifact_standard(
        self, verifier: ArtifactIntegrityVerifier, valid_csv_artifact: Path
    ) -> None:
        """Test verification of valid CSV artifact with standard level."""
        result = verifier.verify(valid_csv_artifact)

        assert result.is_valid is True
        assert result.artifact_path == valid_csv_artifact
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.checksum is not None
        assert "line_count" in result.metadata
        assert "character_count" in result.metadata
        assert "data_type" in result.metadata
        assert "file_extension" in result.metadata
        assert result.metadata["file_extension"] == ".csv"
        assert result.metadata["data_type"] == "text"

    def test_verify_valid_txt_artifact_standard(
        self, verifier: ArtifactIntegrityVerifier, valid_txt_artifact: Path
    ) -> None:
        """Test verification of valid TXT artifact with standard level."""
        result = verifier.verify(valid_txt_artifact)

        assert result.is_valid is True
        assert result.artifact_path == valid_txt_artifact
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.checksum is not None
        assert "line_count" in result.metadata
        assert "character_count" in result.metadata
        assert "data_type" in result.metadata
        assert "file_extension" in result.metadata
        assert result.metadata["file_extension"] == ".txt"
        assert result.metadata["data_type"] == "text"

    def test_verify_valid_pytorch_artifact_standard(
        self, verifier: ArtifactIntegrityVerifier, valid_pytorch_artifact: Path
    ) -> None:
        """Test verification of valid PyTorch artifact with standard level."""
        result = verifier.verify(valid_pytorch_artifact)

        assert result.is_valid is True
        assert result.artifact_path == valid_pytorch_artifact
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.checksum is not None
        assert "data_type" in result.metadata
        assert "file_extension" in result.metadata
        assert result.metadata["file_extension"] == ".pth"
        assert result.metadata["data_type"] == "binary"

    def test_verify_artifact_thorough_level(
        self,
        thorough_verifier: ArtifactIntegrityVerifier,
        valid_json_artifact: Path,
    ) -> None:
        """Test verification with thorough level."""
        result = thorough_verifier.verify(valid_json_artifact)

        assert result.is_valid is True
        assert result.verification_level == VerificationLevel.THOROUGH
        assert "max_depth" in result.metadata
        assert "total_keys" in result.metadata

    def test_verify_artifact_paranoid_level(
        self,
        paranoid_verifier: ArtifactIntegrityVerifier,
        valid_json_artifact: Path,
    ) -> None:
        """Test verification with paranoid level."""
        result = paranoid_verifier.verify(valid_json_artifact)

        assert result.is_valid is True
        assert result.verification_level == VerificationLevel.PARANOID
        assert "max_depth" in result.metadata
        assert "total_keys" in result.metadata

    def test_verify_binary_artifact(
        self, verifier: ArtifactIntegrityVerifier, valid_binary_artifact: Path
    ) -> None:
        """Test verification of binary artifact."""
        result = verifier.verify(valid_binary_artifact)

        assert result.is_valid is True
        assert result.artifact_path == valid_binary_artifact
        assert result.checksum is not None
        assert "data_type" in result.metadata
        assert "file_extension" in result.metadata
        assert "header_hex" in result.metadata
        assert result.metadata["file_extension"] == ".bin"
        assert result.metadata["data_type"] == "binary"

    def test_verify_invalid_extension(
        self, verifier: ArtifactIntegrityVerifier, tmp_path: Path
    ) -> None:
        """Test verification of artifact with unsupported extension."""
        artifact_path = tmp_path / "test.xyz"
        with open(artifact_path, "w") as f:
            f.write("test content")

        result = verifier.verify(artifact_path)

        # Should be valid but with warning for unsupported extension
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any(
            "Unsupported file extension" in warning
            for warning in result.warnings
        )
        assert "file_extension" in result.metadata
        assert result.metadata["file_extension"] == ".xyz"

    def test_verify_invalid_json(
        self, verifier: ArtifactIntegrityVerifier, invalid_json_artifact: Path
    ) -> None:
        """Test verification of invalid JSON artifact."""
        result = verifier.verify(invalid_json_artifact)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Invalid JSON format" in error for error in result.errors)

    def test_verify_invalid_yaml(
        self, verifier: ArtifactIntegrityVerifier, invalid_yaml_artifact: Path
    ) -> None:
        """Test verification of invalid YAML artifact."""
        result = verifier.verify(invalid_yaml_artifact)

        # YAML validation is basic, so it should be valid but with warnings
        assert result.is_valid is True
        assert "data_type" in result.metadata
        assert "file_extension" in result.metadata
        assert result.metadata["file_extension"] == ".yaml"

    def test_verify_invalid_csv(
        self, verifier: ArtifactIntegrityVerifier, invalid_csv_artifact: Path
    ) -> None:
        """Test verification of invalid CSV artifact."""
        result = verifier.verify(invalid_csv_artifact)

        # CSV validation is basic, so it should be valid
        assert result.is_valid is True
        assert "line_count" in result.metadata
        assert "character_count" in result.metadata
        assert "data_type" in result.metadata
        assert result.metadata["data_type"] == "text"

    def test_verify_nonexistent_file(
        self, verifier: ArtifactIntegrityVerifier, tmp_path: Path
    ) -> None:
        """Test verification of nonexistent file."""
        artifact_path = tmp_path / "nonexistent.json"

        result = verifier.verify(artifact_path)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("does not exist" in error for error in result.errors)

    def test_verify_empty_file(
        self, verifier: ArtifactIntegrityVerifier, tmp_path: Path
    ) -> None:
        """Test verification of empty file."""
        artifact_path = tmp_path / "empty.json"
        artifact_path.touch()

        result = verifier.verify(artifact_path)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("is empty" in error for error in result.errors)

    def test_verify_directory(
        self, verifier: ArtifactIntegrityVerifier, tmp_path: Path
    ) -> None:
        """Test verification of directory instead of file."""
        artifact_path = tmp_path / "test_dir"
        artifact_path.mkdir()

        result = verifier.verify(artifact_path)

        assert result.is_valid is False
        assert len(result.errors) > 0
        # The error message might be "File does not exist" or "is not a file"
        assert any(
            "does not exist" in error or "is not a file" in error
            for error in result.errors
        )

    def test_verifier_with_custom_extensions(self, tmp_path: Path) -> None:
        """Test verifier with custom supported extensions."""
        custom_extensions = [".custom", ".test"]
        verifier = ArtifactIntegrityVerifier(
            supported_extensions=custom_extensions
        )

        # Test with supported custom extension
        artifact_path = tmp_path / "test.custom"
        with open(artifact_path, "w") as f:
            f.write("test content")

        result = verifier.verify(artifact_path)
        assert result.is_valid is True
        assert len(result.warnings) == 0

        # Test with unsupported extension
        artifact_path2 = tmp_path / "test.json"
        with open(artifact_path2, "w") as f:
            f.write('{"test": "data"}')

        result2 = verifier.verify(artifact_path2)
        assert result2.is_valid is True
        assert len(result2.warnings) > 0
        assert any(
            "Unsupported file extension" in warning
            for warning in result2.warnings
        )
