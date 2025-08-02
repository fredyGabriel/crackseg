"""
Tests for checkpoint integrity verification.

This module tests the specialized checkpoint integrity verifier.
"""

from pathlib import Path

import pytest
import torch

from crackseg.utils.integrity.checkpoint_verifier import (
    CheckpointIntegrityVerifier,
)
from crackseg.utils.integrity.core import VerificationLevel


class TestCheckpointIntegrityVerifier:
    """Test checkpoint integrity verifier functionality."""

    @pytest.fixture
    def valid_checkpoint(self, tmp_path: Path) -> Path:
        """Create a valid PyTorch checkpoint for testing."""
        checkpoint_path = tmp_path / "valid_checkpoint.pth"

        # Create a simple model state dict
        model_state_dict = {
            "conv1.weight": torch.randn(64, 3, 7, 7),
            "conv1.bias": torch.randn(64),
            "fc.weight": torch.randn(10, 512),
            "fc.bias": torch.randn(10),
        }

        # Create optimizer state dict
        optimizer_state_dict = {
            "state": {
                0: {"momentum_buffer": torch.randn(64, 3, 7, 7)},
                1: {"momentum_buffer": torch.randn(64)},
                2: {"momentum_buffer": torch.randn(10, 512)},
                3: {"momentum_buffer": torch.randn(10)},
            },
            "param_groups": [
                {"params": [0, 1], "lr": 0.001},
                {"params": [2, 3], "lr": 0.001},
            ],
        }

        # Create checkpoint data
        checkpoint_data = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "epoch": 10,
            "pytorch_version": torch.__version__,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    @pytest.fixture
    def invalid_checkpoint(self, tmp_path: Path) -> Path:
        """Create an invalid PyTorch checkpoint for testing."""
        checkpoint_path = tmp_path / "invalid_checkpoint.pth"

        # Create checkpoint with missing required fields
        checkpoint_data = {
            "model_state_dict": {},
            "epoch": 10,
            # Missing optimizer_state_dict, pytorch_version, timestamp
        }

        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    @pytest.fixture
    def corrupted_checkpoint(self, tmp_path: Path) -> Path:
        """Create a corrupted checkpoint file for testing."""
        checkpoint_path = tmp_path / "corrupted_checkpoint.pth"
        checkpoint_path.write_text("This is not a valid PyTorch checkpoint")
        return checkpoint_path

    def test_verifier_initialization(self) -> None:
        """Test verifier initialization."""
        verifier = CheckpointIntegrityVerifier(VerificationLevel.STANDARD)
        assert verifier.verification_level == VerificationLevel.STANDARD
        assert len(verifier.required_fields) == 5
        assert "model_state_dict" in verifier.required_fields
        assert "optimizer_state_dict" in verifier.required_fields

    def test_verifier_initialization_custom_fields(self) -> None:
        """Test verifier initialization with custom required fields."""
        custom_fields = ["model_state_dict", "custom_field"]
        verifier = CheckpointIntegrityVerifier(required_fields=custom_fields)
        assert verifier.required_fields == custom_fields

    def test_verify_valid_checkpoint_basic(
        self, valid_checkpoint: Path
    ) -> None:
        """Test verification of valid checkpoint with basic level."""
        verifier = CheckpointIntegrityVerifier(VerificationLevel.BASIC)

        result = verifier.verify(valid_checkpoint)

        assert result.is_valid is True
        assert result.artifact_path == valid_checkpoint
        assert result.verification_level == VerificationLevel.BASIC
        assert result.checksum is not None
        assert result.file_size is not None
        assert len(result.errors) == 0

    def test_verify_valid_checkpoint_standard(
        self, valid_checkpoint: Path
    ) -> None:
        """Test verification of valid checkpoint with standard level."""
        verifier = CheckpointIntegrityVerifier(VerificationLevel.STANDARD)

        result = verifier.verify(valid_checkpoint)

        assert result.is_valid is True
        assert result.checksum is not None
        assert result.file_size is not None
        assert len(result.errors) == 0

        # Check metadata from checkpoint
        assert "epoch" in result.metadata
        assert "pytorch_version" in result.metadata
        assert "timestamp" in result.metadata
        assert "model_keys" in result.metadata
        assert result.metadata["epoch"] == 10
        assert result.metadata["pytorch_version"] == torch.__version__

    def test_verify_valid_checkpoint_thorough(
        self, valid_checkpoint: Path
    ) -> None:
        """Test verification of valid checkpoint with thorough level."""
        verifier = CheckpointIntegrityVerifier(VerificationLevel.THOROUGH)

        result = verifier.verify(valid_checkpoint)

        assert result.is_valid is True
        assert len(result.errors) == 0

        # Check deep analysis metadata
        assert "tensor_count" in result.metadata
        assert "total_parameters" in result.metadata
        assert "parameter_shapes" in result.metadata
        assert result.metadata["tensor_count"] == 4
        assert result.metadata["total_parameters"] > 0

    def test_verify_valid_checkpoint_paranoid(
        self, valid_checkpoint: Path
    ) -> None:
        """Test verification of valid checkpoint with paranoid level."""
        verifier = CheckpointIntegrityVerifier(VerificationLevel.PARANOID)

        result = verifier.verify(valid_checkpoint)

        assert result.is_valid is True
        assert len(result.errors) == 0

        # Check cross-reference validation
        # Should not have warnings about PyTorch version mismatch
        pytorch_warnings = [w for w in result.warnings if "PyTorch" in w]
        assert len(pytorch_warnings) == 0

    def test_verify_invalid_checkpoint(self, invalid_checkpoint: Path) -> None:
        """Test verification of invalid checkpoint."""
        verifier = CheckpointIntegrityVerifier(VerificationLevel.STANDARD)

        result = verifier.verify(invalid_checkpoint)

        assert result.is_valid is False
        assert len(result.errors) > 0

        # Should have errors about missing required fields
        missing_field_errors = [
            e for e in result.errors if "Missing required fields" in e
        ]
        assert len(missing_field_errors) == 1

    def test_verify_corrupted_checkpoint(
        self, corrupted_checkpoint: Path
    ) -> None:
        """Test verification of corrupted checkpoint."""
        verifier = CheckpointIntegrityVerifier(VerificationLevel.STANDARD)

        result = verifier.verify(corrupted_checkpoint)

        assert result.is_valid is False
        assert len(result.errors) > 0

        # Should have error about failed to load checkpoint
        load_errors = [
            e for e in result.errors if "Failed to load checkpoint" in e
        ]
        assert len(load_errors) == 1

    def test_verify_nonexistent_checkpoint(self, tmp_path: Path) -> None:
        """Test verification of nonexistent checkpoint."""
        verifier = CheckpointIntegrityVerifier(VerificationLevel.STANDARD)
        nonexistent_checkpoint = tmp_path / "nonexistent.pth"

        result = verifier.verify(nonexistent_checkpoint)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]

    def test_verify_checkpoint_with_nan_values(self, tmp_path: Path) -> None:
        """Test verification of checkpoint with NaN values."""
        checkpoint_path = tmp_path / "nan_checkpoint.pth"

        # Create checkpoint with NaN values
        model_state_dict = {
            "conv1.weight": torch.tensor([[float("nan"), 1.0], [2.0, 3.0]]),
            "conv1.bias": torch.randn(2),
        }

        checkpoint_data = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": {},
            "epoch": 1,
            "pytorch_version": torch.__version__,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        torch.save(checkpoint_data, checkpoint_path)

        verifier = CheckpointIntegrityVerifier(VerificationLevel.THOROUGH)
        result = verifier.verify(checkpoint_path)

        assert result.is_valid is True  # NaN values are warnings, not errors
        assert len(result.warnings) > 0

        # Should have warning about NaN values
        nan_warnings = [
            w for w in result.warnings if "NaN values detected" in w
        ]
        assert len(nan_warnings) == 1

    def test_verify_checkpoint_with_inf_values(self, tmp_path: Path) -> None:
        """Test verification of checkpoint with Inf values."""
        checkpoint_path = tmp_path / "inf_checkpoint.pth"

        # Create checkpoint with Inf values
        model_state_dict = {
            "conv1.weight": torch.tensor([[float("inf"), 1.0], [2.0, 3.0]]),
            "conv1.bias": torch.randn(2),
        }

        checkpoint_data = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": {},
            "epoch": 1,
            "pytorch_version": torch.__version__,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        torch.save(checkpoint_data, checkpoint_path)

        verifier = CheckpointIntegrityVerifier(VerificationLevel.THOROUGH)
        result = verifier.verify(checkpoint_path)

        assert result.is_valid is True  # Inf values are warnings, not errors
        assert len(result.warnings) > 0

        # Should have warning about Inf values
        inf_warnings = [
            w for w in result.warnings if "Inf values detected" in w
        ]
        assert len(inf_warnings) == 1

    def test_verify_checkpoint_pytorch_version_mismatch(
        self, tmp_path: Path
    ) -> None:
        """Test verification of checkpoint with PyTorch version mismatch."""
        checkpoint_path = tmp_path / "version_mismatch_checkpoint.pth"

        # Create checkpoint with different PyTorch version
        model_state_dict = {"conv1.weight": torch.randn(64, 3, 7, 7)}

        checkpoint_data = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": {},
            "epoch": 1,
            "pytorch_version": "1.0.0",  # Different version
            "timestamp": "2024-01-01T00:00:00Z",
        }

        torch.save(checkpoint_data, checkpoint_path)

        verifier = CheckpointIntegrityVerifier(VerificationLevel.PARANOID)
        result = verifier.verify(checkpoint_path)

        assert result.is_valid is True  # Version mismatch is a warning
        assert len(result.warnings) > 0

        # Should have warning about PyTorch version mismatch
        version_warnings = [
            w for w in result.warnings if "PyTorch" in w and "version" in w
        ]
        assert len(version_warnings) == 1

    def test_verify_checkpoint_empty_model_state_dict(
        self, tmp_path: Path
    ) -> None:
        """Test verification of checkpoint with empty model state dict."""
        checkpoint_path = tmp_path / "empty_model_checkpoint.pth"

        checkpoint_data = {
            "model_state_dict": {},  # Empty state dict
            "optimizer_state_dict": {},
            "epoch": 1,
            "pytorch_version": torch.__version__,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        torch.save(checkpoint_data, checkpoint_path)

        verifier = CheckpointIntegrityVerifier(VerificationLevel.STANDARD)
        result = verifier.verify(checkpoint_path)

        assert result.is_valid is True  # Empty state dict is a warning
        assert len(result.warnings) > 0

        # Should have warning about empty model state dict
        empty_warnings = [w for w in result.warnings if "empty" in w]
        assert len(empty_warnings) == 1

    def test_verify_checkpoint_invalid_model_state_dict(
        self, tmp_path: Path
    ) -> None:
        """Test verification of checkpoint with invalid model state dict."""
        checkpoint_path = tmp_path / "invalid_model_checkpoint.pth"

        checkpoint_data = {
            "model_state_dict": "not a dict",  # Invalid type
            "optimizer_state_dict": {},
            "epoch": 1,
            "pytorch_version": torch.__version__,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        torch.save(checkpoint_data, checkpoint_path)

        verifier = CheckpointIntegrityVerifier(VerificationLevel.STANDARD)
        result = verifier.verify(checkpoint_path)

        assert result.is_valid is False
        assert len(result.errors) > 0

        # Should have error about model state dict not being a dictionary
        dict_errors = [
            e for e in result.errors if "has no attribute 'keys'" in e
        ]
        assert len(dict_errors) == 1
