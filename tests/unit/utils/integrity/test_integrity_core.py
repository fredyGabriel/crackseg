"""
Tests for core integrity verification system.

This module tests the base classes and interfaces for the integrity
verification system.
"""

import hashlib
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from crackseg.utils.integrity.core import (
    IntegrityVerifier,
    VerificationLevel,
    VerificationResult,
)


class MockIntegrityVerifier(IntegrityVerifier):
    """Mock verifier for testing base functionality."""

    def verify(self, artifact_path: Path) -> VerificationResult:
        """Mock verification implementation."""
        result = VerificationResult(
            is_valid=True,
            artifact_path=artifact_path,
            verification_level=self.verification_level,
        )

        # Basic structure verification
        if not self._verify_basic_structure(artifact_path, result):
            return result

        # Calculate checksum
        checksum = self._verify_checksum(artifact_path)
        result.checksum = checksum

        return result


class TestVerificationLevel:
    """Test verification level enumeration."""

    def test_verification_levels(self) -> None:
        """Test that all verification levels are defined."""
        levels = list(VerificationLevel)
        assert len(levels) == 4
        assert VerificationLevel.BASIC in levels
        assert VerificationLevel.STANDARD in levels
        assert VerificationLevel.THOROUGH in levels
        assert VerificationLevel.PARANOID in levels


class TestVerificationResult:
    """Test verification result functionality."""

    def test_verification_result_creation(self) -> None:
        """Test creating a verification result."""
        result = VerificationResult(
            is_valid=True,
            artifact_path=Path("/test/path"),
            verification_level=VerificationLevel.STANDARD,
        )

        assert result.is_valid is True
        assert result.artifact_path == Path("/test/path")
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}

    def test_add_error(self) -> None:
        """Test adding errors to verification result."""
        result = VerificationResult(
            is_valid=True,
            artifact_path=Path("/test/path"),
            verification_level=VerificationLevel.STANDARD,
        )

        result.add_error("Test error")

        assert result.is_valid is False
        assert "Test error" in result.errors
        assert len(result.errors) == 1

    def test_add_warning(self) -> None:
        """Test adding warnings to verification result."""
        result = VerificationResult(
            is_valid=True,
            artifact_path=Path("/test/path"),
            verification_level=VerificationLevel.STANDARD,
        )

        result.add_warning("Test warning")

        assert result.is_valid is True  # Warnings don't invalidate
        assert "Test warning" in result.warnings
        assert len(result.warnings) == 1

    def test_add_metadata(self) -> None:
        """Test adding metadata to verification result."""
        result = VerificationResult(
            is_valid=True,
            artifact_path=Path("/test/path"),
            verification_level=VerificationLevel.STANDARD,
        )

        result.add_metadata("test_key", "test_value")

        assert result.metadata["test_key"] == "test_value"
        assert len(result.metadata) == 1


class TestIntegrityVerifier:
    """Test base integrity verifier functionality."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def test_file(self, temp_dir: Path) -> Path:
        """Create a test file for verification."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")
        return test_file

    def test_verifier_initialization(self) -> None:
        """Test verifier initialization."""
        verifier = MockIntegrityVerifier(VerificationLevel.STANDARD)
        assert verifier.verification_level == VerificationLevel.STANDARD

    def test_calculate_checksum(self, test_file: Path) -> None:
        """Test checksum calculation."""
        verifier = MockIntegrityVerifier()

        # Calculate expected checksum
        expected_checksum = hashlib.sha256()
        with open(test_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                expected_checksum.update(chunk)
        expected = expected_checksum.hexdigest()

        # Calculate actual checksum
        actual = verifier._calculate_checksum(test_file)

        assert actual == expected
        assert len(actual) == 64  # SHA256 hex digest length

    def test_calculate_checksum_nonexistent_file(self, temp_dir: Path) -> None:
        """Test checksum calculation for nonexistent file."""
        verifier = MockIntegrityVerifier()
        nonexistent_file = temp_dir / "nonexistent.txt"

        checksum = verifier._calculate_checksum(nonexistent_file)

        assert checksum == ""

    def test_verify_file_exists(self, test_file: Path) -> None:
        """Test file existence verification."""
        verifier = MockIntegrityVerifier()

        assert verifier._verify_file_exists(test_file) is True

        nonexistent_file = test_file.parent / "nonexistent.txt"
        assert verifier._verify_file_exists(nonexistent_file) is False

    def test_verify_file_size(self, test_file: Path) -> None:
        """Test file size verification."""
        verifier = MockIntegrityVerifier()

        size = verifier._verify_file_size(test_file)
        assert size == len("Test content")

        nonexistent_file = test_file.parent / "nonexistent.txt"
        size = verifier._verify_file_size(nonexistent_file)
        assert size == 0

    def test_verify_basic_structure(self, test_file: Path) -> None:
        """Test basic structure verification."""
        verifier = MockIntegrityVerifier()
        result = VerificationResult(
            is_valid=True,
            artifact_path=test_file,
            verification_level=VerificationLevel.STANDARD,
        )

        success = verifier._verify_basic_structure(test_file, result)

        assert success is True
        assert result.is_valid is True
        assert result.file_size == len("Test content")
        assert len(result.errors) == 0

    def test_verify_basic_structure_nonexistent(self, temp_dir: Path) -> None:
        """Test basic structure verification for nonexistent file."""
        verifier = MockIntegrityVerifier()
        nonexistent_file = temp_dir / "nonexistent.txt"
        result = VerificationResult(
            is_valid=True,
            artifact_path=nonexistent_file,
            verification_level=VerificationLevel.STANDARD,
        )

        success = verifier._verify_basic_structure(nonexistent_file, result)

        assert success is False
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]

    def test_verify_basic_structure_empty_file(self, temp_dir: Path) -> None:
        """Test basic structure verification for empty file."""
        verifier = MockIntegrityVerifier()
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")  # Create empty file

        result = VerificationResult(
            is_valid=True,
            artifact_path=empty_file,
            verification_level=VerificationLevel.STANDARD,
        )

        success = verifier._verify_basic_structure(empty_file, result)

        assert success is False
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "empty" in result.errors[0]

    def test_verify_checksum(self, test_file: Path) -> None:
        """Test checksum verification."""
        verifier = MockIntegrityVerifier()

        checksum = verifier._verify_checksum(test_file)
        assert len(checksum) == 64  # SHA256 hex digest length

        # Test with expected checksum
        expected_checksum = hashlib.sha256(b"Test content").hexdigest()
        checksum = verifier._verify_checksum(test_file, expected_checksum)
        assert checksum == expected_checksum

    def test_mock_verifier_verify(self, test_file: Path) -> None:
        """Test mock verifier verification."""
        verifier = MockIntegrityVerifier(VerificationLevel.STANDARD)

        result = verifier.verify(test_file)

        assert result.is_valid is True
        assert result.artifact_path == test_file
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.checksum is not None
        assert result.file_size == len("Test content")
        assert len(result.errors) == 0

    def test_mock_verifier_verify_nonexistent(self, temp_dir: Path) -> None:
        """Test mock verifier verification for nonexistent file."""
        verifier = MockIntegrityVerifier(VerificationLevel.STANDARD)
        nonexistent_file = temp_dir / "nonexistent.txt"

        result = verifier.verify(nonexistent_file)

        assert result.is_valid is False
        assert result.artifact_path == nonexistent_file
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]
