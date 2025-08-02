"""
Core integrity verification system.

This module provides the base classes and interfaces for the integrity
verification system used across all CrackSeg artifacts.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VerificationLevel(Enum):
    """Levels of verification thoroughness."""

    BASIC = auto()  # File existence and basic structure
    STANDARD = auto()  # + Content validation and checksums
    THOROUGH = auto()  # + Deep content analysis
    PARANOID = auto()  # + Cross-reference validation


@dataclass
class VerificationResult:
    """Result of an integrity verification operation."""

    is_valid: bool
    artifact_path: Path
    verification_level: VerificationLevel
    checksum: str | None = None
    file_size: int | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Add an error to the verification result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the verification result."""
        self.warnings.append(warning)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the verification result."""
        self.metadata[key] = value


class IntegrityVerifier(ABC):
    """Base class for integrity verification operations."""

    def __init__(
        self,
        verification_level: VerificationLevel = VerificationLevel.STANDARD,
    ):
        """
        Initialize the integrity verifier.

        Args:
            verification_level: Level of verification thoroughness
        """
        self.verification_level = verification_level

    @abstractmethod
    def verify(self, artifact_path: Path) -> VerificationResult:
        """
        Verify the integrity of an artifact.

        Args:
            artifact_path: Path to the artifact to verify

        Returns:
            VerificationResult with verification details
        """
        pass

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    def _verify_file_exists(self, file_path: Path) -> bool:
        """Verify that a file exists and is accessible."""
        return file_path.exists() and file_path.is_file()

    def _verify_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            return file_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            return 0

    def _verify_basic_structure(
        self, file_path: Path, result: VerificationResult
    ) -> bool:
        """Verify basic file structure and accessibility."""
        if not self._verify_file_exists(file_path):
            result.add_error(f"File does not exist: {file_path}")
            return False

        file_size = self._verify_file_size(file_path)
        if file_size == 0:
            result.add_error(f"File is empty: {file_path}")
            return False

        result.file_size = file_size
        return True

    def _verify_checksum(
        self, file_path: Path, expected_checksum: str | None = None
    ) -> str:
        """Verify file checksum and optionally compare with expected value."""
        current_checksum = self._calculate_checksum(file_path)

        if expected_checksum and current_checksum != expected_checksum:
            logger.error(
                f"Checksum mismatch for {file_path}: "
                f"expected {expected_checksum}, got {current_checksum}"
            )

        return current_checksum
