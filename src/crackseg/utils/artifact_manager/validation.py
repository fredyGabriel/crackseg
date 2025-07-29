"""
Artifact validation functionality.

This module contains the ArtifactValidator class responsible for validating
artifact integrity and metadata.
"""

import hashlib
import logging
from pathlib import Path

from .metadata import ArtifactMetadata

logger = logging.getLogger(__name__)


class ArtifactValidator:
    """Handles validation of artifacts and metadata."""

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    def validate_artifact(
        self, file_path: str | Path, metadata: list[ArtifactMetadata]
    ) -> bool:
        """
        Validate artifact integrity using checksum.

        Args:
            file_path: Path to artifact file
            metadata: List of metadata to search in

        Returns:
            True if artifact is valid, False otherwise
        """
        file_path = Path(file_path)

        # Find metadata for this file
        for meta in metadata:
            if Path(meta.file_path) == file_path:
                current_checksum = self._calculate_checksum(file_path)
                return current_checksum == meta.checksum

        logger.warning(f"No metadata found for {file_path}")
        return False

    def validate_all_artifacts(
        self, metadata: list[ArtifactMetadata]
    ) -> dict[str, bool]:
        """
        Validate integrity of all artifacts in the experiment.

        Args:
            metadata: List of artifact metadata

        Returns:
            Dictionary mapping artifact paths to validation status
        """
        validation_results = {}

        for meta in metadata:
            file_path = Path(meta.file_path)
            if file_path.exists():
                current_checksum = self._calculate_checksum(file_path)
                is_valid = current_checksum == meta.checksum
                validation_results[str(file_path)] = is_valid

                if not is_valid:
                    logger.warning(
                        f"Artifact integrity check failed: {file_path}"
                    )
            else:
                validation_results[str(file_path)] = False
                logger.error(f"Artifact file not found: {file_path}")

        return validation_results

    def repair_artifact_metadata(
        self, file_path: str | Path, metadata: list[ArtifactMetadata]
    ) -> bool:
        """
        Repair metadata for an artifact by recalculating checksum.

        Args:
            file_path: Path to artifact file
            metadata: List of metadata to update

        Returns:
            True if repair was successful, False otherwise
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Cannot repair metadata: file not found {file_path}")
            return False

        # Find and update metadata
        for meta in metadata:
            if Path(meta.file_path) == file_path:
                meta.checksum = self._calculate_checksum(file_path)
                meta.file_size = file_path.stat().st_size
                logger.info(f"Repaired metadata for {file_path}")
                return True

        logger.warning(f"No metadata found to repair for {file_path}")
        return False
