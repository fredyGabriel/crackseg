"""
Artifact versioning system for CrackSeg project.

This module provides the ArtifactVersioner class for managing artifact
versions, change tracking, and version history with integrity verification.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VersionInfo:
    """Information about a specific artifact version."""

    version_id: str
    artifact_id: str
    file_path: str
    checksum: str
    file_size: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version_id": self.version_id,
            "artifact_id": self.artifact_id,
            "file_path": self.file_path,
            "checksum": self.checksum,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionInfo":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ArtifactVersion:
    """Complete version information for an artifact."""

    artifact_id: str
    current_version: str
    versions: dict[str, VersionInfo] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "artifact_id": self.artifact_id,
            "current_version": self.current_version,
            "versions": {
                version_id: version.to_dict()
                for version_id, version in self.versions.items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactVersion":
        """Create from dictionary."""
        artifact_version = cls(
            artifact_id=data["artifact_id"],
            current_version=data["current_version"],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

        for version_id, version_data in data.get("versions", {}).items():
            artifact_version.versions[version_id] = VersionInfo.from_dict(
                version_data
            )

        return artifact_version


class ArtifactVersioner:
    """
    Comprehensive artifact versioning system.

    Manages artifact versions, change tracking, and version history with
    integrity verification using SHA256 checksums.
    """

    def __init__(self, version_file: Path | None = None) -> None:
        """
        Initialize ArtifactVersioner.

        Args:
            version_file: Path to version tracking file (optional)
        """
        self.version_file = version_file or Path("artifact_versions.json")
        self.versions: dict[str, ArtifactVersion] = self._load_versions()

    def _load_versions(self) -> dict[str, ArtifactVersion]:
        """Load existing version information."""
        if self.version_file.exists():
            try:
                with open(self.version_file, encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        artifact_id: ArtifactVersion.from_dict(version_data)
                        for artifact_id, version_data in data.items()
                    }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Failed to load versions from {self.version_file}: {e}"
                )
                return {}
        return {}

    def _save_versions(self) -> None:
        """Save version information to file."""
        try:
            self.version_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.version_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        artifact_id: version.to_dict()
                        for artifact_id, version in self.versions.items()
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(
                f"Failed to save versions to {self.version_file}: {e}"
            )

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

    def _generate_version_id(self, artifact_id: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{artifact_id}_v{timestamp}"

    def create_version(
        self,
        artifact_id: str,
        file_path: Path,
        description: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new version of an artifact.

        Args:
            artifact_id: Unique identifier for the artifact
            file_path: Path to the artifact file
            description: Description of this version
            tags: Tags for categorizing this version
            metadata: Additional metadata for this version

        Returns:
            Version ID of the created version

        Raises:
            FileNotFoundError: If the artifact file doesn't exist
            ValueError: If the file is empty or invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"Artifact file is empty: {file_path}")

        # Calculate checksum
        checksum = self._calculate_checksum(file_path)
        if not checksum:
            raise ValueError(f"Failed to calculate checksum for {file_path}")

        # Generate version ID
        version_id = self._generate_version_id(artifact_id)

        # Create version info
        version_info = VersionInfo(
            version_id=version_id,
            artifact_id=artifact_id,
            file_path=str(file_path),
            checksum=checksum,
            file_size=file_path.stat().st_size,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Add to version tracking
        if artifact_id not in self.versions:
            self.versions[artifact_id] = ArtifactVersion(
                artifact_id=artifact_id,
                current_version=version_id,
            )

        artifact_version = self.versions[artifact_id]
        artifact_version.versions[version_id] = version_info
        artifact_version.current_version = version_id
        artifact_version.updated_at = datetime.now().isoformat()

        # Save to file
        self._save_versions()

        logger.info(f"Created version {version_id} for artifact {artifact_id}")

        return version_id

    def get_version_info(
        self, artifact_id: str, version_id: str | None = None
    ) -> VersionInfo | None:
        """
        Get information about a specific version.

        Args:
            artifact_id: ID of the artifact
            version_id: Specific version ID (None for current version)

        Returns:
            Version information or None if not found
        """
        if artifact_id not in self.versions:
            return None

        artifact_version = self.versions[artifact_id]
        target_version = version_id or artifact_version.current_version

        return artifact_version.versions.get(target_version)

    def get_version_history(self, artifact_id: str) -> list[VersionInfo]:
        """
        Get complete version history for an artifact.

        Args:
            artifact_id: ID of the artifact

        Returns:
            List of version information sorted by creation date
        """
        if artifact_id not in self.versions:
            return []

        artifact_version = self.versions[artifact_id]
        versions = list(artifact_version.versions.values())
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions

    def verify_integrity(
        self, artifact_id: str, version_id: str | None = None
    ) -> bool:
        """
        Verify integrity of an artifact version.

        Args:
            artifact_id: ID of the artifact
            version_id: Specific version ID (None for current version)

        Returns:
            True if integrity is verified, False otherwise
        """
        version_info = self.get_version_info(artifact_id, version_id)
        if not version_info:
            return False

        file_path = Path(version_info.file_path)
        if not file_path.exists():
            logger.error(f"Artifact file not found: {file_path}")
            return False

        # Recalculate checksum
        current_checksum = self._calculate_checksum(file_path)
        if not current_checksum:
            return False

        # Compare with stored checksum
        if current_checksum != version_info.checksum:
            logger.error(
                f"Checksum mismatch for {artifact_id} version "
                f"{version_info.version_id}"
            )
            return False

        # Verify file size
        if file_path.stat().st_size != version_info.file_size:
            logger.error(
                f"File size mismatch for {artifact_id} version "
                f"{version_info.version_id}"
            )
            return False

        return True

    def list_artifacts(self) -> list[str]:
        """Get list of all tracked artifacts."""
        return list(self.versions.keys())

    def get_current_version(self, artifact_id: str) -> str | None:
        """Get current version ID for an artifact."""
        if artifact_id in self.versions:
            return self.versions[artifact_id].current_version
        return None

    def delete_version(self, artifact_id: str, version_id: str) -> bool:
        """
        Delete a specific version of an artifact.

        Args:
            artifact_id: ID of the artifact
            version_id: ID of the version to delete

        Returns:
            True if version was deleted, False otherwise
        """
        if artifact_id not in self.versions:
            return False

        artifact_version = self.versions[artifact_id]
        if version_id not in artifact_version.versions:
            return False

        # Don't delete if it's the current version
        if version_id == artifact_version.current_version:
            logger.warning(
                f"Cannot delete current version {version_id} of artifact "
                f"{artifact_id}"
            )
            return False

        # Remove version
        del artifact_version.versions[version_id]
        artifact_version.updated_at = datetime.now().isoformat()

        # Save changes
        self._save_versions()

        logger.info(f"Deleted version {version_id} of artifact {artifact_id}")

        return True

    def get_version_summary(self, artifact_id: str) -> dict[str, Any]:
        """
        Get summary information for an artifact.

        Args:
            artifact_id: ID of the artifact

        Returns:
            Summary dictionary with version information
        """
        if artifact_id not in self.versions:
            return {}

        artifact_version = self.versions[artifact_id]
        versions = list(artifact_version.versions.values())
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return {
            "artifact_id": artifact_id,
            "current_version": artifact_version.current_version,
            "total_versions": len(versions),
            "created_at": artifact_version.created_at,
            "updated_at": artifact_version.updated_at,
            "versions": [
                {
                    "version_id": v.version_id,
                    "created_at": v.created_at,
                    "file_size": v.file_size,
                    "description": v.description,
                    "tags": v.tags,
                }
                for v in versions
            ],
        }
