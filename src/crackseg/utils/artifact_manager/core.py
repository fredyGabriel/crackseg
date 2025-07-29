"""
Core ArtifactManager functionality.

This module contains the main ArtifactManager class with basic initialization,
directory management, and metadata handling.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .metadata import ArtifactMetadata
from .storage import ArtifactStorage
from .validation import ArtifactValidator
from .versioning import ArtifactVersioner

logger = logging.getLogger(__name__)


@dataclass
class ArtifactManagerConfig:
    """Configuration for ArtifactManager."""

    base_path: str = "artifacts"
    experiment_name: str | None = None
    auto_create_dirs: bool = True
    validate_on_save: bool = True
    enable_versioning: bool = True


class ArtifactManager:
    """
    Comprehensive artifact management system for ML experiments.

    Handles saving, loading, validation, and organization of all experiment
    artifacts including models, metrics, visualizations, and configurations.
    """

    def __init__(self, config: ArtifactManagerConfig | None = None) -> None:
        """
        Initialize ArtifactManager.

        Args:
            config: Configuration for the artifact manager
        """
        self.config = config or ArtifactManagerConfig()

        self.base_path = Path(self.config.base_path)
        self.experiment_name = (
            self.config.experiment_name
            or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.experiment_path = self.base_path / self.experiment_name

        # Initialize components
        self.storage = ArtifactStorage(self.experiment_path)
        self.validator = ArtifactValidator()

        # Initialize versioning if enabled
        self.versioner = None
        if self.config.enable_versioning:
            version_file = self.experiment_path / "artifact_versions.json"
            self.versioner = ArtifactVersioner(version_file)

        # Ensure directory structure exists
        if self.config.auto_create_dirs:
            self._ensure_directories()

        # Initialize metadata tracking
        self.metadata_file = self.experiment_path / "metadata.json"
        self.metadata: list[ArtifactMetadata] = self._load_metadata()

        # Create empty metadata file if it doesn't exist
        if not self.metadata_file.exists():
            self._save_metadata()

        logger.info(
            f"ArtifactManager initialized for experiment: "
            f"{self.experiment_name}"
        )

    def _ensure_directories(self) -> None:
        """Create necessary directory structure."""
        directories = [
            "models",
            "logs",
            "metrics",
            "visualizations",
            "predictions",
            "reports",
            "configs",
        ]

        for directory in directories:
            (self.experiment_path / directory).mkdir(
                parents=True, exist_ok=True
            )

    def _load_metadata(self) -> list[ArtifactMetadata]:
        """Load existing metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                    return [ArtifactMetadata(**item) for item in data]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load metadata: {e}")
        return []

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(
                    [metadata.to_dict() for metadata in self.metadata],
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def connect_with_experiment_manager(self, experiment_manager: Any) -> None:
        """
        Connect with ExperimentManager for coordinated artifact management.

        Args:
            experiment_manager: ExperimentManager instance
        """
        # This method can be extended to integrate with ExperimentManager
        # For now, it's a placeholder for future integration
        logger.info("Connected with ExperimentManager")

    def get_artifact_info(
        self, file_path: str | Path
    ) -> ArtifactMetadata | None:
        """
        Get metadata for a specific artifact.

        Args:
            file_path: Path to the artifact file

        Returns:
            ArtifactMetadata if found, None otherwise
        """
        file_path_str = str(file_path)
        for metadata in self.metadata:
            if metadata.file_path == file_path_str:
                return metadata
        return None

    def list_artifacts(
        self, artifact_type: str | None = None
    ) -> list[ArtifactMetadata]:
        """
        List all artifacts or filter by type.

        Args:
            artifact_type: Optional filter by artifact type

        Returns:
            List of artifact metadata
        """
        if artifact_type is None:
            return self.metadata.copy()

        return [
            metadata
            for metadata in self.metadata
            if metadata.artifact_type == artifact_type
        ]

    def cleanup_artifacts(self, keep_latest: int = 5) -> None:
        """
        Clean up old artifacts, keeping only the latest ones.

        Args:
            keep_latest: Number of latest artifacts to keep per type
        """
        # Group artifacts by type
        artifacts_by_type: dict[str, list[ArtifactMetadata]] = {}
        for metadata in self.metadata:
            artifact_type = metadata.artifact_type
            if artifact_type not in artifacts_by_type:
                artifacts_by_type[artifact_type] = []
            artifacts_by_type[artifact_type].append(metadata)

        # Keep only the latest artifacts per type
        for _artifact_type, artifacts in artifacts_by_type.items():
            # Sort by timestamp (newest first)
            artifacts.sort(key=lambda x: x.timestamp, reverse=True)

            # Remove excess artifacts
            if len(artifacts) > keep_latest:
                artifacts_to_remove = artifacts[keep_latest:]
                for artifact in artifacts_to_remove:
                    self.metadata.remove(artifact)
                    logger.info(f"Removed old artifact: {artifact.file_path}")

        self._save_metadata()

    def export_experiment_summary(self) -> dict[str, Any]:
        """
        Export a summary of all artifacts in the experiment.

        Returns:
            Dictionary containing experiment summary
        """
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_path": str(self.experiment_path),
            "total_artifacts": len(self.metadata),
            "artifact_types": {},
            "created_at": datetime.now().isoformat(),
        }

        # Group artifacts by type
        for metadata in self.metadata:
            artifact_type = metadata.artifact_type
            if artifact_type not in summary["artifact_types"]:
                summary["artifact_types"][artifact_type] = {
                    "count": 0,
                    "total_size": 0,
                    "artifacts": [],
                }

            summary["artifact_types"][artifact_type]["count"] += 1
            summary["artifact_types"][artifact_type][
                "total_size"
            ] += metadata.file_size
            summary["artifact_types"][artifact_type]["artifacts"].append(
                {
                    "file_path": metadata.file_path,
                    "file_size": metadata.file_size,
                    "timestamp": metadata.timestamp,
                    "description": metadata.description,
                    "tags": metadata.tags,
                }
            )

        return summary

    # Versioning methods (if versioning is enabled)
    def create_version(
        self,
        artifact_id: str,
        file_path: Path,
        description: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Create a new version of an artifact.

        Args:
            artifact_id: Unique identifier for the artifact
            file_path: Path to the artifact file
            description: Description of this version
            tags: Tags for categorizing this version
            metadata: Additional metadata for this version

        Returns:
            Version ID if successful, None if versioning is disabled
        """
        if not self.versioner:
            logger.warning("Versioning is disabled")
            return None

        return self.versioner.create_version(
            artifact_id, file_path, description, tags, metadata
        )

    def get_version_info(
        self, artifact_id: str, version_id: str | None = None
    ) -> Any | None:
        """
        Get information about a specific version.

        Args:
            artifact_id: ID of the artifact
            version_id: Specific version ID (None for current version)

        Returns:
            Version information or None if not found
        """
        if not self.versioner:
            return None

        return self.versioner.get_version_info(artifact_id, version_id)

    def get_version_history(self, artifact_id: str) -> list[Any]:
        """
        Get complete version history for an artifact.

        Args:
            artifact_id: ID of the artifact

        Returns:
            List of version information sorted by creation date
        """
        if not self.versioner:
            return []

        return self.versioner.get_version_history(artifact_id)

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
        if not self.versioner:
            return False

        return self.versioner.verify_integrity(artifact_id, version_id)

    def get_version_summary(self, artifact_id: str) -> dict[str, Any]:
        """
        Get summary information for an artifact.

        Args:
            artifact_id: ID of the artifact

        Returns:
            Summary dictionary with version information
        """
        if not self.versioner:
            return {}

        return self.versioner.get_version_summary(artifact_id)
