"""
Traceability storage module for CrackSeg project.

This module provides data persistence, CRUD operations, and management
capabilities for the traceability system using JSON-based storage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    ArtifactEntity,
    ExperimentEntity,
    LineageEntity,
    VersionEntity,
)

logger = logging.getLogger(__name__)


def _convert_datetime_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Convert datetime fields to ISO format strings for JSON serialization.

    Args:
        data: Dictionary containing potential datetime fields

    Returns:
        Dictionary with datetime fields converted to ISO strings
    """
    converted = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            converted[key] = value.isoformat()
        elif isinstance(value, Path):
            converted[key] = str(value)
        else:
            converted[key] = value
    return converted


class TraceabilityStorage:
    """Storage manager for traceability data using JSON files."""

    def __init__(self, storage_path: Path) -> None:
        """Initialize storage with specified path.

        Args:
            storage_path: Directory path for storing traceability data
        """
        self.storage_path = Path(storage_path)
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize storage directory and files."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize JSON files if they don't exist
        files = [
            "artifacts.json",
            "experiments.json",
            "versions.json",
            "lineage.json",
        ]

        for file_name in files:
            file_path = self.storage_path / file_name
            if not file_path.exists():
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    def save_artifact(self, artifact: ArtifactEntity) -> bool:
        """Save artifact to storage.

        Args:
            artifact: Artifact entity to save

        Returns:
            True if save was successful, False otherwise
        """
        try:
            artifacts = self._load_artifacts()

            # Check if artifact already exists
            existing_index = None
            for i, existing in enumerate(artifacts):
                if existing.get("artifact_id") == artifact.artifact_id:
                    existing_index = i
                    break

            artifact_dict = artifact.model_dump()
            artifact_dict = _convert_datetime_fields(artifact_dict)
            artifact_dict["updated_at"] = datetime.now().isoformat()

            if existing_index is not None:
                # Update existing artifact
                artifacts[existing_index] = artifact_dict
            else:
                # Add new artifact
                artifact_dict["created_at"] = datetime.now().isoformat()
                artifacts.append(artifact_dict)

            self._save_artifacts(artifacts)
            logger.info(f"Saved artifact: {artifact.artifact_id}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to save artifact {artifact.artifact_id}: {e}"
            )
            return False

    def save_experiment(self, experiment: ExperimentEntity) -> bool:
        """Save experiment to storage.

        Args:
            experiment: Experiment entity to save

        Returns:
            True if save was successful, False otherwise
        """
        try:
            experiments = self._load_experiments()

            # Check if experiment already exists
            existing_index = None
            for i, existing in enumerate(experiments):
                if existing.get("experiment_id") == experiment.experiment_id:
                    existing_index = i
                    break

            experiment_dict = experiment.model_dump()
            experiment_dict = _convert_datetime_fields(experiment_dict)
            experiment_dict["updated_at"] = datetime.now().isoformat()

            if existing_index is not None:
                # Update existing experiment
                experiments[existing_index] = experiment_dict
            else:
                # Add new experiment
                experiment_dict["created_at"] = datetime.now().isoformat()
                experiments.append(experiment_dict)

            self._save_experiments(experiments)
            logger.info(f"Saved experiment: {experiment.experiment_id}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to save experiment {experiment.experiment_id}: {e}"
            )
            return False

    def save_version(self, version: VersionEntity) -> bool:
        """Save version to storage.

        Args:
            version: Version entity to save

        Returns:
            True if save was successful, False otherwise
        """
        try:
            versions = self._load_versions()

            # Check if version already exists
            existing_index = None
            for i, existing in enumerate(versions):
                if existing.get("version_id") == version.version_id:
                    existing_index = i
                    break

            version_dict = version.model_dump()
            version_dict = _convert_datetime_fields(version_dict)
            version_dict["updated_at"] = datetime.now().isoformat()

            if existing_index is not None:
                # Update existing version
                versions[existing_index] = version_dict
            else:
                # Add new version
                version_dict["created_at"] = datetime.now().isoformat()
                versions.append(version_dict)

            self._save_versions(versions)
            logger.info(f"Saved version: {version.version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save version {version.version_id}: {e}")
            return False

    def save_lineage(self, lineage: LineageEntity) -> bool:
        """Save lineage relationship to storage.

        Args:
            lineage: Lineage entity to save

        Returns:
            True if save was successful, False otherwise
        """
        try:
            lineage_data = self._load_lineage()

            # Check if lineage already exists
            existing_index = None
            for i, existing in enumerate(lineage_data):
                if existing.get("lineage_id") == lineage.lineage_id:
                    existing_index = i
                    break

            lineage_dict = lineage.model_dump()
            lineage_dict = _convert_datetime_fields(lineage_dict)
            lineage_dict["updated_at"] = datetime.now().isoformat()

            if existing_index is not None:
                # Update existing lineage
                lineage_data[existing_index] = lineage_dict
            else:
                # Add new lineage
                lineage_dict["created_at"] = datetime.now().isoformat()
                lineage_data.append(lineage_dict)

            self._save_lineage(lineage_data)
            logger.info(f"Saved lineage: {lineage.lineage_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save lineage {lineage.lineage_id}: {e}")
            return False

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact from storage.

        Args:
            artifact_id: ID of artifact to delete

        Returns:
            True if delete was successful, False otherwise
        """
        try:
            artifacts = self._load_artifacts()
            original_count = len(artifacts)

            artifacts = [
                a for a in artifacts if a.get("artifact_id") != artifact_id
            ]

            if len(artifacts) < original_count:
                self._save_artifacts(artifacts)
                logger.info(f"Deleted artifact: {artifact_id}")
                return True
            else:
                logger.warning(f"Artifact not found: {artifact_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment from storage.

        Args:
            experiment_id: ID of experiment to delete

        Returns:
            True if delete was successful, False otherwise
        """
        try:
            experiments = self._load_experiments()
            original_count = len(experiments)

            experiments = [
                e
                for e in experiments
                if e.get("experiment_id") != experiment_id
            ]

            if len(experiments) < original_count:
                self._save_experiments(experiments)
                logger.info(f"Deleted experiment: {experiment_id}")
                return True
            else:
                logger.warning(f"Experiment not found: {experiment_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            artifacts = self._load_artifacts()
            experiments = self._load_experiments()
            versions = self._load_versions()
            lineage = self._load_lineage()

            return {
                "total_artifacts": len(artifacts),
                "total_experiments": len(experiments),
                "total_versions": len(versions),
                "total_lineage": len(lineage),
                "storage_path": str(self.storage_path),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}

    def export_data(self, export_path: Path) -> bool:
        """Export all data to a single file.

        Args:
            export_path: Path to export file

        Returns:
            True if export was successful, False otherwise
        """
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "artifacts": self._load_artifacts(),
                "experiments": self._load_experiments(),
                "versions": self._load_versions(),
                "lineage": self._load_lineage(),
            }

            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported data to: {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False

    def import_data(self, import_path: Path) -> bool:
        """Import data from export file.

        Args:
            import_path: Path to import file

        Returns:
            True if import was successful, False otherwise
        """
        try:
            with open(import_path, encoding="utf-8") as f:
                import_data = json.load(f)

            # Import each data type
            if "artifacts" in import_data:
                self._save_artifacts(import_data["artifacts"])

            if "experiments" in import_data:
                self._save_experiments(import_data["experiments"])

            if "versions" in import_data:
                self._save_versions(import_data["versions"])

            if "lineage" in import_data:
                self._save_lineage(import_data["lineage"])

            logger.info(f"Imported data from: {import_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return False

    def _load_artifacts(self) -> list[dict[str, Any]]:
        """Load artifacts from storage."""
        try:
            file_path = self.storage_path / "artifacts.json"
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Failed to load artifacts, returning empty list")
            return []

    def _load_experiments(self) -> list[dict[str, Any]]:
        """Load experiments from storage."""
        try:
            file_path = self.storage_path / "experiments.json"
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Failed to load experiments, returning empty list")
            return []

    def _load_versions(self) -> list[dict[str, Any]]:
        """Load versions from storage."""
        try:
            file_path = self.storage_path / "versions.json"
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Failed to load versions, returning empty list")
            return []

    def _load_lineage(self) -> list[dict[str, Any]]:
        """Load lineage from storage."""
        try:
            file_path = self.storage_path / "lineage.json"
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Failed to load lineage, returning empty list")
            return []

    def _save_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        """Save artifacts to storage."""
        file_path = self.storage_path / "artifacts.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2)

    def _save_experiments(self, experiments: list[dict[str, Any]]) -> None:
        """Save experiments to storage."""
        file_path = self.storage_path / "experiments.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)

    def _save_versions(self, versions: list[dict[str, Any]]) -> None:
        """Save versions to storage."""
        file_path = self.storage_path / "versions.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(versions, f, indent=2)

    def _save_lineage(self, lineage: list[dict[str, Any]]) -> None:
        """Save lineage to storage."""
        file_path = self.storage_path / "lineage.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(lineage, f, indent=2)
