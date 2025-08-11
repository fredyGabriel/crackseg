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
from .utils.storage_utils import (
    convert_datetime_fields,
    load_json_list,
    save_json_list,
)

logger = logging.getLogger(__name__)


def _convert_datetime_fields(data: dict[str, Any]) -> dict[str, Any]:
    return convert_datetime_fields(data)


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
            file_path = self._file(file_name)
            if not file_path.exists():
                save_json_list(file_path, [])

    # -------------------------
    # Generic helpers (DRY)
    # -------------------------
    def _file(self, filename: str) -> Path:
        return self.storage_path / filename

    def _load(self, filename: str) -> list[dict[str, Any]]:
        try:
            return load_json_list(self._file(filename))
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Failed to load {filename}, returning empty list")
            return []

    def _save(self, filename: str, data: list[dict[str, Any]]) -> None:
        save_json_list(self._file(filename), data)

    @staticmethod
    def _find_index_by_id(
        items: list[dict[str, Any]], id_key: str, id_value: str
    ) -> int | None:
        for i, existing in enumerate(items):
            if existing.get(id_key) == id_value:
                return i
        return None

    @classmethod
    def _upsert_by_id(
        cls,
        items: list[dict[str, Any]],
        id_key: str,
        obj_dict: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Insert or update an object in a list by its identifier key.

        Returns updated list and a flag indicating whether it was created (True) or updated (False).
        """
        existing_index = cls._find_index_by_id(
            items, id_key, obj_dict.get(id_key, "")
        )  # type: ignore[arg-type]
        if existing_index is not None:
            items[existing_index] = obj_dict
            return items, False
        items.append(obj_dict)
        return items, True

    @classmethod
    def _delete_by_id(
        cls, items: list[dict[str, Any]], id_key: str, id_value: str
    ) -> tuple[list[dict[str, Any]], bool]:
        original_len = len(items)
        items = [x for x in items if x.get(id_key) != id_value]
        return items, len(items) < original_len

    def save_artifact(self, artifact: ArtifactEntity) -> bool:
        """Save artifact to storage.

        Args:
            artifact: Artifact entity to save

        Returns:
            True if save was successful, False otherwise
        """
        try:
            artifacts = self._load("artifacts.json")

            artifact_dict = artifact.model_dump()
            artifact_dict = _convert_datetime_fields(artifact_dict)
            artifact_dict["updated_at"] = datetime.now().isoformat()

            artifacts, created = self._upsert_by_id(
                artifacts, "artifact_id", artifact_dict
            )
            if created:
                artifacts[-1]["created_at"] = datetime.now().isoformat()

            self._save("artifacts.json", artifacts)
            logger.info(
                f"Saved artifact: {artifact.artifact_id} ({'created' if created else 'updated'})"
            )
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
            experiments = self._load("experiments.json")

            experiment_dict = experiment.model_dump()
            experiment_dict = _convert_datetime_fields(experiment_dict)
            experiment_dict["updated_at"] = datetime.now().isoformat()

            experiments, created = self._upsert_by_id(
                experiments, "experiment_id", experiment_dict
            )
            if created:
                experiments[-1]["created_at"] = datetime.now().isoformat()

            self._save("experiments.json", experiments)
            logger.info(
                f"Saved experiment: {experiment.experiment_id} ({'created' if created else 'updated'})"
            )
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
            versions = self._load("versions.json")

            version_dict = version.model_dump()
            version_dict = _convert_datetime_fields(version_dict)
            version_dict["updated_at"] = datetime.now().isoformat()

            versions, created = self._upsert_by_id(
                versions, "version_id", version_dict
            )
            if created:
                versions[-1]["created_at"] = datetime.now().isoformat()

            self._save("versions.json", versions)
            logger.info(
                f"Saved version: {version.version_id} ({'created' if created else 'updated'})"
            )
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
            lineage_data = self._load("lineage.json")

            lineage_dict = lineage.model_dump()
            lineage_dict = _convert_datetime_fields(lineage_dict)
            lineage_dict["updated_at"] = datetime.now().isoformat()

            lineage_data, created = self._upsert_by_id(
                lineage_data, "lineage_id", lineage_dict
            )
            if created:
                lineage_data[-1]["created_at"] = datetime.now().isoformat()

            self._save("lineage.json", lineage_data)
            logger.info(
                f"Saved lineage: {lineage.lineage_id} ({'created' if created else 'updated'})"
            )
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
            artifacts = self._load("artifacts.json")
            artifacts, deleted = self._delete_by_id(
                artifacts, "artifact_id", artifact_id
            )
            if deleted:
                self._save("artifacts.json", artifacts)
                logger.info(f"Deleted artifact: {artifact_id}")
                return True
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
            experiments = self._load("experiments.json")
            experiments, deleted = self._delete_by_id(
                experiments, "experiment_id", experiment_id
            )
            if deleted:
                self._save("experiments.json", experiments)
                logger.info(f"Deleted experiment: {experiment_id}")
                return True
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
            artifacts = self._load("artifacts.json")
            experiments = self._load("experiments.json")
            versions = self._load("versions.json")
            lineage = self._load("lineage.json")

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
                "artifacts": self._load("artifacts.json"),
                "experiments": self._load("experiments.json"),
                "versions": self._load("versions.json"),
                "lineage": self._load("lineage.json"),
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
                self._save("artifacts.json", import_data["artifacts"])  # type: ignore[arg-type]

            if "experiments" in import_data:
                self._save("experiments.json", import_data["experiments"])  # type: ignore[arg-type]

            if "versions" in import_data:
                self._save("versions.json", import_data["versions"])  # type: ignore[arg-type]

            if "lineage" in import_data:
                self._save("lineage.json", import_data["lineage"])  # type: ignore[arg-type]

            logger.info(f"Imported data from: {import_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return False

    # Removed specialized _load/_save methods in favor of generic helpers
