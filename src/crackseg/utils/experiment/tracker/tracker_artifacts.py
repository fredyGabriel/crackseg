"""
Artifact management for ExperimentTracker.

This module provides artifact management methods for the ExperimentTracker
component.
"""

from collections.abc import Callable
from typing import Any

from crackseg.utils.experiment.metadata import ExperimentMetadata
from crackseg.utils.logging.base import get_logger

logger = get_logger(__name__)


class ExperimentArtifactManager:
    """Manages artifact associations for experiments."""

    def __init__(
        self, metadata: ExperimentMetadata, auto_save: bool = True
    ) -> None:
        """
        Initialize the artifact manager.

        Args:
            metadata: Experiment metadata to manage
            auto_save: Whether to automatically save metadata changes
        """
        self.metadata = metadata
        self.auto_save = auto_save

    def add_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        file_path: str,
        description: str = "",
        tags: list[str] | None = None,
        save_callback: Callable[[], None] | None = None,
    ) -> None:
        """
        Associate an artifact with the experiment.

        Args:
            artifact_id: Unique identifier for the artifact
            artifact_type: Type of artifact (model, metrics, visualiz., etc.)
            file_path: Path to the artifact file
            description: Description of the artifact
            tags: Tags for categorizing the artifact
            save_callback: Callback to save metadata
        """
        if artifact_id not in self.metadata.artifact_ids:
            self.metadata.artifact_ids.append(artifact_id)

        # Add to specific lists based on type
        if artifact_type == "checkpoint":
            if file_path not in self.metadata.checkpoint_paths:
                self.metadata.checkpoint_paths.append(file_path)
        elif artifact_type == "metrics":
            if file_path not in self.metadata.metric_files:
                self.metadata.metric_files.append(file_path)
        elif artifact_type == "visualization":
            if file_path not in self.metadata.visualization_files:
                self.metadata.visualization_files.append(file_path)

        if self.auto_save and save_callback:
            save_callback()

        logger.debug(
            f"Added artifact {artifact_id} to experiment "
            f"{self.metadata.experiment_id}"
        )

    def get_artifacts_by_type(self, artifact_type: str) -> list[str]:
        """
        Get artifact paths by type.

        Args:
            artifact_type: Type of artifacts to retrieve

        Returns:
            List of artifact file paths
        """
        if artifact_type == "checkpoint":
            return self.metadata.checkpoint_paths
        elif artifact_type == "metrics":
            return self.metadata.metric_files
        elif artifact_type == "visualization":
            return self.metadata.visualization_files
        else:
            return []

    def get_experiment_summary(self) -> dict[str, Any]:
        """Get a summary of the experiment."""
        return {
            "experiment_id": self.metadata.experiment_id,
            "experiment_name": self.metadata.experiment_name,
            "status": self.metadata.status,
            "description": self.metadata.description,
            "tags": self.metadata.tags,
            "created_at": self.metadata.created_at,
            "started_at": self.metadata.started_at,
            "completed_at": self.metadata.completed_at,
            "total_epochs": self.metadata.total_epochs,
            "current_epoch": self.metadata.current_epoch,
            "best_metrics": self.metadata.best_metrics,
            "training_time_seconds": self.metadata.training_time_seconds,
            "artifact_count": len(self.metadata.artifact_ids),
            "config_hash": self.metadata.config_hash,
        }
