"""
ExperimentTracker for comprehensive experiment metadata tracking.

This module provides the ExperimentTracker class for tracking experiment
metadata, artifact associations, and lifecycle management.
"""

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import psutil
import torch
from omegaconf import DictConfig

from crackseg.utils.experiment.metadata import ExperimentMetadata
from crackseg.utils.experiment.tracker import (
    ExperimentArtifactManager,
    ExperimentConfigManager,
    ExperimentGitManager,
    ExperimentLifecycleManager,
)
from crackseg.utils.logging.base import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    Comprehensive experiment metadata tracker.

    Tracks experiment metadata, artifact associations, and lifecycle
    management. Provides detailed experiment information for
    reproducibility and analysis.
    """

    def __init__(
        self,
        experiment_dir: Path,
        experiment_id: str,
        experiment_name: str,
        config: DictConfig | None = None,
        auto_save: bool = True,
    ) -> None:
        """
        Initialize the experiment tracker.

        Args:
            experiment_dir: Directory for experiment data
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable experiment name
            config: Experiment configuration (optional)
            auto_save: Whether to automatically save metadata changes
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.config = config
        self.auto_save = auto_save

        # Create experiment directory if it doesn't exist
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        self.metadata = self._load_or_create_metadata()

        # Initialize managers
        self.lifecycle_manager = ExperimentLifecycleManager(
            self.metadata, auto_save
        )
        self.artifact_manager = ExperimentArtifactManager(
            self.metadata, auto_save
        )

        logger.info(
            f"ExperimentTracker initialized for experiment: {experiment_id}"
        )

    @property
    def metadata_file(self) -> Path:
        """Get the metadata file path."""
        return self.experiment_dir / "experiment_tracker.json"

    def _load_or_create_metadata(self) -> ExperimentMetadata:
        """Load existing metadata or create new metadata."""
        metadata_file = self.metadata_file

        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    metadata_dict = json.load(f)
                metadata = ExperimentMetadata(**metadata_dict)
                logger.info(
                    f"Loaded existing metadata for {self.experiment_id}"
                )
                return metadata
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    f"Failed to load metadata from {metadata_file}: {e}"
                )

        # Create new metadata
        metadata = self._create_initial_metadata()
        if self.auto_save:
            self._save_metadata()
        logger.info(f"Created new metadata for {self.experiment_id}")
        return metadata

    def _create_initial_metadata(self) -> ExperimentMetadata:
        """Create initial experiment metadata."""
        metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
        )

        # Collect system metadata
        metadata.hostname = platform.node()
        metadata.username = os.getenv("USERNAME", "unknown")
        metadata.platform = platform.platform()
        metadata.python_version = sys.version
        metadata.memory_gb = psutil.virtual_memory().total / (1024**3)

        # Collect PyTorch metadata
        metadata.pytorch_version = torch.__version__
        metadata.cuda_available = torch.cuda.is_available()
        if metadata.cuda_available:
            metadata.cuda_version = torch.version.cuda
            metadata.gpu_info = {
                "count": torch.cuda.device_count(),
                "current": torch.cuda.current_device(),
                "name": torch.cuda.get_device_name(0),
            }

        # Collect Git metadata
        ExperimentGitManager.collect_git_metadata(metadata)

        # Configuration metadata
        if self.config is not None:
            metadata.config_hash = (
                ExperimentConfigManager.calculate_config_hash(self.config)
            )
            metadata.config_summary = (
                ExperimentConfigManager.extract_config_summary(self.config)
            )

        return metadata

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    # Lifecycle management methods
    def start_experiment(self) -> None:
        """Mark experiment as started."""
        self.lifecycle_manager.start_experiment(self._save_metadata)

    def complete_experiment(self) -> None:
        """Mark experiment as completed."""
        self.lifecycle_manager.complete_experiment(self._save_metadata)

    def fail_experiment(self, error_message: str = "") -> None:
        """Mark experiment as failed."""
        self.lifecycle_manager.fail_experiment(
            error_message, self._save_metadata
        )

    def abort_experiment(self) -> None:
        """Mark experiment as aborted."""
        self.lifecycle_manager.abort_experiment(self._save_metadata)

    def update_training_progress(
        self,
        current_epoch: int,
        total_epochs: int,
        best_metrics: dict[str, float] | None = None,
        training_time_seconds: float | None = None,
    ) -> None:
        """
        Update training progress metadata.

        Args:
            current_epoch: Current training epoch
            total_epochs: Total number of epochs
            best_metrics: Best metrics achieved so far
            training_time_seconds: Total training time in seconds
        """
        self.lifecycle_manager.update_training_progress(
            current_epoch,
            total_epochs,
            best_metrics,
            training_time_seconds,
            self._save_metadata,
        )

    # Artifact management methods
    def add_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        file_path: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """
        Associate an artifact with the experiment.

        Args:
            artifact_id: Unique identifier for the artifact
            artifact_type: Type of artifact (checkpoint, metrics, etc.)
            file_path: Path to the artifact file
            description: Description of the artifact
            tags: Tags for categorizing the artifact
        """
        self.artifact_manager.add_artifact(
            artifact_id,
            artifact_type,
            file_path,
            description,
            tags,
            self._save_metadata,
        )

    def get_artifacts_by_type(self, artifact_type: str) -> list[str]:
        """
        Get artifact paths by type.

        Args:
            artifact_type: Type of artifacts to retrieve

        Returns:
            List of artifact file paths
        """
        return self.artifact_manager.get_artifacts_by_type(artifact_type)

    def get_experiment_summary(self) -> dict[str, Any]:
        """Get a summary of the experiment."""
        return self.artifact_manager.get_experiment_summary()

    # Metadata access methods
    def get_metadata(self) -> ExperimentMetadata:
        """Get the experiment metadata."""
        return self.metadata

    def get_metadata_dict(self) -> dict[str, Any]:
        """Get metadata as dictionary."""
        return self.metadata.to_dict()

    def update_description(self, description: str) -> None:
        """Update experiment description."""
        self.metadata.description = description
        self.metadata.update_timestamp()
        if self.auto_save:
            self._save_metadata()

    def add_tags(self, tags: list[str]) -> None:
        """Add tags to the experiment."""
        for tag in tags:
            if tag not in self.metadata.tags:
                self.metadata.tags.append(tag)
        self.metadata.update_timestamp()
        if self.auto_save:
            self._save_metadata()

    def remove_tags(self, tags: list[str]) -> None:
        """Remove tags from the experiment."""
        for tag in tags:
            if tag in self.metadata.tags:
                self.metadata.tags.remove(tag)
        self.metadata.update_timestamp()
        if self.auto_save:
            self._save_metadata()
