"""
Metadata and access control system for CrackSeg project.

This module provides comprehensive metadata management and access control
capabilities for the traceability system, including metadata validation,
access control, and metadata enrichment.
"""

import logging
from datetime import datetime
from typing import Any

from .models import (
    ArtifactEntity,
    ExperimentEntity,
)
from .storage import TraceabilityStorage

logger = logging.getLogger(__name__)


class MetadataManager:
    """Advanced metadata management for traceability system."""

    def __init__(self, storage: TraceabilityStorage) -> None:
        """Initialize metadata manager with storage.

        Args:
            storage: Traceability storage instance
        """
        self.storage = storage

    def enrich_artifact_metadata(
        self, artifact: ArtifactEntity, metadata: dict[str, Any]
    ) -> ArtifactEntity:
        """Enrich artifact with additional metadata.

        Args:
            artifact: Artifact entity to enrich
            metadata: Additional metadata to add

        Returns:
            Enriched artifact entity
        """
        # Validate metadata
        validated_metadata = self._validate_metadata(metadata)

        # Merge with existing metadata
        enriched_metadata = {**artifact.metadata, **validated_metadata}

        # Create enriched artifact
        enriched_artifact = artifact.model_copy(
            update={
                "metadata": enriched_metadata,
                "updated_at": datetime.now(),
            }
        )

        # Save to storage
        if self.storage.save_artifact(enriched_artifact):
            logger.info(
                f"Enriched metadata for artifact: {artifact.artifact_id}"
            )
            return enriched_artifact
        else:
            raise RuntimeError("Failed to save enriched artifact")

    def enrich_experiment_metadata(
        self, experiment: ExperimentEntity, metadata: dict[str, Any]
    ) -> ExperimentEntity:
        """Enrich experiment with additional metadata.

        Args:
            experiment: Experiment entity to enrich
            metadata: Additional metadata to add

        Returns:
            Enriched experiment entity
        """
        # Validate metadata
        validated_metadata = self._validate_metadata(metadata)

        # Merge with existing metadata
        enriched_metadata = {**experiment.metadata, **validated_metadata}

        # Create enriched experiment
        enriched_experiment = experiment.model_copy(
            update={
                "metadata": enriched_metadata,
                "updated_at": datetime.now(),
            }
        )

        # Save to storage
        if self.storage.save_experiment(enriched_experiment):
            logger.info(
                f"Enriched metadata for experiment: {experiment.experiment_id}"
            )
            return enriched_experiment
        else:
            raise RuntimeError("Failed to save enriched experiment")

    def search_by_metadata(
        self,
        metadata_key: str,
        metadata_value: Any,
        entity_type: str = "artifact",
    ) -> list[dict[str, Any]]:
        """Search entities by metadata key-value pairs.

        Args:
            metadata_key: Metadata key to search for
            metadata_value: Metadata value to match
            entity_type: Type of entity to search
            ("artifact", "experiment", "lineage")

        Returns:
            List of matching entities
        """
        if entity_type == "artifact":
            return self._search_artifacts_by_metadata(
                metadata_key, metadata_value
            )
        elif entity_type == "experiment":
            return self._search_experiments_by_metadata(
                metadata_key, metadata_value
            )
        elif entity_type == "lineage":
            return self._search_lineage_by_metadata(
                metadata_key, metadata_value
            )
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

    def get_metadata_statistics(self) -> dict[str, Any]:
        """Get metadata usage statistics.

        Returns:
            Dictionary with metadata statistics
        """
        artifacts_data = self.storage._load_artifacts()
        experiments_data = self.storage._load_experiments()
        lineage_data = self.storage._load_lineage()

        # Analyze metadata usage
        artifact_metadata_keys = set()
        experiment_metadata_keys = set()
        lineage_metadata_keys = set()

        for artifact in artifacts_data:
            if "metadata" in artifact:
                artifact_metadata_keys.update(artifact["metadata"].keys())

        for experiment in experiments_data:
            if "metadata" in experiment:
                experiment_metadata_keys.update(experiment["metadata"].keys())

        for lineage in lineage_data:
            if "metadata" in lineage:
                lineage_metadata_keys.update(lineage["metadata"].keys())

        return {
            "total_artifacts": len(artifacts_data),
            "total_experiments": len(experiments_data),
            "total_lineage": len(lineage_data),
            "artifact_metadata_keys": list(artifact_metadata_keys),
            "experiment_metadata_keys": list(experiment_metadata_keys),
            "lineage_metadata_keys": list(lineage_metadata_keys),
            "total_unique_metadata_keys": len(
                artifact_metadata_keys
                | experiment_metadata_keys
                | lineage_metadata_keys
            ),
        }

    def validate_metadata_completeness(self) -> dict[str, Any]:
        """Validate metadata completeness across all entities.

        Returns:
            Validation results with completeness metrics
        """
        artifacts_data = self.storage._load_artifacts()
        experiments_data = self.storage._load_experiments()

        # Define required metadata fields for different entity types
        required_artifact_metadata = {
            "accuracy": "Model accuracy metric",
            "model_type": "Type of model architecture",
            "training_dataset": "Dataset used for training",
        }

        required_experiment_metadata = {
            "objective": "Experiment objective",
            "hypothesis": "Research hypothesis",
            "methodology": "Experimental methodology",
        }

        # Check artifact metadata completeness
        artifact_completeness = {}
        for artifact in artifacts_data:
            artifact_id = artifact.get("artifact_id", "unknown")
            metadata = artifact.get("metadata", {})

            missing_fields = []
            for field, description in required_artifact_metadata.items():
                if field not in metadata:
                    missing_fields.append(f"{field} ({description})")

            artifact_completeness[artifact_id] = {
                "completeness_score": 1.0
                - (len(missing_fields) / len(required_artifact_metadata)),
                "missing_fields": missing_fields,
                "total_required": len(required_artifact_metadata),
                "present_fields": len(metadata),
            }

        # Check experiment metadata completeness
        experiment_completeness = {}
        for experiment in experiments_data:
            experiment_id = experiment.get("experiment_id", "unknown")
            metadata = experiment.get("metadata", {})

            missing_fields = []
            for field, description in required_experiment_metadata.items():
                if field not in metadata:
                    missing_fields.append(f"{field} ({description})")

            experiment_completeness[experiment_id] = {
                "completeness_score": 1.0
                - (len(missing_fields) / len(required_experiment_metadata)),
                "missing_fields": missing_fields,
                "total_required": len(required_experiment_metadata),
                "present_fields": len(metadata),
            }

        return {
            "artifact_completeness": artifact_completeness,
            "experiment_completeness": experiment_completeness,
            "overall_artifact_completeness": (
                sum(
                    info["completeness_score"]
                    for info in artifact_completeness.values()
                )
                / len(artifact_completeness)
                if artifact_completeness
                else 0.0
            ),
            "overall_experiment_completeness": (
                sum(
                    info["completeness_score"]
                    for info in experiment_completeness.values()
                )
                / len(experiment_completeness)
                if experiment_completeness
                else 0.0
            ),
        }

    def _validate_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata structure and content.

        Args:
            metadata: Metadata dictionary to validate

        Returns:
            Validated metadata dictionary

        Raises:
            ValueError: If metadata is invalid
        """
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        validated_metadata = {}

        for key, value in metadata.items():
            # Validate key
            if not isinstance(key, str):
                raise ValueError(
                    f"Metadata key must be string, got {type(key)}"
                )

            if not key or key.strip() == "":
                raise ValueError("Metadata key cannot be empty")

            # Validate value types
            if isinstance(value, str | int | float | bool | list | dict):
                validated_metadata[key] = value
            else:
                # Convert other types to string
                validated_metadata[key] = str(value)

        return validated_metadata

    def _search_artifacts_by_metadata(
        self, metadata_key: str, metadata_value: Any
    ) -> list[dict[str, Any]]:
        """Search artifacts by metadata key-value pair.

        Args:
            metadata_key: Metadata key to search for
            metadata_value: Metadata value to match

        Returns:
            List of matching artifacts
        """
        artifacts_data = self.storage._load_artifacts()
        matching_artifacts = []

        for artifact in artifacts_data:
            metadata = artifact.get("metadata", {})
            if (
                metadata_key in metadata
                and metadata[metadata_key] == metadata_value
            ):
                matching_artifacts.append(artifact)

        return matching_artifacts

    def _search_experiments_by_metadata(
        self, metadata_key: str, metadata_value: Any
    ) -> list[dict[str, Any]]:
        """Search experiments by metadata key-value pair.

        Args:
            metadata_key: Metadata key to search for
            metadata_value: Metadata value to match

        Returns:
            List of matching experiments
        """
        experiments_data = self.storage._load_experiments()
        matching_experiments = []

        for experiment in experiments_data:
            metadata = experiment.get("metadata", {})
            if (
                metadata_key in metadata
                and metadata[metadata_key] == metadata_value
            ):
                matching_experiments.append(experiment)

        return matching_experiments

    def _search_lineage_by_metadata(
        self, metadata_key: str, metadata_value: Any
    ) -> list[dict[str, Any]]:
        """Search lineage by metadata key-value pair.

        Args:
            metadata_key: Metadata key to search for
            metadata_value: Metadata value to match

        Returns:
            List of matching lineage relationships
        """
        lineage_data = self.storage._load_lineage()
        matching_lineage = []

        for lineage in lineage_data:
            metadata = lineage.get("metadata", {})
            if (
                metadata_key in metadata
                and metadata[metadata_key] == metadata_value
            ):
                matching_lineage.append(lineage)

        return matching_lineage
