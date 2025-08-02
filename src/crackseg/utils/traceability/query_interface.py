"""
Query interface for traceability system.

This module provides the main query interface for searching and retrieving
traceability information with support for complex queries and filtering.
"""

import logging
from datetime import datetime
from typing import Any

from .models import (
    ArtifactEntity,
    ExperimentEntity,
    LineageEntity,
    TraceabilityQuery,
    TraceabilityResult,
)
from .storage import TraceabilityStorage

logger = logging.getLogger(__name__)


class TraceabilityQueryInterface:
    """Main query interface for traceability system."""

    def __init__(self, storage: TraceabilityStorage) -> None:
        """Initialize query interface with storage.

        Args:
            storage: Traceability storage instance
        """
        self.storage = storage

    def search_artifacts(self, query: TraceabilityQuery) -> TraceabilityResult:
        """Search artifacts based on query criteria.

        Args:
            query: Query parameters for artifact search

        Returns:
            TraceabilityResult with matching artifacts and metadata
        """
        start_time = datetime.now()
        artifacts = self._load_artifacts()
        filtered_artifacts = self._apply_artifact_filters(artifacts, query)

        # Apply pagination and sorting
        total_count = len(filtered_artifacts)
        paginated_artifacts = self._apply_pagination(
            filtered_artifacts, query.limit, query.offset
        )
        sorted_artifacts = self._apply_sorting(
            paginated_artifacts, query.sort_by, query.sort_order
        )

        # Load related data if requested
        experiments = []
        versions = []
        lineage = []

        if query.include_lineage:
            artifact_ids = [
                str(a.get("artifact_id"))
                for a in sorted_artifacts
                if a.get("artifact_id") is not None
            ]
            lineage = self._load_related_lineage(
                artifact_ids, query.max_lineage_depth
            )

        # Convert to entity objects
        artifact_entities = [
            ArtifactEntity.model_validate(artifact)
            for artifact in sorted_artifacts
        ]

        query_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return TraceabilityResult(
            query=query,
            total_count=total_count,
            artifacts=artifact_entities,
            experiments=experiments,
            versions=versions,
            lineage=lineage,
            query_time_ms=query_time_ms,
        )

    def search_experiments(
        self, query: TraceabilityQuery
    ) -> TraceabilityResult:
        """Search experiments based on query criteria.

        Args:
            query: Query parameters for experiment search

        Returns:
            TraceabilityResult with matching experiments and metadata
        """
        start_time = datetime.now()
        experiments = self._load_experiments()
        filtered_experiments = self._apply_experiment_filters(
            experiments, query
        )

        # Apply pagination and sorting
        total_count = len(filtered_experiments)
        paginated_experiments = self._apply_pagination(
            filtered_experiments, query.limit, query.offset
        )
        sorted_experiments = self._apply_sorting(
            paginated_experiments, query.sort_by, query.sort_order
        )

        # Convert to entity objects
        experiment_entities = [
            ExperimentEntity.model_validate(experiment)
            for experiment in sorted_experiments
        ]

        query_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return TraceabilityResult(
            query=query,
            total_count=total_count,
            artifacts=[],
            experiments=experiment_entities,
            versions=[],
            lineage=[],
            query_time_ms=query_time_ms,
        )

    def get_artifact_by_id(self, artifact_id: str) -> ArtifactEntity | None:
        """Get artifact by ID.

        Args:
            artifact_id: Unique artifact identifier

        Returns:
            ArtifactEntity if found, None otherwise
        """
        artifacts = self._load_artifacts()
        for artifact in artifacts:
            if artifact.get("artifact_id") == artifact_id:
                return ArtifactEntity.model_validate(artifact)
        return None

    def get_experiment_by_id(
        self, experiment_id: str
    ) -> ExperimentEntity | None:
        """Get experiment by ID.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            ExperimentEntity if found, None otherwise
        """
        experiments = self._load_experiments()
        for experiment in experiments:
            if experiment.get("experiment_id") == experiment_id:
                return ExperimentEntity.model_validate(experiment)
        return None

    def get_artifact_lineage(
        self, artifact_id: str, max_depth: int = 5
    ) -> list[LineageEntity]:
        """Get lineage relationships for an artifact.

        Args:
            artifact_id: Artifact ID to get lineage for
            max_depth: Maximum lineage depth to retrieve

        Returns:
            List of lineage relationships
        """
        lineage_data = self._load_lineage()
        lineage_entities = []
        processed_lineage_ids = set()

        def _collect_lineage(
            target_id: str, depth: int, visited: set[str]
        ) -> None:
            if depth > max_depth or target_id in visited:
                return

            visited.add(target_id)

            for lineage in lineage_data:
                lineage_id = lineage.get("lineage_id")
                if lineage_id in processed_lineage_ids:
                    continue

                if lineage.get("source_artifact_id") == target_id:
                    lineage_entities.append(
                        LineageEntity.model_validate(lineage)
                    )
                    processed_lineage_ids.add(lineage_id)
                    _collect_lineage(
                        lineage.get("target_artifact_id", ""),
                        depth + 1,
                        visited,
                    )
                elif lineage.get("target_artifact_id") == target_id:
                    lineage_entities.append(
                        LineageEntity.model_validate(lineage)
                    )
                    processed_lineage_ids.add(lineage_id)
                    _collect_lineage(
                        lineage.get("source_artifact_id", ""),
                        depth + 1,
                        visited,
                    )

        _collect_lineage(artifact_id, 0, set())
        return lineage_entities

    def _load_artifacts(self) -> list[dict[str, Any]]:
        """Load artifacts from storage."""
        try:
            return self.storage._load_artifacts()
        except Exception:
            logger.warning("Failed to load artifacts, returning empty list")
            return []

    def _load_experiments(self) -> list[dict[str, Any]]:
        """Load experiments from storage."""
        try:
            return self.storage._load_experiments()
        except Exception:
            logger.warning("Failed to load experiments, returning empty list")
            return []

    def _load_lineage(self) -> list[dict[str, Any]]:
        """Load lineage data from storage."""
        try:
            return self.storage._load_lineage()
        except Exception:
            logger.warning("Failed to load lineage, returning empty list")
            return []

    def _apply_artifact_filters(
        self, artifacts: list[dict[str, Any]], query: TraceabilityQuery
    ) -> list[dict[str, Any]]:
        """Apply filters to artifacts."""
        filtered = artifacts

        # Filter by artifact types
        if query.artifact_types:
            filtered = [
                a
                for a in filtered
                if a.get("artifact_type")
                in [t.value for t in query.artifact_types]
            ]

        # Filter by experiment IDs
        if query.experiment_ids:
            filtered = [
                a
                for a in filtered
                if a.get("experiment_id") in query.experiment_ids
            ]

        # Filter by tags
        if query.tags:
            filtered = [
                a
                for a in filtered
                if any(tag in a.get("tags", []) for tag in query.tags)
            ]

        # Filter by owners
        if query.owners:
            filtered = [a for a in filtered if a.get("owner") in query.owners]

        # Filter by date range
        if query.created_after:
            filtered = [
                a
                for a in filtered
                if datetime.fromisoformat(a.get("created_at", ""))
                >= query.created_after
            ]

        if query.created_before:
            filtered = [
                a
                for a in filtered
                if datetime.fromisoformat(a.get("created_at", ""))
                <= query.created_before
            ]

        # Filter by verification status
        if query.verification_status:
            filtered = [
                a
                for a in filtered
                if a.get("verification_status")
                in [s.value for s in query.verification_status]
            ]

        # Filter by compliance level
        if query.compliance_level:
            filtered = [
                a
                for a in filtered
                if a.get("compliance_level")
                in [level.value for level in query.compliance_level]
            ]

        return filtered

    def _apply_experiment_filters(
        self, experiments: list[dict[str, Any]], query: TraceabilityQuery
    ) -> list[dict[str, Any]]:
        """Apply filters to experiments."""
        filtered = experiments

        # Filter by experiment IDs
        if query.experiment_ids:
            filtered = [
                e
                for e in filtered
                if e.get("experiment_id") in query.experiment_ids
            ]

        # Filter by date range
        if query.created_after:
            filtered = [
                e
                for e in filtered
                if datetime.fromisoformat(e.get("created_at", ""))
                >= query.created_after
            ]

        if query.created_before:
            filtered = [
                e
                for e in filtered
                if datetime.fromisoformat(e.get("created_at", ""))
                <= query.created_before
            ]

        return filtered

    def _apply_pagination(
        self, items: list[dict[str, Any]], limit: int, offset: int
    ) -> list[dict[str, Any]]:
        """Apply pagination to results."""
        return items[offset : offset + limit]

    def _apply_sorting(
        self, items: list[dict[str, Any]], sort_by: str, sort_order: str
    ) -> list[dict[str, Any]]:
        """Apply sorting to results."""
        reverse = sort_order.lower() == "desc"

        def _get_sort_key(item: dict[str, Any]) -> Any:
            value = item.get(sort_by)
            if isinstance(value, str):
                return value.lower()
            return value

        return sorted(items, key=_get_sort_key, reverse=reverse)

    def _load_related_lineage(
        self, artifact_ids: list[str], max_depth: int
    ) -> list[LineageEntity]:
        """Load lineage relationships for given artifact IDs."""
        lineage_data = self._load_lineage()
        lineage_entities = []

        for lineage in lineage_data:
            source_id = lineage.get("source_artifact_id")
            target_id = lineage.get("target_artifact_id")

            if source_id in artifact_ids or target_id in artifact_ids:
                lineage_entities.append(LineageEntity.model_validate(lineage))

        return lineage_entities
