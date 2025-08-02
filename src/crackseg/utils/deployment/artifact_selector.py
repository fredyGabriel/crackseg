"""Artifact selection for deployment.

This module provides intelligent artifact selection capabilities that integrate
with the traceability system to automatically select the most appropriate
artifacts for deployment based on performance, size, format, and target
environment requirements.
"""

import logging
from dataclasses import dataclass
from typing import Any

from ..traceability import (
    ArtifactEntity,
    TraceabilityQuery,
    TraceabilityQueryInterface,
)
from ..traceability.enums import ArtifactType


@dataclass
class SelectionCriteria:
    """Criteria for artifact selection."""

    # Performance requirements
    min_accuracy: float = 0.0
    max_inference_time_ms: float = 1000.0
    max_memory_usage_mb: float = 2048.0

    # Size requirements
    max_model_size_mb: float = 500.0
    preferred_format: str = (
        "pytorch"  # "pytorch", "onnx", "tensorrt", "torchscript"
    )

    # Environment requirements
    target_environment: str = (
        "production"  # "production", "staging", "development"
    )
    deployment_type: str = "container"  # "container", "serverless", "edge"

    # Additional filters
    model_family: str | None = None  # e.g., "swin-unet", "resnet"
    version_constraint: str | None = None  # e.g., ">=1.0.0"
    tags: list[str] | None = None  # e.g., ["optimized", "quantized"]


@dataclass
class SelectionResult:
    """Result of artifact selection."""

    success: bool
    selected_artifacts: list[ArtifactEntity]
    selection_reason: str
    total_candidates: int
    filtered_candidates: int

    # Selection metrics
    selection_time_ms: float = 0.0
    criteria_used: SelectionCriteria | None = None
    error_message: str | None = None


class ArtifactSelector:
    """Intelligent artifact selector for deployment.

    Integrates with the traceability system to automatically select the most
    appropriate artifacts based on performance, size, format, and deployment
    requirements.
    """

    def __init__(self, query_interface: TraceabilityQueryInterface) -> None:
        """Initialize artifact selector.

        Args:
            query_interface: Interface to query traceability data
        """
        self.query_interface = query_interface
        self.logger = logging.getLogger(__name__)

        self.logger.info("ArtifactSelector initialized")

    def select_artifacts(self, criteria: SelectionCriteria) -> SelectionResult:
        """Select artifacts based on criteria.

        Args:
            criteria: Selection criteria

        Returns:
            Selection result with selected artifacts
        """
        import time

        start_time = time.time()
        self.logger.info(f"Selecting artifacts with criteria: {criteria}")

        try:
            # 1. Query available artifacts
            query = TraceabilityQuery(
                artifact_types=[ArtifactType.MODEL],
                created_after=None,
                created_before=None,
            )

            # Add filters based on criteria
            if criteria.model_family:
                query.tags = [criteria.model_family]

            if criteria.tags:
                query.tags.extend(criteria.tags)

            # Search for artifacts
            search_results = self.query_interface.search_artifacts(query)
            total_candidates = search_results.total_count

            if total_candidates == 0:
                return SelectionResult(
                    success=False,
                    selected_artifacts=[],
                    selection_reason="No artifacts found matching criteria",
                    total_candidates=0,
                    filtered_candidates=0,
                    criteria_used=criteria,
                    error_message="No artifacts available",
                )

            # 2. Filter artifacts based on criteria
            filtered_artifacts = self._filter_artifacts(
                search_results.artifacts, criteria
            )
            filtered_count = len(filtered_artifacts)

            if filtered_count == 0:
                return SelectionResult(
                    success=False,
                    selected_artifacts=[],
                    selection_reason="No artifacts passed filtering criteria",
                    total_candidates=total_candidates,
                    filtered_candidates=0,
                    criteria_used=criteria,
                    error_message="All artifacts filtered out",
                )

            # 3. Rank artifacts by suitability
            ranked_artifacts = self._rank_artifacts(
                filtered_artifacts, criteria
            )

            # 4. Select best artifacts
            selected_artifacts = self._select_best_artifacts(
                ranked_artifacts, criteria
            )

            selection_time = (time.time() - start_time) * 1000

            return SelectionResult(
                success=True,
                selected_artifacts=selected_artifacts,
                selection_reason=f"Selected {len(selected_artifacts)} "
                "artifacts from {filtered_count} candidates",
                total_candidates=total_candidates,
                filtered_candidates=filtered_count,
                selection_time_ms=selection_time,
                criteria_used=criteria,
            )

        except Exception as e:
            self.logger.error(f"Artifact selection failed: {e}")
            return SelectionResult(
                success=False,
                selected_artifacts=[],
                selection_reason="Selection failed due to error",
                total_candidates=0,
                filtered_candidates=0,
                criteria_used=criteria,
                error_message=str(e),
            )

    def _filter_artifacts(
        self, artifacts: list[ArtifactEntity], criteria: SelectionCriteria
    ) -> list[ArtifactEntity]:
        """Filter artifacts based on criteria.

        Args:
            artifacts: List of artifacts to filter
            criteria: Selection criteria

        Returns:
            Filtered list of artifacts
        """
        filtered = []

        for artifact in artifacts:
            # Check performance criteria
            if not self._check_performance_criteria(artifact, criteria):
                continue

            # Check size criteria
            if not self._check_size_criteria(artifact, criteria):
                continue

            # Check format criteria
            if not self._check_format_criteria(artifact, criteria):
                continue

            # Check version constraint
            if not self._check_version_criteria(artifact, criteria):
                continue

            filtered.append(artifact)

        return filtered

    def _check_performance_criteria(
        self, artifact: ArtifactEntity, criteria: SelectionCriteria
    ) -> bool:
        """Check if artifact meets performance criteria.

        Args:
            artifact: Artifact to check
            criteria: Selection criteria

        Returns:
            True if artifact meets performance criteria
        """
        metadata = artifact.metadata or {}

        # Check accuracy
        accuracy = metadata.get("accuracy", 0.0)
        if accuracy < criteria.min_accuracy:
            return False

        # Check inference time
        inference_time = metadata.get("inference_time_ms", float("inf"))
        if inference_time > criteria.max_inference_time_ms:
            return False

        # Check memory usage
        memory_usage = metadata.get("memory_usage_mb", float("inf"))
        if memory_usage > criteria.max_memory_usage_mb:
            return False

        return True

    def _check_size_criteria(
        self, artifact: ArtifactEntity, criteria: SelectionCriteria
    ) -> bool:
        """Check if artifact meets size criteria.

        Args:
            artifact: Artifact to check
            criteria: Selection criteria

        Returns:
            True if artifact meets size criteria
        """
        metadata = artifact.metadata or {}
        model_size = metadata.get("model_size_mb", float("inf"))

        return model_size <= criteria.max_model_size_mb

    def _check_format_criteria(
        self, artifact: ArtifactEntity, criteria: SelectionCriteria
    ) -> bool:
        """Check if artifact meets format criteria.

        Args:
            artifact: Artifact to check
            criteria: Selection criteria

        Returns:
            True if artifact meets format criteria
        """
        metadata = artifact.metadata or {}
        model_format = metadata.get("model_format", "pytorch")

        # If no specific format is preferred, accept all
        if not criteria.preferred_format:
            return True

        return model_format == criteria.preferred_format

    def _check_version_criteria(
        self, artifact: ArtifactEntity, criteria: SelectionCriteria
    ) -> bool:
        """Check if artifact meets version criteria.

        Args:
            artifact: Artifact to check
            criteria: Selection criteria

        Returns:
            True if artifact meets version criteria
        """
        if not criteria.version_constraint:
            return True

        # Simple version checking - can be enhanced with proper semver parsing
        metadata = artifact.metadata or {}
        version = metadata.get("version", "0.0.0")

        # For now, just check if version exists
        return version != "0.0.0"

    def _rank_artifacts(
        self, artifacts: list[ArtifactEntity], criteria: SelectionCriteria
    ) -> list[tuple[ArtifactEntity, float]]:
        """Rank artifacts by suitability score.

        Args:
            artifacts: List of artifacts to rank
            criteria: Selection criteria

        Returns:
            List of (artifact, score) tuples sorted by score
        """
        ranked = []

        for artifact in artifacts:
            score = self._calculate_suitability_score(artifact, criteria)
            ranked.append((artifact, score))

        # Sort by score (highest first)
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _calculate_suitability_score(
        self, artifact: ArtifactEntity, criteria: SelectionCriteria
    ) -> float:
        """Calculate suitability score for artifact.

        Args:
            artifact: Artifact to score
            criteria: Selection criteria

        Returns:
            Suitability score (0.0 to 1.0)
        """
        metadata = artifact.metadata or {}
        score = 0.0

        # Performance score (40% weight)
        accuracy = metadata.get("accuracy", 0.0)
        inference_time = metadata.get("inference_time_ms", 1000.0)
        memory_usage = metadata.get("memory_usage_mb", 2048.0)

        perf_score = (
            (accuracy / 1.0) * 0.4
            + (1.0 - min(inference_time / criteria.max_inference_time_ms, 1.0))
            * 0.3
            + (1.0 - min(memory_usage / criteria.max_memory_usage_mb, 1.0))
            * 0.3
        )
        score += perf_score * 0.4

        # Size score (20% weight)
        model_size = metadata.get("model_size_mb", 500.0)
        size_score = 1.0 - min(model_size / criteria.max_model_size_mb, 1.0)
        score += size_score * 0.2

        # Format preference score (20% weight)
        model_format = metadata.get("model_format", "pytorch")
        format_score = (
            1.0 if model_format == criteria.preferred_format else 0.5
        )
        score += format_score * 0.2

        # Recency score (20% weight)
        # Prefer newer artifacts
        recency_score = 0.8  # Placeholder - could use creation date
        score += recency_score * 0.2

        return min(score, 1.0)

    def _select_best_artifacts(
        self,
        ranked_artifacts: list[tuple[ArtifactEntity, float]],
        criteria: SelectionCriteria,
    ) -> list[ArtifactEntity]:
        """Select the best artifacts from ranked list.

        Args:
            ranked_artifacts: Ranked list of (artifact, score) tuples
            criteria: Selection criteria

        Returns:
            List of selected artifacts
        """
        if not ranked_artifacts:
            return []

        # For now, select the top artifact
        # This could be enhanced to select multiple artifacts for different
        # scenarios
        selected = [ranked_artifacts[0][0]]

        self.logger.info(
            f"Selected artifact {selected[0].artifact_id} "
            f"with score {ranked_artifacts[0][1]:.3f}"
        )

        return selected

    def get_artifact_recommendations(
        self, target_environment: str, deployment_type: str
    ) -> dict[str, Any]:
        """Get artifact recommendations for target environment.

        Args:
            target_environment: Target deployment environment
            deployment_type: Type of deployment

        Returns:
            Dictionary with recommendations
        """
        # Define criteria based on environment and deployment type
        if target_environment == "production":
            criteria = SelectionCriteria(
                min_accuracy=0.85,
                max_inference_time_ms=500.0,
                max_memory_usage_mb=1024.0,
                max_model_size_mb=200.0,
                preferred_format="onnx",
                target_environment="production",
                deployment_type=deployment_type,
            )
        elif target_environment == "staging":
            criteria = SelectionCriteria(
                min_accuracy=0.80,
                max_inference_time_ms=1000.0,
                max_memory_usage_mb=2048.0,
                max_model_size_mb=500.0,
                preferred_format="pytorch",
                target_environment="staging",
                deployment_type=deployment_type,
            )
        else:  # development
            criteria = SelectionCriteria(
                min_accuracy=0.70,
                max_inference_time_ms=2000.0,
                max_memory_usage_mb=4096.0,
                max_model_size_mb=1000.0,
                preferred_format="pytorch",
                target_environment="development",
                deployment_type=deployment_type,
            )

        result = self.select_artifacts(criteria)

        return {
            "target_environment": target_environment,
            "deployment_type": deployment_type,
            "criteria": criteria,
            "selection_result": result,
            "recommendations": [
                {
                    "artifact_id": artifact.artifact_id,
                    "reason": "Best match for target environment",
                    "metadata": artifact.metadata,
                }
                for artifact in result.selected_artifacts
            ],
        }
