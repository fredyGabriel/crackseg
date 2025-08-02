"""
Lineage management system for CrackSeg project.

This module provides comprehensive lineage management capabilities including
lineage creation, validation, analysis, and visualization for artifact
relationships in the traceability system.
"""

import logging
from datetime import datetime
from typing import Any

from .enums import ArtifactType
from .models import (
    ArtifactEntity,
    LineageEntity,
)
from .storage import TraceabilityStorage

logger = logging.getLogger(__name__)


class LineageManager:
    """Advanced lineage management for artifact relationships."""

    def __init__(self, storage: TraceabilityStorage) -> None:
        """Initialize lineage manager with storage.

        Args:
            storage: Traceability storage instance
        """
        self.storage = storage

    def create_lineage(
        self,
        source_artifact_id: str,
        target_artifact_id: str,
        relationship_type: str,
        relationship_description: str = "",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> LineageEntity:
        """Create a new lineage relationship.

        Args:
            source_artifact_id: Source artifact ID
            target_artifact_id: Target artifact ID
            relationship_type: Type of relationship
            relationship_description: Description of relationship
            confidence: Confidence level (0.0-1.0)
            metadata: Additional metadata

        Returns:
            Created lineage entity

        Raises:
            ValueError: If artifacts don't exist or relationship is invalid
        """
        # Validate artifacts exist
        source_artifact = self._get_artifact(source_artifact_id)
        target_artifact = self._get_artifact(target_artifact_id)

        if not source_artifact or not target_artifact:
            raise ValueError("Source or target artifact not found")

        # Validate relationship
        self._validate_relationship(
            source_artifact, target_artifact, relationship_type
        )

        # Check for circular dependencies
        if self._would_create_cycle(source_artifact_id, target_artifact_id):
            raise ValueError("Lineage would create circular dependency")

        # Create lineage entity
        lineage = LineageEntity(
            lineage_id=f"lineage-{source_artifact_id}-{target_artifact_id}",
            source_artifact_id=source_artifact_id,
            target_artifact_id=target_artifact_id,
            relationship_type=relationship_type,
            relationship_description=relationship_description,
            confidence=confidence,
            metadata=metadata or {},
        )

        # Save to storage
        if self.storage.save_lineage(lineage):
            logger.info(
                f"Created lineage: {source_artifact_id} -> "
                f"{target_artifact_id}"
            )
            return lineage
        else:
            raise RuntimeError("Failed to save lineage relationship")

    def update_lineage(
        self,
        lineage_id: str,
        relationship_description: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageEntity | None:
        """Update an existing lineage relationship.

        Args:
            lineage_id: Lineage ID to update
            relationship_description: New description
            confidence: New confidence level
            metadata: New metadata

        Returns:
            Updated lineage entity or None if not found
        """
        # Load existing lineage
        lineage_data = self.storage._load_lineage()
        lineage_dict = None

        for lineage in lineage_data:
            if lineage.get("lineage_id") == lineage_id:
                lineage_dict = lineage
                break

        if not lineage_dict:
            return None

        # Update fields
        if relationship_description is not None:
            lineage_dict["relationship_description"] = relationship_description
        if confidence is not None:
            lineage_dict["confidence"] = confidence
        if metadata is not None:
            lineage_dict["metadata"] = metadata

        lineage_dict["updated_at"] = datetime.now().isoformat()

        # Save updated lineage
        updated_lineage = LineageEntity.model_validate(lineage_dict)
        if self.storage.save_lineage(updated_lineage):
            logger.info(f"Updated lineage: {lineage_id}")
            return updated_lineage
        else:
            raise RuntimeError("Failed to update lineage relationship")

    def delete_lineage(self, lineage_id: str) -> bool:
        """Delete a lineage relationship.

        Args:
            lineage_id: Lineage ID to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        lineage_data = self.storage._load_lineage()
        updated_data = [
            lineage
            for lineage in lineage_data
            if lineage.get("lineage_id") != lineage_id
        ]

        if len(updated_data) == len(lineage_data):
            return False  # Lineage not found

        self.storage._save_lineage(updated_data)
        logger.info(f"Deleted lineage: {lineage_id}")
        return True

    def get_lineage_path(
        self, source_id: str, target_id: str, max_depth: int = 10
    ) -> list[LineageEntity]:
        """Find lineage path between two artifacts.

        Args:
            source_id: Source artifact ID
            target_id: Target artifact ID
            max_depth: Maximum search depth

        Returns:
            List of lineage entities forming the path
        """
        lineage_data = self.storage._load_lineage()
        visited = set()
        path = []

        def _find_path(current_id: str, target_id: str, depth: int) -> bool:
            if depth > max_depth or current_id in visited:
                return False

            visited.add(current_id)

            for lineage in lineage_data:
                if lineage.get("source_artifact_id") == current_id:
                    if lineage.get("target_artifact_id") == target_id:
                        path.append(LineageEntity.model_validate(lineage))
                        return True

                    if _find_path(
                        lineage.get("target_artifact_id", ""),
                        target_id,
                        depth + 1,
                    ):
                        path.append(LineageEntity.model_validate(lineage))
                        return True

            return False

        if _find_path(source_id, target_id, 0):
            return list(reversed(path))
        return []

    def get_lineage_tree(
        self, artifact_id: str, max_depth: int = 5
    ) -> dict[str, Any]:
        """Get lineage tree for an artifact.

        Args:
            artifact_id: Root artifact ID
            max_depth: Maximum tree depth

        Returns:
            Tree structure with lineage information
        """
        lineage_data = self.storage._load_lineage()
        tree = {
            "artifact_id": artifact_id,
            "children": [],
            "parents": [],
            "depth": 0,
        }

        def _build_tree(
            current_id: str, depth: int, direction: str
        ) -> list[dict[str, Any]]:
            if depth > max_depth:
                return []

            nodes = []
            for lineage in lineage_data:
                if (
                    direction == "children"
                    and lineage.get("source_artifact_id") == current_id
                ):
                    nodes.append(
                        {
                            "lineage": LineageEntity.model_validate(lineage),
                            "artifact_id": lineage.get("target_artifact_id"),
                            "children": _build_tree(
                                lineage.get("target_artifact_id", ""),
                                depth + 1,
                                "children",
                            ),
                            "parents": [],
                            "depth": depth + 1,
                        }
                    )
                elif (
                    direction == "parents"
                    and lineage.get("target_artifact_id") == current_id
                ):
                    nodes.append(
                        {
                            "lineage": LineageEntity.model_validate(lineage),
                            "artifact_id": lineage.get("source_artifact_id"),
                            "children": [],
                            "parents": _build_tree(
                                lineage.get("source_artifact_id", ""),
                                depth + 1,
                                "parents",
                            ),
                            "depth": depth + 1,
                        }
                    )

            return nodes

        tree["children"] = _build_tree(artifact_id, 0, "children")
        tree["parents"] = _build_tree(artifact_id, 0, "parents")

        return tree

    def analyze_lineage_impact(
        self, artifact_id: str, max_depth: int = 5
    ) -> dict[str, Any]:
        """Analyze the impact of an artifact through its lineage.

        Args:
            artifact_id: Artifact ID to analyze
            max_depth: Maximum analysis depth

        Returns:
            Impact analysis results
        """
        tree = self.get_lineage_tree(artifact_id, max_depth)
        lineage_data = self.storage._load_lineage()

        # Count relationships
        total_relationships = len(lineage_data)
        direct_relationships = sum(
            1
            for lineage in lineage_data
            if lineage.get("source_artifact_id") == artifact_id
            or lineage.get("target_artifact_id") == artifact_id
        )

        # Analyze relationship types
        relationship_types = {}
        for lineage in lineage_data:
            if (
                lineage.get("source_artifact_id") == artifact_id
                or lineage.get("target_artifact_id") == artifact_id
            ):
                rel_type = lineage.get("relationship_type", "unknown")
                relationship_types[rel_type] = (
                    relationship_types.get(rel_type, 0) + 1
                )

        # Calculate confidence statistics
        confidences = [
            lineage.get("confidence", 1.0)
            for lineage in lineage_data
            if (
                lineage.get("source_artifact_id") == artifact_id
                or lineage.get("target_artifact_id") == artifact_id
            )
        ]

        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return {
            "artifact_id": artifact_id,
            "total_relationships": total_relationships,
            "direct_relationships": direct_relationships,
            "relationship_types": relationship_types,
            "average_confidence": avg_confidence,
            "tree_depth": max_depth,
            "tree_size": len(tree["children"]) + len(tree["parents"]),
        }

    def validate_lineage_integrity(self) -> dict[str, Any]:
        """Validate integrity of all lineage relationships.

        Returns:
            Validation results with issues found
        """
        lineage_data = self.storage._load_lineage()
        artifacts_data = self.storage._load_artifacts()

        issues = []
        valid_count = 0

        # Create lookup sets for quick validation
        artifact_ids = {a.get("artifact_id") for a in artifacts_data}

        for lineage in lineage_data:
            source_id = lineage.get("source_artifact_id")
            target_id = lineage.get("target_artifact_id")

            # Check if artifacts exist
            if source_id not in artifact_ids:
                issues.append(
                    {
                        "type": "missing_source_artifact",
                        "lineage_id": lineage.get("lineage_id"),
                        "artifact_id": source_id,
                    }
                )

            if target_id not in artifact_ids:
                issues.append(
                    {
                        "type": "missing_target_artifact",
                        "lineage_id": lineage.get("lineage_id"),
                        "artifact_id": target_id,
                    }
                )

            # Check for self-references
            if source_id == target_id:
                issues.append(
                    {
                        "type": "self_reference",
                        "lineage_id": lineage.get("lineage_id"),
                        "artifact_id": source_id,
                    }
                )

            # Check for duplicate relationships
            duplicate_count = sum(
                1
                for other in lineage_data
                if (
                    other.get("source_artifact_id") == source_id
                    and other.get("target_artifact_id") == target_id
                )
            )
            if duplicate_count > 1:
                issues.append(
                    {
                        "type": "duplicate_relationship",
                        "source_id": source_id,
                        "target_id": target_id,
                        "count": duplicate_count,
                    }
                )

            if not issues:  # No issues found for this lineage
                valid_count += 1

        return {
            "total_lineage": len(lineage_data),
            "valid_lineage": valid_count,
            "issues": issues,
            "integrity_score": (
                valid_count / len(lineage_data) if lineage_data else 1.0
            ),
        }

    def _get_artifact(self, artifact_id: str) -> ArtifactEntity | None:
        """Get artifact by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact entity or None if not found
        """
        artifacts_data = self.storage._load_artifacts()
        for artifact in artifacts_data:
            if artifact.get("artifact_id") == artifact_id:
                return ArtifactEntity.model_validate(artifact)
        return None

    def _validate_relationship(
        self,
        source: ArtifactEntity,
        target: ArtifactEntity,
        relationship_type: str,
    ) -> None:
        """Validate lineage relationship.

        Args:
            source: Source artifact
            target: Target artifact
            relationship_type: Type of relationship

        Raises:
            ValueError: If relationship is invalid
        """
        # Validate relationship types
        valid_types = {
            "derived_from",
            "influenced_by",
            "depends_on",
            "evolves_to",
            "replaces",
            "complements",
        }

        if relationship_type not in valid_types:
            raise ValueError(f"Invalid relationship type: {relationship_type}")

        # Validate artifact type compatibility
        if relationship_type == "derived_from":
            if (
                source.artifact_type == ArtifactType.MODEL
                and target.artifact_type == ArtifactType.DATASET
            ):
                raise ValueError("Model cannot be derived from dataset")

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Check if adding lineage would create a cycle.

        Args:
            source_id: Source artifact ID
            target_id: Target artifact ID

        Returns:
            True if cycle would be created
        """
        # Check if target can reach source
        path = self.get_lineage_path(target_id, source_id, max_depth=10)
        return len(path) > 0
