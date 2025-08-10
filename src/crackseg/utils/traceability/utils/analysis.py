"""Analysis and traversal utilities for lineage relationships.

Extracted from LineageManager to reduce module size and improve modularity.
"""

from __future__ import annotations

from typing import Any

from ..models import LineageEntity


def find_lineage_path(
    lineage_data: list[dict[str, Any]],
    source_id: str,
    target_id: str,
    max_depth: int,
) -> list[LineageEntity]:
    visited: set[str] = set()
    path: list[LineageEntity] = []

    def _find_path(current_id: str, target: str, depth: int) -> bool:
        if depth > max_depth or current_id in visited:
            return False

        visited.add(current_id)

        for lineage in lineage_data:
            if lineage.get("source_artifact_id") == current_id:
                if lineage.get("target_artifact_id") == target:
                    path.append(LineageEntity.model_validate(lineage))
                    return True

                if _find_path(
                    lineage.get("target_artifact_id", ""), target, depth + 1
                ):
                    path.append(LineageEntity.model_validate(lineage))
                    return True

        return False

    if _find_path(source_id, target_id, 0):
        return list(reversed(path))
    return []


def build_lineage_tree(
    lineage_data: list[dict[str, Any]], artifact_id: str, max_depth: int
) -> dict[str, Any]:
    tree: dict[str, Any] = {
        "artifact_id": artifact_id,
        "children": [],
        "parents": [],
        "depth": 0,
    }

    def _build(
        current_id: str, depth: int, direction: str
    ) -> list[dict[str, Any]]:
        if depth > max_depth:
            return []

        nodes: list[dict[str, Any]] = []
        for lineage in lineage_data:
            if (
                direction == "children"
                and lineage.get("source_artifact_id") == current_id
            ):
                nodes.append(
                    {
                        "lineage": LineageEntity.model_validate(lineage),
                        "artifact_id": lineage.get("target_artifact_id"),
                        "children": _build(
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
                        "parents": _build(
                            lineage.get("source_artifact_id", ""),
                            depth + 1,
                            "parents",
                        ),
                        "depth": depth + 1,
                    }
                )

        return nodes

    tree["children"] = _build(artifact_id, 0, "children")
    tree["parents"] = _build(artifact_id, 0, "parents")
    return tree


def analyze_impact(
    tree: dict[str, Any],
    lineage_data: list[dict[str, Any]],
    artifact_id: str,
    max_depth: int,
) -> dict[str, Any]:
    total_relationships = len(lineage_data)
    direct_relationships = sum(
        1
        for lineage in lineage_data
        if lineage.get("source_artifact_id") == artifact_id
        or lineage.get("target_artifact_id") == artifact_id
    )

    relationship_types: dict[str, int] = {}
    for lineage in lineage_data:
        if (
            lineage.get("source_artifact_id") == artifact_id
            or lineage.get("target_artifact_id") == artifact_id
        ):
            rel_type = lineage.get("relationship_type", "unknown")
            relationship_types[rel_type] = (
                relationship_types.get(rel_type, 0) + 1
            )

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


def validate_integrity(
    lineage_data: list[dict[str, Any]],
    artifacts_data: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    valid_count = 0

    artifact_ids = {a.get("artifact_id") for a in artifacts_data}

    for lineage in lineage_data:
        source_id = lineage.get("source_artifact_id")
        target_id = lineage.get("target_artifact_id")

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

        if source_id == target_id:
            issues.append(
                {
                    "type": "self_reference",
                    "lineage_id": lineage.get("lineage_id"),
                    "artifact_id": source_id,
                }
            )

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

        if not issues:
            valid_count += 1

    return {
        "total_lineage": len(lineage_data),
        "valid_lineage": valid_count,
        "issues": issues,
        "integrity_score": (
            valid_count / len(lineage_data) if lineage_data else 1.0
        ),
    }
