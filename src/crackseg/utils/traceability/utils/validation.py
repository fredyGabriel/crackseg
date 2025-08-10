"""Traceability validation helpers extracted from lineage manager."""

from __future__ import annotations

from ..enums import ArtifactType
from ..models import ArtifactEntity


def is_valid_relationship_type(rel_type: str) -> bool:
    valid_types = {
        "derived_from",
        "influenced_by",
        "depends_on",
        "evolves_to",
        "replaces",
        "complements",
    }
    return rel_type in valid_types


def validate_relationship(
    source: ArtifactEntity, target: ArtifactEntity, relationship_type: str
) -> None:
    """Raise ValueError if relationship is invalid."""
    if not is_valid_relationship_type(relationship_type):
        raise ValueError(f"Invalid relationship type: {relationship_type}")

    if relationship_type == "derived_from":
        if (
            source.artifact_type == ArtifactType.MODEL
            and target.artifact_type == ArtifactType.DATASET
        ):
            raise ValueError("Model cannot be derived from dataset")
