"""
Query and result models for traceability system.

This module defines the query interface and result models for
traceability searches and exports.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .entities import (
    ArtifactEntity,
    ExperimentEntity,
    LineageEntity,
    VersionEntity,
)
from .enums import ArtifactType, ComplianceLevel, VerificationStatus


class TraceabilityQuery(BaseModel):
    """Query model for traceability searches."""

    # Basic filters
    artifact_types: list[ArtifactType] = Field(
        default_factory=list, description="Filter by artifact types"
    )
    experiment_ids: list[str] = Field(
        default_factory=list, description="Filter by experiment IDs"
    )
    tags: list[str] = Field(default_factory=list, description="Filter by tags")
    owners: list[str] = Field(
        default_factory=list, description="Filter by owners"
    )

    # Date filters
    created_after: datetime | None = Field(
        None, description="Created after date"
    )
    created_before: datetime | None = Field(
        None, description="Created before date"
    )

    # Status filters
    verification_status: list[VerificationStatus] = Field(
        default_factory=list, description="Filter by verification status"
    )
    compliance_level: list[ComplianceLevel] = Field(
        default_factory=list, description="Filter by compliance level"
    )

    # Lineage filters
    include_lineage: bool = Field(
        default=True, description="Include lineage information"
    )
    max_lineage_depth: int = Field(
        default=5, description="Maximum lineage depth"
    )

    # Pagination
    limit: int = Field(default=100, description="Maximum results")
    offset: int = Field(default=0, description="Result offset")

    # Sorting
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(
        default="desc", description="Sort order (asc/desc)"
    )


class TraceabilityResult(BaseModel):
    """Result model for traceability queries."""

    # Query information
    query: TraceabilityQuery = Field(..., description="Original query")
    total_count: int = Field(..., description="Total matching results")

    # Results
    artifacts: list[ArtifactEntity] = Field(
        default_factory=list, description="Matching artifacts"
    )
    experiments: list[ExperimentEntity] = Field(
        default_factory=list, description="Matching experiments"
    )
    versions: list[VersionEntity] = Field(
        default_factory=list, description="Matching versions"
    )
    lineage: list[LineageEntity] = Field(
        default_factory=list, description="Matching lineage relationships"
    )

    # Metadata
    query_time_ms: float = Field(
        ..., description="Query execution time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Result timestamp"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query.model_dump(),
            "total_count": self.total_count,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "experiments": [
                experiment.to_dict() for experiment in self.experiments
            ],
            "versions": [version.to_dict() for version in self.versions],
            "lineage": [lineage.to_dict() for lineage in self.lineage],
            "query_time_ms": self.query_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }
