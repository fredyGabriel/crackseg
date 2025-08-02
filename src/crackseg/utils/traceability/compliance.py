"""
Compliance and export models for traceability system.

This module defines compliance records and export functionality
for the traceability system.
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
from .enums import ComplianceLevel


class ComplianceRecord(BaseModel):
    """Compliance and audit record."""

    record_id: str = Field(..., description="Unique compliance record ID")
    artifact_id: str = Field(..., description="Associated artifact ID")

    # Compliance information
    compliance_level: ComplianceLevel = Field(
        ..., description="Compliance level"
    )
    audit_timestamp: datetime = Field(
        default_factory=datetime.now, description="Audit timestamp"
    )
    auditor: str = Field(..., description="Auditor name")

    # Audit results
    passed: bool = Field(..., description="Whether compliance check passed")
    findings: list[str] = Field(
        default_factory=list, description="Audit findings"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Audit recommendations"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "record_id": self.record_id,
            "artifact_id": self.artifact_id,
            "compliance_level": self.compliance_level.value,
            "audit_timestamp": self.audit_timestamp.isoformat(),
            "auditor": self.auditor,
            "passed": self.passed,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class TraceabilityExport(BaseModel):
    """Export model for traceability data."""

    # Export metadata
    export_id: str = Field(..., description="Unique export identifier")
    export_timestamp: datetime = Field(
        default_factory=datetime.now, description="Export timestamp"
    )
    export_format: str = Field(default="json", description="Export format")

    # Data
    artifacts: list[ArtifactEntity] = Field(
        default_factory=list, description="Exported artifacts"
    )
    experiments: list[ExperimentEntity] = Field(
        default_factory=list, description="Exported experiments"
    )
    versions: list[VersionEntity] = Field(
        default_factory=list, description="Exported versions"
    )
    lineage: list[LineageEntity] = Field(
        default_factory=list, description="Exported lineage"
    )
    compliance_records: list[ComplianceRecord] = Field(
        default_factory=list, description="Exported compliance records"
    )

    # Summary
    total_artifacts: int = Field(..., description="Total artifacts exported")
    total_experiments: int = Field(
        ..., description="Total experiments exported"
    )
    total_versions: int = Field(..., description="Total versions exported")
    total_lineage: int = Field(
        ..., description="Total lineage relationships exported"
    )
    total_compliance_records: int = Field(
        ..., description="Total compliance records exported"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "export_id": self.export_id,
            "export_timestamp": self.export_timestamp.isoformat(),
            "export_format": self.export_format,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "experiments": [
                experiment.to_dict() for experiment in self.experiments
            ],
            "versions": [version.to_dict() for version in self.versions],
            "lineage": [lineage.to_dict() for lineage in self.lineage],
            "compliance_records": [
                record.to_dict() for record in self.compliance_records
            ],
            "total_artifacts": self.total_artifacts,
            "total_experiments": self.total_experiments,
            "total_versions": self.total_versions,
            "total_lineage": self.total_lineage,
            "total_compliance_records": self.total_compliance_records,
        }
