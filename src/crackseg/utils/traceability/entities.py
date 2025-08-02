"""
Core traceability entities for CrackSeg project.

This module defines the main entity models for artifacts, experiments,
versions, and lineage tracking.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .enums import (
    ArtifactType,
    ComplianceLevel,
    ExperimentStatus,
    VerificationStatus,
)


class ArtifactEntity(BaseModel):
    """Core artifact entity with comprehensive metadata."""

    artifact_id: str = Field(..., description="Unique artifact identifier")
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    file_path: Path = Field(..., description="Path to artifact file")
    file_size: int = Field(..., description="File size in bytes")
    checksum: str = Field(..., description="SHA256 checksum for integrity")

    # Metadata
    name: str = Field(..., description="Human-readable artifact name")
    description: str = Field(default="", description="Artifact description")
    tags: list[str] = Field(
        default_factory=list, description="Categorization tags"
    )

    # Ownership and compliance
    owner: str = Field(..., description="Artifact owner/creator")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )

    # Verification and compliance
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Integrity verification status",
    )
    compliance_level: ComplianceLevel = Field(
        default=ComplianceLevel.BASIC, description="Compliance level"
    )

    # Relationships
    experiment_id: str | None = Field(
        None, description="Associated experiment ID"
    )
    parent_artifact_ids: list[str] = Field(
        default_factory=list, description="Parent artifact dependencies"
    )
    child_artifact_ids: list[str] = Field(
        default_factory=list, description="Child artifacts that depend on this"
    )

    # Versioning
    version: str = Field(default="1.0.0", description="Artifact version")
    version_history: list[str] = Field(
        default_factory=list, description="Previous version IDs"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, v: str) -> str:
        """Validate artifact ID format."""
        if not v or " " in v:
            raise ValueError("Artifact ID cannot be empty or contain spaces")
        return v

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, v: str) -> str:
        """Validate SHA256 checksum format."""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(
                "Checksum must be valid SHA256 hash (64 hex chars)"
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "checksum": self.checksum,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "verification_status": self.verification_status.value,
            "compliance_level": self.compliance_level.value,
            "experiment_id": self.experiment_id,
            "parent_artifact_ids": self.parent_artifact_ids,
            "child_artifact_ids": self.child_artifact_ids,
            "version": self.version,
            "version_history": self.version_history,
            "metadata": self.metadata,
        }


class ExperimentEntity(BaseModel):
    """Experiment entity with comprehensive tracking."""

    experiment_id: str = Field(..., description="Unique experiment identifier")
    experiment_name: str = Field(
        ..., description="Human-readable experiment name"
    )
    status: ExperimentStatus = Field(
        default=ExperimentStatus.CREATED, description="Experiment status"
    )

    # Metadata
    description: str = Field(default="", description="Experiment description")
    tags: list[str] = Field(
        default_factory=list, description="Categorization tags"
    )

    # Configuration
    config_hash: str = Field(..., description="SHA256 hash of configuration")
    config_summary: dict[str, Any] = Field(
        default_factory=dict, description="Key configuration parameters"
    )

    # Environment
    python_version: str = Field(..., description="Python version")
    pytorch_version: str = Field(..., description="PyTorch version")
    platform: str = Field(..., description="System platform")
    cuda_available: bool = Field(
        default=False, description="CUDA availability"
    )
    cuda_version: str | None = Field(
        None, description="CUDA version if available"
    )

    # Training metadata
    total_epochs: int = Field(default=0, description="Total training epochs")
    current_epoch: int = Field(default=0, description="Current training epoch")
    best_metrics: dict[str, float] = Field(
        default_factory=dict, description="Best metrics achieved"
    )
    training_time_seconds: float = Field(
        default=0.0, description="Total training time"
    )

    # Git metadata
    git_commit: str | None = Field(None, description="Git commit hash")
    git_branch: str | None = Field(None, description="Git branch name")
    git_dirty: bool = Field(default=False, description="Uncommitted changes")

    # System metadata
    hostname: str = Field(..., description="System hostname")
    username: str = Field(..., description="Username")
    memory_gb: float = Field(..., description="Available memory in GB")
    gpu_info: dict[str, Any] = Field(
        default_factory=dict, description="GPU information"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    started_at: datetime | None = Field(None, description="Start timestamp")
    completed_at: datetime | None = Field(
        None, description="Completion timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )

    # Artifact associations
    artifact_ids: list[str] = Field(
        default_factory=list, description="All associated artifact IDs"
    )
    checkpoint_paths: list[str] = Field(
        default_factory=list, description="Checkpoint file paths"
    )
    metric_files: list[str] = Field(
        default_factory=list, description="Metric file paths"
    )
    visualization_files: list[str] = Field(
        default_factory=list, description="Visualization file paths"
    )

    # Lineage
    parent_experiment_ids: list[str] = Field(
        default_factory=list, description="Parent experiments"
    )
    child_experiment_ids: list[str] = Field(
        default_factory=list, description="Child experiments"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, v: str) -> str:
        """Validate experiment ID format."""
        if not v or " " in v:
            raise ValueError("Experiment ID cannot be empty or contain spaces")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "description": self.description,
            "tags": self.tags,
            "config_hash": self.config_hash,
            "config_summary": self.config_summary,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "platform": self.platform,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "total_epochs": self.total_epochs,
            "current_epoch": self.current_epoch,
            "best_metrics": self.best_metrics,
            "training_time_seconds": self.training_time_seconds,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "hostname": self.hostname,
            "username": self.username,
            "memory_gb": self.memory_gb,
            "gpu_info": self.gpu_info,
            "created_at": self.created_at.isoformat(),
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "updated_at": self.updated_at.isoformat(),
            "artifact_ids": self.artifact_ids,
            "checkpoint_paths": self.checkpoint_paths,
            "metric_files": self.metric_files,
            "visualization_files": self.visualization_files,
            "parent_experiment_ids": self.parent_experiment_ids,
            "child_experiment_ids": self.child_experiment_ids,
            "metadata": self.metadata,
        }


class VersionEntity(BaseModel):
    """Version tracking entity for artifacts."""

    version_id: str = Field(..., description="Unique version identifier")
    artifact_id: str = Field(..., description="Associated artifact ID")
    version_number: str = Field(..., description="Semantic version number")

    # File information
    file_path: Path = Field(..., description="Path to versioned file")
    file_size: int = Field(..., description="File size in bytes")
    checksum: str = Field(..., description="SHA256 checksum")

    # Metadata
    description: str = Field(default="", description="Version description")
    tags: list[str] = Field(default_factory=list, description="Version tags")

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    # Change tracking
    change_summary: str = Field(default="", description="Summary of changes")
    change_type: str = Field(
        default="patch", description="Type of change (major/minor/patch)"
    )

    # Dependencies
    dependencies: dict[str, str] = Field(
        default_factory=dict, description="Dependency versions"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("version_number")
    @classmethod
    def validate_version_number(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(
                "Version must be in semantic version format (x.y.z)"
            )
        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            # Validate that all components are non-negative
            if major < 0 or minor < 0 or patch < 0:
                raise ValueError("Version components must be non-negative")
        except ValueError as err:
            raise ValueError("Version components must be integers") from err
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version_id": self.version_id,
            "artifact_id": self.artifact_id,
            "version_number": self.version_number,
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "checksum": self.checksum,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "change_summary": self.change_summary,
            "change_type": self.change_type,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


class LineageEntity(BaseModel):
    """Lineage tracking entity for artifact relationships."""

    lineage_id: str = Field(..., description="Unique lineage identifier")

    # Source and target
    source_artifact_id: str = Field(..., description="Source artifact ID")
    target_artifact_id: str = Field(..., description="Target artifact ID")

    # Relationship type
    relationship_type: str = Field(..., description="Type of relationship")
    relationship_description: str = Field(
        default="", description="Relationship description"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    confidence: float = Field(
        default=1.0, description="Confidence in relationship (0.0-1.0)"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence value range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "lineage_id": self.lineage_id,
            "source_artifact_id": self.source_artifact_id,
            "target_artifact_id": self.target_artifact_id,
            "relationship_type": self.relationship_type,
            "relationship_description": self.relationship_description,
            "created_at": self.created_at.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
