"""
Traceability data models for CrackSeg project.

This module provides a unified interface for all traceability models,
re-exporting from the modular implementation files.
"""

# Re-export enums
# Re-export compliance and export models
from .compliance import ComplianceRecord, TraceabilityExport

# Re-export core entities
from .entities import (
    ArtifactEntity,
    ExperimentEntity,
    LineageEntity,
    VersionEntity,
)
from .enums import (
    ArtifactType,
    ComplianceLevel,
    ExperimentStatus,
    VerificationStatus,
)

# Re-export query and result models
from .queries import TraceabilityQuery, TraceabilityResult

__all__ = [
    # Enums
    "ArtifactType",
    "ExperimentStatus",
    "VerificationStatus",
    "ComplianceLevel",
    # Core entities
    "ArtifactEntity",
    "ExperimentEntity",
    "VersionEntity",
    "LineageEntity",
    # Query and results
    "TraceabilityQuery",
    "TraceabilityResult",
    # Compliance and export
    "ComplianceRecord",
    "TraceabilityExport",
]
