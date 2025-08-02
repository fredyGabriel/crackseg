"""
Traceability system for CrackSeg project.

This module provides comprehensive artifact traceability, including data
models, query interfaces, and lineage tracking for ML experiments.
"""

from .access_control import AccessControl
from .integration_manager import TraceabilityIntegrationManager
from .lineage_manager import LineageManager
from .metadata_manager import MetadataManager
from .models import (
    ArtifactEntity,
    ArtifactType,
    ComplianceLevel,
    ComplianceRecord,
    ExperimentEntity,
    ExperimentStatus,
    LineageEntity,
    TraceabilityExport,
    TraceabilityQuery,
    TraceabilityResult,
    VerificationStatus,
    VersionEntity,
)
from .query_interface import TraceabilityQueryInterface
from .storage import TraceabilityStorage

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
    # Query interface
    "TraceabilityQueryInterface",
    # Storage
    "TraceabilityStorage",
    # Lineage management
    "LineageManager",
    # Metadata and access control
    "MetadataManager",
    "AccessControl",
    # Integration manager
    "TraceabilityIntegrationManager",
]
