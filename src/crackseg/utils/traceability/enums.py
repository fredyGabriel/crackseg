"""
Traceability enums for CrackSeg project.

This module defines all enumeration types used in the traceability system.
"""

from enum import Enum


class ArtifactType(str, Enum):
    """Types of artifacts in the system."""

    MODEL = "model"
    CHECKPOINT = "checkpoint"
    METRICS = "metrics"
    VISUALIZATION = "visualization"
    CONFIGURATION = "configuration"
    PREDICTION = "prediction"
    REPORT = "report"
    LOG = "log"
    DATASET = "dataset"
    EVALUATION = "evaluation"


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    PAUSED = "paused"


class VerificationStatus(str, Enum):
    """Integrity verification status."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    WARNING = "warning"


class ComplianceLevel(str, Enum):
    """Compliance and audit levels."""

    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    AUDIT = "audit"
