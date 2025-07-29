"""
Experiment metadata dataclass for comprehensive experiment tracking.

This module provides the ExperimentMetadata dataclass for structuring
experiment metadata and lifecycle information.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExperimentMetadata:
    """Comprehensive metadata for experiment tracking."""

    experiment_id: str
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "created"  # created, running, completed, failed, aborted
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Configuration metadata
    config_hash: str = ""
    config_summary: dict[str, Any] = field(default_factory=dict)

    # Environment metadata
    python_version: str = ""
    pytorch_version: str = ""
    platform: str = ""
    cuda_available: bool = False
    cuda_version: str = ""

    # Training metadata
    total_epochs: int = 0
    current_epoch: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0

    # Artifact associations
    artifact_ids: list[str] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    metric_files: list[str] = field(default_factory=list)
    visualization_files: list[str] = field(default_factory=list)

    # Git metadata
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False

    # System metadata
    hostname: str = ""
    username: str = ""
    memory_gb: float = 0.0
    gpu_info: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str = ""
    completed_at: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()
