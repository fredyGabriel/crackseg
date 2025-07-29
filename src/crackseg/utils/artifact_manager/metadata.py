"""
Metadata management for artifacts.

This module contains the ArtifactMetadata dataclass and related functionality
for tracking artifact information.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ArtifactMetadata:
    """Metadata for artifact tracking and versioning."""

    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    artifact_type: str = ""
    file_path: str = ""
    file_size: int = 0
    checksum: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
