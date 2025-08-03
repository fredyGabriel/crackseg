"""Core data structures for results scanning.

This module contains the fundamental data structures used by the results
scanning system, following the single responsibility principle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path


class TripletType(Enum):
    """Types of prediction triplets for crack segmentation."""

    IMAGE = auto()
    MASK = auto()
    PREDICTION = auto()


class TripletHealth(Enum):
    """Represents the health status of a triplet's files."""

    HEALTHY = "Healthy"  # All files exist.
    DEGRADED = "Degraded"  # Some files are missing.
    BROKEN = "Broken"  # All essential files are missing.


@dataclass
class ResultTriplet:
    """Represents a complete prediction triplet (image|mask|prediction)."""

    id: str
    image_path: Path | None
    mask_path: Path | None
    prediction_path: Path | None
    dataset_name: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, str | int | float] = field(default_factory=dict)
    health_status: TripletHealth = TripletHealth.HEALTHY
    missing_files: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate triplet paths and check initial health."""
        if not self.id:
            raise ValueError("Triplet ID cannot be empty")
        self.check_health()

    def check_health(self) -> TripletHealth:
        """
        Checks for the existence of triplet files and updates health status.

        Returns:
            The current health status of the triplet.
        """
        self.missing_files.clear()
        paths_to_check = {
            self.image_path,
            self.mask_path,
            self.prediction_path,
        }

        for p in [self.image_path, self.mask_path, self.prediction_path]:
            if p is None or not p.exists():
                if p is not None:
                    self.missing_files.append(p)

        if not self.missing_files:
            self.health_status = TripletHealth.HEALTHY
        elif len(self.missing_files) < len(paths_to_check):
            self.health_status = TripletHealth.DEGRADED
        else:
            self.health_status = TripletHealth.BROKEN

        return self.health_status

    @property
    def is_complete(self) -> bool:
        """Check if the triplet is complete (all files exist)."""
        return self.health_status == TripletHealth.HEALTHY

    @property
    def file_sizes(self) -> dict[str, int]:
        """Get file sizes for each component."""
        return {
            "image": (
                self.image_path.stat().st_size
                if self.image_path and self.image_path.exists()
                else 0
            ),
            "mask": (
                self.mask_path.stat().st_size
                if self.mask_path and self.mask_path.exists()
                else 0
            ),
            "prediction": (
                self.prediction_path.stat().st_size
                if self.prediction_path and self.prediction_path.exists()
                else 0
            ),
        }


@dataclass
class ScanProgress:
    """Progress information for async scanning operations."""

    total_files: int = 0
    processed_files: int = 0
    found_triplets: int = 0
    errors: int = 0
    current_directory: Path | None = None

    @property
    def progress_percent(self) -> float:
        """Calculate completion percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if scanning is complete."""
        return (
            self.processed_files >= self.total_files and self.total_files > 0
        )
