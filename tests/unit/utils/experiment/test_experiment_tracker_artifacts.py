"""
Unit tests for ExperimentTracker artifact management.

Tests artifact association and retrieval functionality.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from crackseg.utils.experiment import ExperimentTracker


class TestExperimentTrackerArtifacts:
    """Test suite for ExperimentTracker artifact management."""

    @pytest.fixture
    def temp_experiment_dir(self) -> Iterator[Path]:
        """Create temporary experiment directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_artifact_association(self, temp_experiment_dir: Path) -> None:
        """Test artifact association functionality."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Add different types of artifacts
        tracker.add_artifact(
            artifact_id="model-001",
            artifact_type="checkpoint",
            file_path="/path/to/model.pth",
            description="Best model checkpoint",
            tags=["best", "model"],
        )

        tracker.add_artifact(
            artifact_id="metrics-001",
            artifact_type="metrics",
            file_path="/path/to/metrics.json",
            description="Training metrics",
            tags=["metrics"],
        )

        tracker.add_artifact(
            artifact_id="viz-001",
            artifact_type="visualization",
            file_path="/path/to/plot.png",
            description="Training visualization",
            tags=["visualization"],
        )

        # Verify artifact associations
        assert "model-001" in tracker.metadata.artifact_ids
        assert "metrics-001" in tracker.metadata.artifact_ids
        assert "viz-001" in tracker.metadata.artifact_ids

        # Verify type-specific lists
        assert "/path/to/model.pth" in tracker.metadata.checkpoint_paths
        assert "/path/to/metrics.json" in tracker.metadata.metric_files
        assert "/path/to/plot.png" in tracker.metadata.visualization_files

    def test_artifact_retrieval_by_type(
        self, temp_experiment_dir: Path
    ) -> None:
        """Test artifact retrieval by type."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Add artifacts
        tracker.add_artifact("model-001", "checkpoint", "/path/to/model.pth")
        tracker.add_artifact("metrics-001", "metrics", "/path/to/metrics.json")
        tracker.add_artifact("viz-001", "visualization", "/path/to/plot.png")

        # Test retrieval
        checkpoints = tracker.get_artifacts_by_type("checkpoint")
        metrics = tracker.get_artifacts_by_type("metrics")
        visualizations = tracker.get_artifacts_by_type("visualization")
        unknown = tracker.get_artifacts_by_type("unknown")

        assert checkpoints == ["/path/to/model.pth"]
        assert metrics == ["/path/to/metrics.json"]
        assert visualizations == ["/path/to/plot.png"]
        assert unknown == []

    def test_experiment_summary(self, temp_experiment_dir: Path) -> None:
        """Test experiment summary generation."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Add some artifacts and update progress
        tracker.add_artifact("model-001", "checkpoint", "/path/to/model.pth")
        tracker.update_training_progress(10, 100, {"iou": 0.85})

        summary = tracker.get_experiment_summary()

        assert summary["experiment_id"] == "test-123"
        assert summary["experiment_name"] == "test_experiment"
        assert summary["status"] == "created"
        assert summary["total_epochs"] == 100
        assert summary["current_epoch"] == 10
        assert summary["best_metrics"]["iou"] == 0.85
        assert summary["artifact_count"] == 1
