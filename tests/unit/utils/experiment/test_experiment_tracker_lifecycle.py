"""
Unit tests for ExperimentTracker lifecycle management.

Tests experiment lifecycle management functionality.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from crackseg.utils.experiment import ExperimentTracker


class TestExperimentTrackerLifecycle:
    """Test suite for ExperimentTracker lifecycle management."""

    @pytest.fixture
    def temp_experiment_dir(self) -> Iterator[Path]:
        """Create temporary experiment directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_experiment_lifecycle(self, temp_experiment_dir: Path) -> None:
        """Test experiment lifecycle management."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        # Test start
        tracker.start_experiment()
        assert tracker.metadata.status == "running"
        assert tracker.metadata.started_at != ""

        # Test complete
        tracker.complete_experiment()
        assert tracker.metadata.status == "completed"
        assert tracker.metadata.completed_at != ""

    def test_experiment_failure(self, temp_experiment_dir: Path) -> None:
        """Test experiment failure handling."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        tracker.fail_experiment("Test error message")
        assert tracker.metadata.status == "failed"
        assert "Test error message" in tracker.metadata.description

    def test_experiment_abort(self, temp_experiment_dir: Path) -> None:
        """Test experiment abort handling."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        tracker.abort_experiment()
        assert tracker.metadata.status == "aborted"
        assert tracker.metadata.completed_at != ""

    def test_training_progress_update(self, temp_experiment_dir: Path) -> None:
        """Test training progress metadata updates."""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        tracker.update_training_progress(
            current_epoch=5,
            total_epochs=100,
            best_metrics={"iou": 0.85, "dice": 0.90},
            training_time_seconds=3600.0,
        )

        assert tracker.metadata.current_epoch == 5
        assert tracker.metadata.total_epochs == 100
        assert tracker.metadata.best_metrics["iou"] == 0.85
        assert tracker.metadata.best_metrics["dice"] == 0.90
        assert tracker.metadata.training_time_seconds == 3600.0
