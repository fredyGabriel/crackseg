"""
Unit tests for ExperimentMetadata dataclass.

Tests metadata creation, serialization, and timestamp functionality.
"""

import time

from crackseg.utils.experiment.metadata import ExperimentMetadata


class TestExperimentMetadata:
    """Test suite for ExperimentMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test basic metadata creation."""
        metadata = ExperimentMetadata(
            experiment_id="test-123",
            experiment_name="test_experiment",
            description="Test experiment",
            tags=["test", "unit"],
        )

        assert metadata.experiment_id == "test-123"
        assert metadata.experiment_name == "test_experiment"
        assert metadata.description == "Test experiment"
        assert metadata.tags == ["test", "unit"]
        assert metadata.status == "created"

    def test_metadata_to_dict(self) -> None:
        """Test metadata serialization to dictionary."""
        metadata = ExperimentMetadata(
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        metadata_dict = metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["experiment_id"] == "test-123"
        assert metadata_dict["experiment_name"] == "test_experiment"
        assert "created_at" in metadata_dict
        assert "updated_at" in metadata_dict

    def test_metadata_timestamp_update(self) -> None:
        """Test metadata timestamp update functionality."""
        metadata = ExperimentMetadata(
            experiment_id="test-123",
            experiment_name="test_experiment",
        )

        original_updated = metadata.updated_at
        time.sleep(0.001)  # Small delay to ensure timestamp difference
        metadata.update_timestamp()

        assert metadata.updated_at != original_updated
