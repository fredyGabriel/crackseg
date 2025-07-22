"""
Unit tests for retention policies and retention manager.

Tests the data retention functionality including time-based,
count-based, and composite retention policies.
"""

import tempfile
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from crackseg.utils.monitoring.retention import (
    CompositeRetentionPolicy,
    CountBasedRetentionPolicy,
    RetentionManager,
    TimeBasedRetentionPolicy,
)


class TestTimeBasedRetentionPolicy:
    """Test suite for TimeBasedRetentionPolicy."""

    def test_initialization(self) -> None:
        """Test policy initialization."""
        policy = TimeBasedRetentionPolicy(max_age_seconds=3600.0)
        assert policy.max_age_seconds == 3600.0

    def test_should_retain_recent_data(self) -> None:
        """Test that recent data is retained."""
        policy = TimeBasedRetentionPolicy(max_age_seconds=3600.0)
        current_time = time.time()
        recent_timestamp = current_time - 1800.0  # 30 minutes ago

        assert policy.should_retain(recent_timestamp, current_time)

    def test_should_not_retain_old_data(self) -> None:
        """Test that old data is not retained."""
        policy = TimeBasedRetentionPolicy(max_age_seconds=3600.0)
        current_time = time.time()
        old_timestamp = current_time - 7200.0  # 2 hours ago

        assert not policy.should_retain(old_timestamp, current_time)

    def test_boundary_condition(self) -> None:
        """Test boundary condition at exact max age."""
        policy = TimeBasedRetentionPolicy(max_age_seconds=3600.0)
        current_time = time.time()
        boundary_timestamp = current_time - 3600.0  # Exactly max age

        assert policy.should_retain(boundary_timestamp, current_time)

    def test_get_description(self) -> None:
        """Test policy description."""
        policy = TimeBasedRetentionPolicy(max_age_seconds=3600.0)
        description = policy.get_description()
        assert "TimeBasedRetentionPolicy" in description
        assert "3600.0" in description


class TestCountBasedRetentionPolicy:
    """Test suite for CountBasedRetentionPolicy."""

    def test_initialization(self) -> None:
        """Test policy initialization."""
        policy = CountBasedRetentionPolicy(max_count=1000)
        assert policy.max_count == 1000

    def test_should_retain_always_true(self) -> None:
        """Test that count-based policy always returns True for
        should_retain."""
        policy = CountBasedRetentionPolicy(max_count=100)
        current_time = time.time()

        # Should always return True regardless of timestamp
        assert policy.should_retain(current_time, current_time)
        assert policy.should_retain(current_time - 10000, current_time)

    def test_get_description(self) -> None:
        """Test policy description."""
        policy = CountBasedRetentionPolicy(max_count=500)
        description = policy.get_description()
        assert "CountBasedRetentionPolicy" in description
        assert "500" in description


class TestCompositeRetentionPolicy:
    """Test suite for CompositeRetentionPolicy."""

    def test_initialization(self) -> None:
        """Test composite policy initialization."""
        time_policy = TimeBasedRetentionPolicy(max_age_seconds=3600.0)
        count_policy = CountBasedRetentionPolicy(max_count=1000)
        composite = CompositeRetentionPolicy([time_policy, count_policy])

        assert len(composite.policies) == 2

    def test_all_policies_must_agree(self) -> None:
        """Test that all policies must agree to retain data."""
        # Create policies with different criteria
        permissive_policy = TimeBasedRetentionPolicy(
            max_age_seconds=7200.0
        )  # 2 hours
        restrictive_policy = TimeBasedRetentionPolicy(
            max_age_seconds=1800.0
        )  # 30 minutes

        composite = CompositeRetentionPolicy(
            [permissive_policy, restrictive_policy]
        )

        current_time = time.time()
        recent_timestamp = current_time - 900.0  # 15 minutes ago
        medium_timestamp = current_time - 3600.0  # 1 hour ago

        # Recent data should be retained (both policies agree)
        assert composite.should_retain(recent_timestamp, current_time)

        # Medium age data should not be retained (restrictive policy disagrees)
        assert not composite.should_retain(medium_timestamp, current_time)

    def test_get_description(self) -> None:
        """Test composite policy description."""
        time_policy = TimeBasedRetentionPolicy(max_age_seconds=3600.0)
        count_policy = CountBasedRetentionPolicy(max_count=1000)
        composite = CompositeRetentionPolicy([time_policy, count_policy])

        description = composite.get_description()
        assert "CompositeRetentionPolicy" in description
        assert "TimeBasedRetentionPolicy" in description
        assert "CountBasedRetentionPolicy" in description


class TestRetentionManager:
    """Test suite for RetentionManager."""

    def test_initialization_default(self) -> None:
        """Test retention manager initialization with defaults."""
        manager = RetentionManager()
        assert manager.retention_policy is not None
        assert isinstance(manager.retention_policy, TimeBasedRetentionPolicy)

    def test_initialization_custom(self) -> None:
        """Test retention manager initialization with custom policy."""
        policy = CountBasedRetentionPolicy(max_count=500)
        manager = RetentionManager(retention_policy=policy)
        assert manager.retention_policy is policy

    def test_apply_time_based_retention(self) -> None:
        """Test applying time-based retention policy."""
        policy = TimeBasedRetentionPolicy(max_age_seconds=3600.0)
        manager = RetentionManager(retention_policy=policy)

        current_time = time.time()

        # Create history with mixed age data
        history = defaultdict(list)
        history["train/loss_values"] = [0.5, 0.3, 0.1]
        history["train/loss_steps"] = [1, 2, 3]
        history["train/loss_timestamps"] = [
            current_time - 7200.0,  # 2 hours ago (should be removed)
            current_time - 1800.0,  # 30 minutes ago (should be kept)
            current_time - 900.0,  # 15 minutes ago (should be kept)
        ]

        filtered_history = manager.apply_retention_policy(history)

        # Should keep only the last 2 entries
        assert len(filtered_history["train/loss_values"]) == 2
        assert filtered_history["train/loss_values"] == [0.3, 0.1]
        assert filtered_history["train/loss_steps"] == [2, 3]

    def test_apply_count_based_retention(self) -> None:
        """Test applying count-based retention policy."""
        policy = CountBasedRetentionPolicy(max_count=2)
        manager = RetentionManager(retention_policy=policy)

        current_time = time.time()

        # Create history with more data than max_count
        history = defaultdict(list)
        history["train/loss_values"] = [0.8, 0.5, 0.3, 0.1]
        history["train/loss_steps"] = [1, 2, 3, 4]
        history["train/loss_timestamps"] = [
            current_time - 3600.0,
            current_time - 2700.0,
            current_time - 1800.0,
            current_time - 900.0,
        ]

        filtered_history = manager.apply_retention_policy(history)

        # Should keep only the last 2 entries
        assert len(filtered_history["train/loss_values"]) == 2
        assert filtered_history["train/loss_values"] == [0.3, 0.1]
        assert filtered_history["train/loss_steps"] == [3, 4]

    def test_group_metrics(self) -> None:
        """Test metric grouping functionality."""
        manager = RetentionManager()

        history = defaultdict(list)
        history["train/loss_values"] = [0.5, 0.3]
        history["train/loss_steps"] = [1, 2]
        history["train/loss_timestamps"] = [1234567890.0, 1234567900.0]
        history["train/accuracy_values"] = [0.8, 0.9]
        history["train/accuracy_steps"] = [1, 2]
        history["train/accuracy_timestamps"] = [1234567890.0, 1234567900.0]

        groups = manager._group_metrics(history)

        assert "train/loss" in groups
        assert "train/accuracy" in groups
        assert len(groups["train/loss"]) == 3  # values, steps, timestamps
        assert len(groups["train/accuracy"]) == 3

    def test_save_and_load_from_disk(self) -> None:
        """Test saving and loading metrics from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "metrics.json"
            manager = RetentionManager(storage_path=storage_path)

            # Create test data
            history = defaultdict(list)
            history["train/loss_values"] = [0.5, 0.3]
            history["train/loss_steps"] = [1, 2]

            # Save to disk
            success = manager.save_to_disk(history)
            assert success
            assert storage_path.exists()

            # Load from disk
            loaded_history = manager.load_from_disk()
            assert loaded_history["train/loss_values"] == [0.5, 0.3]
            assert loaded_history["train/loss_steps"] == [1, 2]

    def test_save_without_storage_path(self) -> None:
        """Test saving without storage path returns False."""
        manager = RetentionManager()
        history = defaultdict(list)

        success = manager.save_to_disk(history)
        assert not success

    def test_load_nonexistent_file(self) -> None:
        """Test loading from non-existent file returns empty dict."""
        storage_path = Path("/nonexistent/path/metrics.json")
        manager = RetentionManager(storage_path=storage_path)

        loaded_history = manager.load_from_disk()
        assert len(loaded_history) == 0

    def test_get_storage_size(self) -> None:
        """Test getting storage size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "metrics.json"
            manager = RetentionManager(storage_path=storage_path)

            # Initially no file
            assert manager.get_storage_size() == 0

            # Save some data
            history = defaultdict(list)
            history["train/loss_values"] = [0.5, 0.3]
            manager.save_to_disk(history)

            # Should have some size now
            assert manager.get_storage_size() > 0

    def test_clear_storage(self) -> None:
        """Test clearing storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "metrics.json"
            manager = RetentionManager(storage_path=storage_path)

            # Create and save data
            history = defaultdict(list)
            history["train/loss_values"] = [0.5, 0.3]
            manager.save_to_disk(history)
            assert storage_path.exists()

            # Clear storage
            success = manager.clear_storage()
            assert success
            assert not storage_path.exists()

    def test_should_cleanup_timing(self) -> None:
        """Test cleanup timing logic."""
        manager = RetentionManager(auto_cleanup_interval=1.0)  # 1 second

        # Initially should not need cleanup
        assert not manager.should_cleanup()

        # Mark cleanup as done, then wait
        manager.mark_cleanup_done()

        with patch("time.time", return_value=time.time() + 2.0):
            assert manager.should_cleanup()

    def test_empty_history_handling(self) -> None:
        """Test handling of empty history."""
        manager = RetentionManager()
        empty_history = defaultdict(list)

        filtered_history = manager.apply_retention_policy(empty_history)
        assert len(filtered_history) == 0

    def test_metrics_without_timestamps(self) -> None:
        """Test handling metrics without timestamps."""
        manager = RetentionManager()

        history = defaultdict(list)
        history["train/loss_values"] = [0.5, 0.3]
        history["train/loss_steps"] = [1, 2]
        # No timestamps

        filtered_history = manager.apply_retention_policy(history)

        # Should keep all data when no timestamps
        assert filtered_history["train/loss_values"] == [0.5, 0.3]
        assert filtered_history["train/loss_steps"] == [1, 2]
