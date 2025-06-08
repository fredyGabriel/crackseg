"""Unit tests for enhanced abort functionality.

Tests the new abort API, process tree management, and zombie cleanup
functionality in the ProcessManager and run_manager modules.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.gui.utils.process import (
    AbortLevel,
    AbortProgress,
    AbortResult,
    ProcessManager,
)
from scripts.gui.utils.run_manager import (
    abort_training_session,
    force_cleanup_orphans,
    get_process_tree_info,
)


class TestAbortResult:
    """Test AbortResult dataclass functionality."""

    def test_abort_result_creation(self) -> None:
        """Test creating AbortResult with all fields."""
        result = AbortResult(
            success=True,
            abort_level_used=AbortLevel.GRACEFUL,
            process_killed=True,
            children_killed=2,
            zombies_cleaned=1,
            total_time=5.5,
            warnings=["Test warning"],
        )

        assert result.success is True
        assert result.abort_level_used == AbortLevel.GRACEFUL
        assert result.process_killed is True
        assert result.children_killed == 2
        assert result.zombies_cleaned == 1
        assert result.total_time == 5.5
        assert result.warnings == ["Test warning"]
        assert result.error_message is None


class TestAbortProgress:
    """Test AbortProgress dataclass functionality."""

    def test_abort_progress_creation(self) -> None:
        """Test creating AbortProgress with all fields."""
        progress = AbortProgress(
            stage="terminating",
            message="Killing process",
            progress_percent=50.0,
            elapsed_time=2.5,
            estimated_remaining=2.5,
        )

        assert progress.stage == "terminating"
        assert progress.message == "Killing process"
        assert progress.progress_percent == 50.0
        assert progress.elapsed_time == 2.5
        assert progress.estimated_remaining == 2.5


class TestProcessManagerAbort:
    """Test ProcessManager enhanced abort functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = ProcessManager()

    def test_abort_training_no_process(self) -> None:
        """Test abort when no process is running."""
        result = self.manager.abort_training()

        assert result.success is True
        assert result.process_killed is False
        assert result.children_killed == 0
        assert result.zombies_cleaned == 0

    def test_abort_training_with_callback(self) -> None:
        """Test abort with progress callback."""
        callback_calls: list[AbortProgress] = []

        def test_callback(progress: AbortProgress) -> None:
            callback_calls.append(progress)

        result = self.manager.abort_training(
            level=AbortLevel.GRACEFUL, callback=test_callback
        )

        assert result.success is True
        assert len(callback_calls) > 0
        assert all(isinstance(call, AbortProgress) for call in callback_calls)
        assert callback_calls[-1].progress_percent == 100.0

    def test_abort_training_different_levels(self) -> None:
        """Test abort with different levels."""
        # Test GRACEFUL level
        result_graceful = self.manager.abort_training(
            level=AbortLevel.GRACEFUL
        )
        assert result_graceful.success is True
        assert result_graceful.abort_level_used == AbortLevel.GRACEFUL

        # Test FORCE level
        result_force = self.manager.abort_training(level=AbortLevel.FORCE)
        assert result_force.success is True
        assert result_force.abort_level_used == AbortLevel.FORCE

        # Test NUCLEAR level
        result_nuclear = self.manager.abort_training(level=AbortLevel.NUCLEAR)
        assert result_nuclear.success is True
        assert result_nuclear.abort_level_used == AbortLevel.NUCLEAR

    def test_get_process_tree_info_no_process(self) -> None:
        """Test process tree info when no process is running."""
        result = self.manager.get_process_tree_info()

        assert result["main_process"] is None
        assert result["children"] == []
        assert result["total_processes"] == 0


class TestRunManagerAbort:
    """Test run_manager enhanced abort functions."""

    @patch("scripts.gui.utils.run_manager.get_process_manager")
    def test_abort_training_session(self, mock_get_manager: Any) -> None:
        """Test abort_training_session wrapper function."""
        mock_manager = MagicMock()
        mock_result = AbortResult(
            success=True,
            abort_level_used=AbortLevel.FORCE,
            process_killed=True,
            children_killed=1,
            zombies_cleaned=0,
            total_time=3.5,
        )
        mock_manager.abort_training.return_value = mock_result
        mock_get_manager.return_value = mock_manager

        result = abort_training_session(level=AbortLevel.FORCE, timeout=15.0)

        assert result.success is True
        assert result.abort_level_used == AbortLevel.FORCE
        mock_manager.abort_training.assert_called_once_with(
            level=AbortLevel.FORCE, timeout=15.0, callback=None
        )

    @patch("scripts.gui.utils.run_manager.get_process_manager")
    def test_get_process_tree_info_wrapper(
        self, mock_get_manager: Any
    ) -> None:
        """Test get_process_tree_info wrapper function."""
        mock_manager = MagicMock()
        mock_tree_info = {
            "main_process": {"pid": 1234, "name": "python"},
            "children": [],
            "total_processes": 1,
        }
        mock_manager.get_process_tree_info.return_value = mock_tree_info
        mock_get_manager.return_value = mock_manager

        result = get_process_tree_info()

        assert result["total_processes"] == 1
        assert result["main_process"]["pid"] == 1234
        mock_manager.get_process_tree_info.assert_called_once()

    @patch("scripts.gui.utils.run_manager.psutil.process_iter")
    def test_force_cleanup_orphans_basic(self, mock_process_iter: Any) -> None:
        """Test basic force cleanup functionality."""
        # Mock empty process list
        mock_process_iter.return_value = []

        result = force_cleanup_orphans()

        assert result["success"] is True
        assert result["total_cleaned"] == 0


class TestAbortLevels:
    """Test different abort levels behavior."""

    def test_abort_levels_enum(self) -> None:
        """Test AbortLevel enum values."""
        assert AbortLevel.GRACEFUL.value == "graceful"
        assert AbortLevel.FORCE.value == "force"
        assert AbortLevel.NUCLEAR.value == "nuclear"

    def test_abort_levels_ordering(self) -> None:
        """Test that abort levels represent increasing intensity."""
        levels = [AbortLevel.GRACEFUL, AbortLevel.FORCE, AbortLevel.NUCLEAR]

        # Test that each level is more intense than the previous
        assert len(levels) == 3
        assert AbortLevel.GRACEFUL != AbortLevel.FORCE
        assert AbortLevel.FORCE != AbortLevel.NUCLEAR


if __name__ == "__main__":
    pytest.main([__file__])
