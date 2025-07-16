"""Unit tests for the SessionState and SessionStateManager."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from gui.utils.session_state import SessionState, SessionStateManager


@pytest.fixture
def initial_state() -> SessionState:
    """Fixture for a clean SessionState instance."""
    return SessionState()


class TestSessionState:
    """Tests for the SessionState dataclass and its methods."""

    def test_initialization(self, initial_state: SessionState):
        """Test that the dataclass initializes with correct defaults."""
        assert initial_state.current_page == "Home"
        assert initial_state.config_loaded is False
        assert initial_state.process_state == "idle"
        assert isinstance(initial_state.notifications, list)
        assert isinstance(initial_state.last_updated, datetime)

    def test_to_dict_serialization(self, initial_state: SessionState):
        """Test serialization to a dictionary."""
        initial_state.config_path = "/fake/path"
        initial_state.last_updated = datetime(2023, 1, 1)
        data = initial_state.to_dict()

        assert data["config_path"] == "/fake/path"
        assert data["last_updated"] == "2023-01-01T00:00:00"
        assert "_update_lock" not in data
        assert "current_model" not in data

    def test_from_dict_deserialization(self):
        """Test deserialization from a dictionary."""
        data = {
            "config_path": "/fake/path",
            "config_loaded": True,
            "last_updated": "2023-01-01T00:00:00",
            "unknown_field": "should be ignored",
        }
        state = SessionState.from_dict(data)

        assert state.config_path == "/fake/path"
        assert state.config_loaded is True
        assert state.last_updated == datetime(2023, 1, 1)

    def test_update_config(self, initial_state: SessionState):
        """Test the update_config method."""
        initial_state.update_config("/new/path", {"key": "value"})
        assert initial_state.config_path == "/new/path"
        assert initial_state.config_data == {"key": "value"}
        assert initial_state.config_loaded is True

    def test_is_ready_for_training(self, initial_state: SessionState):
        """Test the is_ready_for_training check."""
        assert initial_state.is_ready_for_training() is False
        initial_state.config_loaded = True
        assert initial_state.is_ready_for_training() is False
        initial_state.run_directory = "/some/dir"
        assert initial_state.is_ready_for_training() is True

    def test_add_notification(self, initial_state: SessionState):
        """Test adding notifications."""
        assert len(initial_state.notifications) == 0
        initial_state.add_notification("Test message")
        assert len(initial_state.notifications) == 1
        assert "Test message" in initial_state.notifications[0]


class TestSessionStateManager:
    """Tests for the SessionStateManager using dependency injection."""

    def test_initialize_and_get(self):
        """Test initialize and get with a mock state container."""
        mock_state: dict[str, Any] = {}
        SessionStateManager.initialize(mock_state)
        assert "_crackseg_state" in mock_state
        assert isinstance(mock_state["_crackseg_state"], SessionState)
        assert (
            SessionStateManager.get(mock_state)
            == mock_state["_crackseg_state"]
        )

    def test_update(self):
        """Test updating the state via the manager."""
        mock_state: dict[str, Any] = {}
        SessionStateManager.initialize(mock_state)
        SessionStateManager.update(
            {"theme": "light", "config_loaded": True}, mock_state
        )
        state = SessionStateManager.get(mock_state)
        assert state.theme == "light"
        assert state.config_loaded is True

    def test_save_and_load_file(self, tmp_path: Path):
        """Test saving and loading state from a file."""
        mock_state: dict[str, Any] = {}
        file_path = tmp_path / "session.json"

        # Save
        SessionStateManager.initialize(mock_state)
        SessionStateManager.update({"config_path": "/test/path"}, mock_state)
        save_success = SessionStateManager.save_to_file(file_path, mock_state)
        assert save_success

        # Load into a new mock state
        new_mock_state: dict[str, Any] = {}
        load_success = SessionStateManager.load_from_file(
            file_path, new_mock_state
        )
        assert load_success
        loaded_state = SessionStateManager.get(new_mock_state)
        assert loaded_state.config_path == "/test/path"

    def test_load_non_existent_file(self):
        """Test that loading a non-existent file returns False."""
        mock_state: dict[str, Any] = {}
        result = SessionStateManager.load_from_file(
            Path("/non/existent/file.json"), mock_state
        )
        assert not result
