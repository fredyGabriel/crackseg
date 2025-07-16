"""Unit tests for session state update functionality.

Tests the session state synchronization implementation for subtask 5.6,
including SessionState extensions, SessionStateManager integration,
and SessionSyncCoordinator functionality.
"""

# pyright: reportPrivateUsage=false

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from gui.utils.session_state import SessionState, SessionStateManager
from gui.utils.session_sync import SessionSyncCoordinator


class TestSessionStateExtensions:
    """Test the extended SessionState class functionality."""

    def test_session_state_initialization(self) -> None:
        """Test that SessionState initializes with new fields."""
        state = SessionState()

        # Test process lifecycle fields
        assert state.process_pid is None
        assert state.process_state == "idle"
        assert state.process_start_time is None
        assert state.process_command == []
        assert state.process_working_dir is None
        assert state.process_return_code is None
        assert state.process_error_message is None
        assert state.process_memory_usage == {}

        # Test log streaming fields
        assert state.log_streaming_active is False
        assert state.log_buffer_size == 0
        assert state.log_last_update is None
        assert state.recent_logs == []
        assert state.hydra_run_dir is None

        # Test training statistics fields
        assert state.training_epoch is None
        assert state.training_loss is None
        assert state.training_learning_rate is None
        assert state.validation_metrics == {}

        # Test thread safety - verify lock exists
        assert hasattr(state, "_update_lock")

    def test_update_process_state(self) -> None:
        """Test process state update functionality."""
        state = SessionState()

        # Test basic process state update
        state.update_process_state(
            state="running",
            pid=12345,
            command=["python", "run.py"],
            start_time=time.time(),
            working_dir="/path/to/work",
            memory_usage={"rss": 1024.0, "vms": 2048.0},
        )

        assert state.process_state == "running"
        assert state.process_pid == 12345
        assert state.process_command == ["python", "run.py"]
        assert state.process_working_dir == "/path/to/work"
        assert state.process_memory_usage == {"rss": 1024.0, "vms": 2048.0}
        assert state.training_active is True  # Should be set automatically

        # Test state transition to completed
        state.update_process_state(state="completed", return_code=0)
        assert state.process_state == "completed"
        assert state.process_return_code == 0
        assert state.training_active is False  # Should be set automatically

    def test_update_log_streaming_state(self) -> None:
        """Test log streaming state update functionality."""
        state = SessionState()

        recent_logs = [
            {
                "message": "Training started",
                "level": "info",
                "timestamp": time.time(),
            },
            {
                "message": "Epoch 1 completed",
                "level": "info",
                "timestamp": time.time(),
            },
        ]

        state.update_log_streaming_state(
            active=True,
            buffer_size=100,
            recent_logs=recent_logs,
            hydra_run_dir="/path/to/hydra/run",
        )

        assert state.log_streaming_active is True
        assert state.log_buffer_size == 100
        assert state.recent_logs == recent_logs
        assert state.hydra_run_dir == "/path/to/hydra/run"
        assert state.log_last_update is not None

    def test_update_training_stats_from_logs(self) -> None:
        """Test training statistics update from logs."""
        state = SessionState()

        # Test updating training statistics
        validation_metrics = {"accuracy": 0.95, "f1_score": 0.89}

        state.update_training_stats_from_logs(
            epoch=10,
            loss=0.1234,
            learning_rate=0.001,
            validation_metrics=validation_metrics,
        )

        assert state.training_epoch == 10
        assert state.training_loss == 0.1234
        assert state.training_learning_rate == 0.001
        assert state.validation_metrics == validation_metrics

    def test_reset_process_state(self) -> None:
        """Test process state reset functionality."""
        state = SessionState()

        # Set up some state
        state.update_process_state(
            state="running", pid=12345, command=["python", "run.py"]
        )
        state.update_log_streaming_state(active=True, buffer_size=50)
        state.update_training_stats_from_logs(epoch=5, loss=0.5)

        # Reset state
        state.reset_process_state()

        # Verify all process-related state is reset
        assert state.process_pid is None
        assert state.process_state == "idle"
        assert state.process_command == []
        assert state.log_streaming_active is False
        assert state.log_buffer_size == 0
        assert state.recent_logs == []
        assert state.training_active is False
        assert state.training_epoch is None

    def test_thread_safety(self) -> None:
        """Test thread safety of session state updates."""
        state = SessionState()
        errors = []

        def update_worker(worker_id: int) -> None:
            try:
                for i in range(100):
                    state.update_process_state(
                        state=f"state_{worker_id}_{i}",
                        pid=worker_id * 1000 + i,
                    )
                    time.sleep(
                        0.001
                    )  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)

        # Start multiple threads updating state concurrently
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=update_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

    def test_is_process_running(self) -> None:
        """Test process running status check."""
        state = SessionState()

        # Initially not running
        assert state.is_process_running() is False

        # Test running states
        for running_state in ["starting", "running"]:
            state.update_process_state(state=running_state)
            assert state.is_process_running() is True

        # Test non-running states
        for non_running_state in ["idle", "completed", "failed", "aborted"]:
            state.update_process_state(state=non_running_state)
            assert state.is_process_running() is False

    def test_get_process_status_summary(self) -> None:
        """Test process status summary generation."""
        state = SessionState()

        # Set up process state
        state.update_process_state(
            state="running",
            pid=12345,
            start_time=time.time(),
            memory_usage={"rss": 1024.0},
        )
        state.update_log_streaming_state(
            active=True, hydra_run_dir="/path/to/hydra"
        )

        summary = state.get_process_status_summary()

        assert summary["state"] == "running"
        assert summary["pid"] == 12345
        assert summary["active"] is True
        assert summary["memory_usage"] == {"rss": 1024.0}
        assert summary["log_streaming"] is True
        assert summary["hydra_dir"] == "/path/to/hydra"

    def test_state_validation(self) -> None:
        """Test session state validation functionality."""
        state = SessionState()

        # Test validation with inconsistent state
        state.process_state = "running"
        state.training_active = False

        issues = state.validate()
        assert any("inconsistent" in issue.lower() for issue in issues)

        # Test validation with consistent state
        state.training_active = True
        issues = state.validate()
        assert not any("inconsistent" in issue.lower() for issue in issues)


class TestSessionStateManager:
    """Test the extended SessionStateManager functionality."""

    @patch("streamlit.session_state")
    def test_update_from_process_info(
        self, mock_st_session_state: MagicMock
    ) -> None:
        """Test updating session state from ProcessInfo object."""
        # Mock a ProcessInfo object
        mock_process_info = MagicMock()
        mock_process_info.state.value = "running"
        mock_process_info.pid = 12345
        mock_process_info.command = ["python", "run.py"]
        mock_process_info.start_time = time.time()
        mock_process_info.working_directory = Path("/work/dir")
        mock_process_info.return_code = None
        mock_process_info.error_message = None

        # Mock session state
        mock_state = SessionState()
        # Configure mock to behave like a dictionary with app_state
        mock_st_session_state.__contains__ = MagicMock(return_value=True)
        mock_st_session_state.__getitem__ = MagicMock(return_value=mock_state)
        mock_st_session_state.app_state = mock_state

        # Update from process info
        SessionStateManager.update_from_process_info(mock_process_info)

        # Verify state was updated
        assert mock_state.process_state == "running"
        assert mock_state.process_pid == 12345
        assert mock_state.process_command == ["python", "run.py"]
        assert mock_state.process_working_dir == str(Path("/work/dir"))

    @patch("streamlit.session_state")
    def test_update_from_log_stream_info(
        self, mock_st_session_state: MagicMock
    ) -> None:
        """Test updating session state from log stream information."""
        mock_state = SessionState()
        mock_st_session_state.app_state = mock_state

        recent_logs = [{"message": "test log", "level": "info"}]

        SessionStateManager.update_from_log_stream_info(
            active=True,
            buffer_size=150,
            recent_logs=recent_logs,
            hydra_run_dir="/hydra/run",
        )

        assert mock_state.log_streaming_active is True
        assert mock_state.log_buffer_size == 150
        assert mock_state.recent_logs == recent_logs
        assert mock_state.hydra_run_dir == "/hydra/run"

    @patch("streamlit.session_state")
    def test_extract_training_stats_from_logs(
        self, mock_st_session_state: MagicMock
    ) -> None:
        """Test extracting training statistics from log entries."""
        mock_state = SessionState()
        mock_st_session_state.app_state = mock_state

        # Mock log entries with training information
        logs = [
            {"message": "epoch: 5 started", "timestamp": time.time()},
            {"message": "training loss: 0.1234", "timestamp": time.time()},
            {"message": "lr: 0.001", "timestamp": time.time()},
        ]

        SessionStateManager.extract_training_stats_from_logs(logs)

        # Verify statistics were extracted
        assert mock_state.training_epoch == 5
        assert mock_state.training_loss == 0.1234
        assert mock_state.training_learning_rate == 0.001

    @patch("streamlit.session_state")
    def test_reset_training_session(
        self, mock_st_session_state: MagicMock
    ) -> None:
        """Test resetting training session state."""
        mock_state = SessionState()
        mock_st_session_state.app_state = mock_state

        # Set up some state
        mock_state.update_process_state(state="running", pid=12345)
        mock_state.update_log_streaming_state(active=True)

        # Reset session
        SessionStateManager.reset_training_session()

        # Verify state was reset
        assert mock_state.process_state == "idle"
        assert mock_state.process_pid is None
        assert mock_state.log_streaming_active is False


class TestSessionSyncCoordinator:
    """Test the SessionSyncCoordinator functionality."""

    def test_coordinator_initialization(self) -> None:
        """Test coordinator initialization."""
        coordinator = SessionSyncCoordinator()

        # Test public interface behavior
        status = coordinator.get_sync_status()
        assert status["active"] is False
        assert "callback_counts" in status
        assert "process_update" in status["callback_counts"]
        assert "log_update" in status["callback_counts"]
        assert "metrics_update" in status["callback_counts"]

    def test_start_stop_coordinator(self) -> None:
        """Test starting and stopping the coordinator."""
        coordinator = SessionSyncCoordinator()

        # Test initial state
        status = coordinator.get_sync_status()
        assert status["active"] is False

        # Test start
        coordinator.start()
        status = coordinator.get_sync_status()
        assert status["active"] is True

        # Test stop
        coordinator.stop()
        status = coordinator.get_sync_status()
        assert status["active"] is False

    def test_callback_registration(self) -> None:
        """Test callback registration and management."""
        coordinator = SessionSyncCoordinator()

        callback_called = False

        def test_callback(*args: Any) -> None:
            nonlocal callback_called
            callback_called = True

        # Test initial callback count
        status = coordinator.get_sync_status()
        initial_count = status["callback_counts"]["process_update"]

        # Register callback
        coordinator.register_callback("process_update", test_callback)
        status = coordinator.get_sync_status()
        assert status["callback_counts"]["process_update"] == initial_count + 1

        # Unregister callback
        success = coordinator.unregister_callback(
            "process_update", test_callback
        )
        assert success is True
        status = coordinator.get_sync_status()
        assert status["callback_counts"]["process_update"] == initial_count

    def test_sync_status(self) -> None:
        """Test getting synchronization status."""
        coordinator = SessionSyncCoordinator()

        status = coordinator.get_sync_status()

        assert "active" in status
        assert "last_process_update" in status
        assert "last_log_update" in status
        assert "callback_counts" in status
        assert status["active"] is False

    def test_update_frequency_setting(self) -> None:
        """Test setting update frequency."""
        coordinator = SessionSyncCoordinator()

        # Test valid frequency
        coordinator.set_update_frequency(2.0)
        assert coordinator._update_frequency == 2.0

        # Test frequency bounds
        coordinator.set_update_frequency(0.01)  # Too low
        assert coordinator._update_frequency == 0.1

        coordinator.set_update_frequency(20.0)  # Too high
        assert coordinator._update_frequency == 10.0

    @patch("scripts.gui.utils.session_sync.SessionStateManager")
    def test_register_with_process_manager(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Test registering with ProcessManager."""
        coordinator = SessionSyncCoordinator()
        coordinator.start()

        # Mock process manager
        mock_process_manager = MagicMock()
        mock_process_manager.process_info = MagicMock()
        mock_process_manager.get_memory_usage.return_value = {"rss": 1024.0}

        # Register with process manager
        coordinator.register_with_process_manager(mock_process_manager)

        # Verify no errors occurred during registration
        status = coordinator.get_sync_status()
        assert status["active"] is True

    @patch("scripts.gui.utils.session_sync.SessionStateManager")
    def test_register_with_log_stream_manager(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Test registering with LogStreamManager."""
        coordinator = SessionSyncCoordinator()
        coordinator.start()

        # Mock log stream manager
        mock_log_manager = MagicMock()
        mock_log_manager.get_recent_logs.return_value = []
        mock_log_manager.buffer_size = 100
        mock_log_manager.is_active = True
        mock_log_manager.add_callback = MagicMock()

        # Register with log stream manager
        coordinator.register_with_log_stream_manager(mock_log_manager)

        # Verify callback was registered
        mock_log_manager.add_callback.assert_called_once()

    def test_force_sync_all(self) -> None:
        """Test forcing synchronization of all state."""
        coordinator = SessionSyncCoordinator()

        # Set some update times
        coordinator._last_process_update = time.time()
        coordinator._last_log_update = time.time()

        # Force sync
        coordinator.force_sync_all()

        # Verify update times were reset
        assert coordinator._last_process_update == 0.0
        assert coordinator._last_log_update == 0.0


class TestSessionStateIntegration:
    """Test integration between session state components."""

    @patch("streamlit.session_state")
    def test_end_to_end_process_lifecycle(
        self, mock_st_session_state: MagicMock
    ) -> None:
        """Test end-to-end process lifecycle state management."""
        mock_state = SessionState()
        mock_st_session_state.app_state = mock_state

        # Simulate process starting
        SessionStateManager.update_from_process_info(
            self._create_mock_process_info("starting", 12345)
        )
        assert mock_state.process_state == "starting"
        assert mock_state.is_process_running() is True

        # Simulate process running
        SessionStateManager.update_from_process_info(
            self._create_mock_process_info("running", 12345)
        )
        assert mock_state.process_state == "running"
        assert mock_state.training_active is True

        # Simulate process completed
        SessionStateManager.update_from_process_info(
            self._create_mock_process_info("completed", 12345, return_code=0)
        )
        assert mock_state.process_state == "completed"
        assert mock_state.training_active is False
        assert mock_state.process_return_code == 0

    def _create_mock_process_info(
        self, state: str, pid: int, return_code: int | None = None
    ) -> MagicMock:
        """Create a mock ProcessInfo object."""
        mock_info = MagicMock()
        mock_info.state.value = state
        mock_info.pid = pid
        mock_info.command = ["python", "run.py"]
        mock_info.start_time = time.time()
        mock_info.working_directory = Path("/test")
        mock_info.return_code = return_code
        mock_info.error_message = None
        return mock_info
