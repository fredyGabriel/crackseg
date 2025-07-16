"""
Test cases for TensorBoard components with low coverage.

This module implements missing test cases for TensorBoard-related components
to improve coverage for process management, port handling, and rendering.

Areas covered:
1. TensorBoard process lifecycle management
2. Port allocation and conflict resolution
3. Rendering components error handling
4. Session and state management
5. Recovery strategies and error handling
"""

import subprocess
from typing import Any
from unittest.mock import Mock, patch

import pytest

from scripts.gui.components.tensorboard.component import TensorBoardComponent
from scripts.gui.components.tensorboard.state.session_manager import (
    SessionStateManager,
)
from scripts.gui.utils.tensorboard.lifecycle_manager import (
    TensorBoardLifecycleManager,
)
from scripts.gui.utils.tensorboard.port_management import PortManager
from scripts.gui.utils.tensorboard.process_manager import (
    TensorBoardProcessManager,
)


class TestTensorBoardProcessManagement:
    """Test uncovered TensorBoard process management functionality."""

    @patch("subprocess.Popen")
    def test_tensorboard_process_start_success(self, mock_popen: Any) -> None:
        """Test successful TensorBoard process startup."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process running
        mock_popen.return_value = mock_process

        manager = TensorBoardProcessManager()

        result = manager.start_tensorboard("/tmp/logs", port=6006)

        assert result is True
        assert manager.is_running()
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_tensorboard_process_start_failure(self, mock_popen: Any) -> None:
        """Test TensorBoard process startup failure."""
        mock_popen.side_effect = subprocess.SubprocessError("Failed to start")

        manager = TensorBoardProcessManager()

        result = manager.start_tensorboard("/tmp/logs", port=6006)

        assert result is False
        assert not manager.is_running()

    @patch("subprocess.Popen")
    def test_tensorboard_process_stop_graceful(self, mock_popen: Any) -> None:
        """Test graceful TensorBoard process termination."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.terminate.return_value = None
        mock_popen.return_value = mock_process

        manager = TensorBoardProcessManager()
        manager.start_tensorboard("/tmp/logs", port=6006)

        result = manager.stop_tensorboard()

        assert result is True
        mock_process.terminate.assert_called_once()

    @patch("subprocess.Popen")
    def test_tensorboard_process_force_kill(self, mock_popen: Any) -> None:
        """Test force killing unresponsive TensorBoard process."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.terminate.side_effect = ProcessLookupError(
            "Process not found"
        )
        mock_process.kill.return_value = None
        mock_popen.return_value = mock_process

        manager = TensorBoardProcessManager()
        manager.start_tensorboard("/tmp/logs", port=6006)

        result = manager.force_stop_tensorboard()

        assert result is True
        mock_process.kill.assert_called_once()


class TestTensorBoardPortManagement:
    """Test uncovered TensorBoard port management functionality."""

    @patch("socket.socket")
    def test_port_availability_check_free(self, mock_socket: Any) -> None:
        """Test port availability when port is free."""
        mock_sock = Mock()
        mock_sock.bind.return_value = None
        mock_socket.return_value.__enter__.return_value = mock_sock

        port_manager = PortManager()

        is_available = port_manager.is_port_available(6006)

        assert is_available is True
        mock_sock.bind.assert_called_once_with(("localhost", 6006))

    @patch("socket.socket")
    def test_port_availability_check_occupied(self, mock_socket):
        """Test port availability when port is occupied."""
        mock_sock = Mock()
        mock_sock.bind.side_effect = OSError("Address already in use")
        mock_socket.return_value.__enter__.return_value = mock_sock

        port_manager = PortManager()

        is_available = port_manager.is_port_available(6006)

        assert is_available is False

    @patch("socket.socket")
    def test_port_allocation_find_free(self, mock_socket):
        """Test automatic port allocation when ports are occupied."""
        mock_sock = Mock()
        # First two ports occupied, third free
        mock_sock.bind.side_effect = [
            OSError("Address already in use"),  # 6006
            OSError("Address already in use"),  # 6007
            None,  # 6008 free
        ]
        mock_socket.return_value.__enter__.return_value = mock_sock

        port_manager = PortManager()

        allocated_port = port_manager.allocate_port(start_port=6006)

        assert allocated_port == 6008

    def test_port_allocation_range_exhausted(self):
        """Test port allocation when range is exhausted."""
        with patch.object(
            PortManager, "is_port_available", return_value=False
        ):
            port_manager = PortManager()

            allocated_port = port_manager.allocate_port(
                start_port=6006, max_attempts=3
            )

            assert allocated_port is None


class TestTensorBoardRendering:
    """Test uncovered TensorBoard rendering functionality."""

    @patch("streamlit.iframe")
    def test_iframe_renderer_success(self, mock_iframe):
        """Test successful iframe rendering."""
        from scripts.gui.components.tensorboard.rendering import (
            iframe_renderer,
        )

        IFrameRenderer = iframe_renderer.IFrameRenderer

        renderer = IFrameRenderer()

        renderer.render_tensorboard_iframe("http://localhost:6006", height=600)

        mock_iframe.assert_called_once_with(
            "http://localhost:6006", height=600
        )

    @patch("streamlit.error")
    def test_iframe_renderer_invalid_url(self, mock_error):
        """Test iframe rendering with invalid URL."""
        from scripts.gui.components.tensorboard.rendering import (
            iframe_renderer,
        )

        IFrameRenderer = iframe_renderer.IFrameRenderer

        renderer = IFrameRenderer()

        renderer.render_tensorboard_iframe("invalid-url", height=600)

        mock_error.assert_called_once()

    @patch("streamlit.metric")
    def test_status_card_health_display(self, mock_metric):
        """Test health status card display."""
        from scripts.gui.components.tensorboard.rendering.status_cards import (
            health_card,
        )

        HealthCard = health_card.HealthCard

        card = HealthCard()

        card.render_health_status(status="healthy", uptime="2h 30m")

        mock_metric.assert_called()

    @patch("streamlit.metric")
    def test_status_card_network_display(self, mock_metric):
        """Test network status card display."""
        from scripts.gui.components.tensorboard.rendering.status_cards import (
            network_card,
        )

        NetworkCard = network_card.NetworkCard

        card = NetworkCard()

        card.render_network_status(port=6006, url="http://localhost:6006")

        mock_metric.assert_called()

    @patch("streamlit.error")
    def test_error_renderer_display(self, mock_error):
        """Test error renderer functionality."""
        from scripts.gui.components.tensorboard.rendering import error_renderer

        ErrorRenderer = error_renderer.ErrorRenderer

        renderer = ErrorRenderer()

        error_msg = "TensorBoard failed to start"
        renderer.render_error(error_msg, show_details=True)

        mock_error.assert_called()


class TestTensorBoardLifecycleManagement:
    """Test uncovered TensorBoard lifecycle management functionality."""

    def test_lifecycle_manager_initialization(self):
        """Test lifecycle manager initialization."""
        manager = TensorBoardLifecycleManager()

        assert manager.get_status() == "stopped"
        assert not manager.is_active()

    @patch.object(TensorBoardProcessManager, "start_tensorboard")
    @patch.object(PortManager, "allocate_port")
    def test_lifecycle_manager_start_success(self, mock_allocate, mock_start):
        """Test successful lifecycle startup."""
        mock_allocate.return_value = 6006
        mock_start.return_value = True

        manager = TensorBoardLifecycleManager()

        result = manager.start("/tmp/logs")

        assert result is True
        assert manager.get_status() == "running"
        assert manager.is_active()

    @patch.object(TensorBoardProcessManager, "start_tensorboard")
    @patch.object(PortManager, "allocate_port")
    def test_lifecycle_manager_start_failure(self, mock_allocate, mock_start):
        """Test lifecycle startup failure."""
        mock_allocate.return_value = 6006
        mock_start.return_value = False

        manager = TensorBoardLifecycleManager()

        result = manager.start("/tmp/logs")

        assert result is False
        assert manager.get_status() == "error"
        assert not manager.is_active()

    @patch.object(TensorBoardProcessManager, "stop_tensorboard")
    def test_lifecycle_manager_stop_success(self, mock_stop):
        """Test successful lifecycle shutdown."""
        mock_stop.return_value = True

        manager = TensorBoardLifecycleManager()
        manager._status = "running"  # Set initial state

        result = manager.stop()

        assert result is True
        assert manager.get_status() == "stopped"
        assert not manager.is_active()


class TestTensorBoardSessionManagement:
    """Test uncovered TensorBoard session management functionality."""

    def test_session_manager_initialization(self):
        """Test session manager initialization."""
        session_manager = SessionStateManager()

        # Test that initial state is properly set up
        state = session_manager.get_state()
        assert "startup_attempted" in state
        assert state["startup_attempted"] is False

    def test_session_manager_update_status(self):
        """Test session status updates."""
        session_manager = SessionStateManager()

        # Test state value updates
        session_manager.set_value("startup_attempted", True)
        assert session_manager.get_value("startup_attempted") is True

        session_manager.set_value("error_message", "test error")
        assert session_manager.get_value("error_message") == "test error"

    def test_session_manager_log_entry(self):
        """Test session error state management."""
        session_manager = SessionStateManager()

        # Test error state management
        session_manager.set_error("Test error", "connection_error")

        assert session_manager.has_error() is True
        assert session_manager.get_value("error_message") == "Test error"
        assert session_manager.get_value("error_type") == "connection_error"

    def test_session_manager_reset_session(self):
        """Test session reset functionality."""
        session_manager = SessionStateManager()

        # Add some state
        session_manager.set_value("startup_attempted", True)
        session_manager.set_error("Test error", "test_type")

        # Reset session
        session_manager.reset_session()

        assert session_manager.get_status() == "stopped"
        assert len(session_manager.get_logs()) == 0


class TestTensorBoardComponent:
    """Test uncovered TensorBoard component functionality."""

    @patch.object(TensorBoardLifecycleManager, "start")
    def test_component_start_tensorboard(self, mock_start):
        """Test component TensorBoard startup."""
        mock_start.return_value = True

        component = TensorBoardComponent()

        result = component.start_tensorboard("/tmp/logs")

        assert result is True
        mock_start.assert_called_once_with("/tmp/logs")

    @patch.object(TensorBoardLifecycleManager, "stop")
    def test_component_stop_tensorboard(self, mock_stop):
        """Test component TensorBoard shutdown."""
        mock_stop.return_value = True

        component = TensorBoardComponent()

        result = component.stop_tensorboard()

        assert result is True
        mock_stop.assert_called_once()

    @patch("streamlit.container")
    @patch("streamlit.button")
    def test_component_render_controls(self, mock_button, mock_container):
        """Test component control rendering."""
        mock_button.return_value = False
        mock_container.return_value.__enter__.return_value = Mock()

        component = TensorBoardComponent()

        component.render_tensorboard_controls()

        # Should render start/stop buttons
        assert mock_button.call_count >= 2

    @patch("streamlit.info")
    def test_component_render_status_info(self, mock_info):
        """Test component status information rendering."""
        component = TensorBoardComponent()

        component.render_status_info("running", port=6006)

        mock_info.assert_called_once()

    @patch("streamlit.error")
    def test_component_render_status_error(self, mock_error):
        """Test component error status rendering."""
        component = TensorBoardComponent()

        component.render_status_info("error", error_msg="Failed to start")

        mock_error.assert_called_once()


class TestTensorBoardDiagnostics:
    """Test uncovered TensorBoard diagnostics functionality."""

    @patch("streamlit.columns")
    @patch("streamlit.button")
    def test_diagnostic_action_controls(self, mock_button, mock_columns):
        """Test diagnostic action controls rendering."""
        from scripts.gui.components.tensorboard.rendering.diagnostics import (
            action_controls,
        )

        ActionControls = action_controls.ActionControls

        mock_columns.return_value = [Mock(), Mock(), Mock()]
        mock_button.return_value = False

        controls = ActionControls()

        controls.render_diagnostic_actions()

        # Should render multiple action buttons
        assert mock_button.call_count >= 3

    @patch("streamlit.expander")
    def test_diagnostic_panel_display(self, mock_expander):
        """Test diagnostic panel display."""
        from scripts.gui.components.tensorboard.rendering.diagnostics import (
            diagnostic_panel,
        )

        DiagnosticPanel = diagnostic_panel.DiagnosticPanel

        mock_expander.return_value.__enter__.return_value = Mock()

        panel = DiagnosticPanel()

        diagnostics_data = {
            "port_status": "available",
            "process_status": "running",
            "log_entries": ["Info: Started successfully"],
        }

        panel.render_diagnostics(diagnostics_data)

        mock_expander.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
