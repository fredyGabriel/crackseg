"""
Unit tests for the LoadingSpinner component.

Tests cover spinner functionality, styling, timeout handling, and contextual
messaging.
"""

import time
from unittest.mock import Mock, patch

import pytest

from scripts.gui.components.loading_spinner import (
    LoadingSpinner,
    loading_spinner,
)


class TestLoadingSpinner:
    """Test suite for LoadingSpinner component."""

    def test_brand_colors_constants(self):
        """Test that brand colors are properly defined."""
        assert LoadingSpinner._BRAND_COLORS["primary"] == "#2E2E2E"
        assert LoadingSpinner._BRAND_COLORS["accent"] == "#FF4444"
        assert LoadingSpinner._BRAND_COLORS["success"] == "#00FF64"
        assert LoadingSpinner._BRAND_COLORS["warning"] == "#FFB800"

    def test_spinner_styles_constants(self):
        """Test that spinner styles are properly configured."""
        assert "crack_pattern" in LoadingSpinner._SPINNER_STYLES
        assert "road_analysis" in LoadingSpinner._SPINNER_STYLES
        assert "ai_processing" in LoadingSpinner._SPINNER_STYLES
        assert "default" in LoadingSpinner._SPINNER_STYLES

        # Test structure of spinner styles
        for (
            _style_name,
            style_config,
        ) in LoadingSpinner._SPINNER_STYLES.items():
            assert "animation" in style_config
            assert "color" in style_config
            assert "description" in style_config

    @patch("streamlit.markdown")
    def test_inject_custom_css(self, mock_markdown: Mock):
        """Test CSS injection for custom styling."""
        LoadingSpinner._inject_custom_css()

        # Verify CSS was injected
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args
        css_content = call_args[0][0]

        # Check for key CSS classes
        assert "crackseg-spinner-container" in css_content
        assert "crackseg-spinner-icon" in css_content
        assert "crackseg-rotate" in css_content
        assert "crackseg-timeout-warning" in css_content
        assert "crackseg-success-message" in css_content

        # Check unsafe_allow_html was used
        assert call_args[1]["unsafe_allow_html"] is True

    @patch("streamlit.markdown")
    def test_render_custom_spinner_basic(self, mock_markdown: Mock):
        """Test basic custom spinner rendering."""
        message = "Test loading message"

        LoadingSpinner._render_custom_spinner(message)

        # Verify HTML was rendered
        assert mock_markdown.call_count >= 1

        # Check that message appears in rendered HTML
        html_calls = [call[0][0] for call in mock_markdown.call_args_list]
        message_found = any(message in html for html in html_calls)
        assert message_found

    @patch("streamlit.markdown")
    def test_render_custom_spinner_with_subtext(self, mock_markdown: Mock):
        """Test custom spinner rendering with subtext."""
        message = "Primary message"
        subtext = "Secondary details"

        LoadingSpinner._render_custom_spinner(message, subtext)

        # Check that both message and subtext appear in rendered HTML
        html_calls = [call[0][0] for call in mock_markdown.call_args_list]
        combined_html = " ".join(html_calls)

        assert message in combined_html
        assert subtext in combined_html

    def test_get_contextual_message_config(self):
        """Test contextual message for config operations."""
        message, subtext, spinner_type = LoadingSpinner.get_contextual_message(
            "config"
        )

        assert "configuration" in message.lower()
        assert "yaml" in subtext.lower()
        assert spinner_type == "default"

    def test_get_contextual_message_model(self):
        """Test contextual message for model operations."""
        message, subtext, spinner_type = LoadingSpinner.get_contextual_message(
            "model"
        )

        assert "model" in message.lower()
        assert "encoder" in subtext.lower()
        assert spinner_type == "ai_processing"

    def test_get_contextual_message_training(self):
        """Test contextual message for training operations."""
        message, subtext, spinner_type = LoadingSpinner.get_contextual_message(
            "training"
        )

        assert "training" in message.lower()
        assert "data" in subtext.lower()
        assert spinner_type == "crack_pattern"

    def test_get_contextual_message_unknown(self):
        """Test contextual message for unknown operations."""
        message, subtext, spinner_type = LoadingSpinner.get_contextual_message(
            "unknown"
        )

        assert "processing" in message.lower()
        assert "please wait" in subtext.lower()
        assert spinner_type == "default"

    @patch("scripts.gui.components.loading_spinner.SessionStateManager")
    @patch("streamlit.spinner")
    @patch("streamlit.empty")
    def test_spinner_context_manager_basic(
        self, mock_empty: Mock, mock_spinner: Mock, mock_session_manager: Mock
    ):
        """Test basic spinner context manager functionality."""
        # Setup mocks
        mock_session_state = Mock()
        mock_session_manager.get.return_value = mock_session_state
        mock_placeholder = Mock()
        mock_empty.return_value = mock_placeholder

        # Test the context manager
        with LoadingSpinner.spinner("Test message", show_custom_ui=False):
            pass

        # Verify session state was updated
        mock_session_state.add_notification.assert_called()

        # Verify spinner was called
        mock_spinner.assert_called_once_with("Test message")

    @patch("scripts.gui.components.loading_spinner.SessionStateManager")
    @patch("streamlit.spinner")
    @patch("streamlit.empty")
    @patch("time.time")
    def test_spinner_timeout_handling(
        self,
        mock_time: Mock,
        mock_empty: Mock,
        mock_spinner: Mock,
        mock_session_manager: Mock,
    ):
        """Test spinner timeout warning functionality."""
        # Setup mocks for timeout scenario
        mock_session_state = Mock()
        mock_session_manager.get.return_value = mock_session_state
        mock_placeholder = Mock()
        mock_empty.return_value = mock_placeholder

        # Mock the container context manager
        mock_container = Mock()
        mock_container.__enter__ = Mock(return_value=mock_container)
        mock_container.__exit__ = Mock(return_value=None)
        mock_placeholder.container.return_value = mock_container

        # Mock time to simulate timeout (start at 0, end at 35 seconds)
        mock_time.side_effect = [
            0,
            35,
            35,
            35,
        ]  # Additional calls for finally block

        # Mock the st.spinner context manager
        mock_spinner_context = Mock()
        mock_spinner_context.__enter__ = Mock(
            return_value=mock_spinner_context
        )
        mock_spinner_context.__exit__ = Mock(return_value=None)
        mock_spinner.return_value = mock_spinner_context

        with patch("streamlit.markdown") as mock_markdown:
            with LoadingSpinner.spinner("Test message", timeout_seconds=30):
                pass  # Simulate operation

            # Verify timeout warning was displayed
            mock_markdown.assert_called()
            call_args = mock_markdown.call_args
            assert "Operation took longer than expected" in call_args[0][0]
            assert "crackseg-timeout-warning" in call_args[0][0]

    @patch("scripts.gui.components.loading_spinner.SessionStateManager")
    @patch("streamlit.spinner")
    @patch("streamlit.empty")
    def test_spinner_error_handling(
        self, mock_empty: Mock, mock_spinner: Mock, mock_session_manager: Mock
    ):
        """Test spinner error handling with enhanced error messaging."""
        # Setup mocks
        mock_session_state = Mock()
        mock_session_manager.get.return_value = mock_session_state
        mock_placeholder = Mock()
        mock_empty.return_value = mock_placeholder

        # Test error during spinner operation with enhanced error handling
        with patch("streamlit.markdown") as mock_markdown:
            with patch("streamlit.info") as mock_info:
                with pytest.raises(ValueError):
                    with LoadingSpinner.spinner(
                        "Test message", show_custom_ui=False
                    ):
                        raise ValueError("Test error")

        # Verify enhanced error handling was used
        # The new system uses st.markdown() for error display and st.info()
        # for retry options
        assert mock_markdown.call_count >= 1 or mock_info.call_count >= 1
        mock_session_state.add_notification.assert_called()

    @patch("streamlit.columns")
    @patch("streamlit.markdown")
    @patch("streamlit.progress")
    @patch("streamlit.caption")
    def test_show_progress_with_spinner(
        self,
        mock_caption: Mock,
        mock_progress: Mock,
        mock_markdown: Mock,
        mock_columns: Mock,
    ):
        """Test progress display with spinner."""
        # Setup mocks with proper context manager support
        mock_col1 = Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)

        mock_col2 = Mock()
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)

        mock_columns.return_value = [mock_col1, mock_col2]

        # Test progress display
        LoadingSpinner.show_progress_with_spinner(
            "Test progress", 0.5, "Test subtext"
        )

        # Verify components were called
        mock_columns.assert_called_once_with([1, 4])
        mock_markdown.assert_called()
        mock_caption.assert_called_once_with("Test subtext")
        mock_progress.assert_called_once_with(0.5)

    def test_convenience_function(self):
        """Test the convenience loading_spinner function."""
        # Test that the function returns a context manager
        result = loading_spinner("Test message")

        # Should be a context manager (has __enter__ and __exit__)
        assert hasattr(result, "__enter__")
        assert hasattr(result, "__exit__")


class TestLoadingSpinnerIntegration:
    """Integration tests for LoadingSpinner component."""

    @patch("scripts.gui.components.loading_spinner.SessionStateManager")
    @patch("streamlit.spinner")
    @patch("streamlit.empty")
    @patch("time.sleep")
    def test_real_world_config_loading(
        self,
        mock_sleep: Mock,
        mock_empty: Mock,
        mock_spinner: Mock,
        mock_session_manager: Mock,
    ):
        """Test realistic config loading scenario."""
        # Setup mocks
        mock_session_state = Mock()
        mock_session_manager.get.return_value = mock_session_state
        mock_placeholder = Mock()
        mock_empty.return_value = mock_placeholder

        # Test realistic usage
        message, subtext, spinner_type = LoadingSpinner.get_contextual_message(
            "config"
        )

        with LoadingSpinner.spinner(
            message, subtext, spinner_type, show_custom_ui=False
        ):
            # Simulate config loading work
            mock_sleep(0.1)

        # Verify appropriate session state updates
        assert (
            mock_session_state.add_notification.call_count >= 2
        )  # Start and end notifications

    @patch("scripts.gui.components.loading_spinner.SessionStateManager")
    @patch("streamlit.spinner")
    @patch("streamlit.empty")
    def test_training_workflow_integration(
        self, mock_empty: Mock, mock_spinner: Mock, mock_session_manager: Mock
    ):
        """Test integration with training workflow."""
        # Setup mocks
        mock_session_state = Mock()
        mock_session_manager.get.return_value = mock_session_state
        mock_placeholder = Mock()
        mock_empty.return_value = mock_placeholder

        # Test training workflow
        message, subtext, spinner_type = LoadingSpinner.get_contextual_message(
            "training"
        )

        with LoadingSpinner.spinner(
            message, subtext, spinner_type, show_custom_ui=False
        ):
            # Simulate training initialization
            pass

        # Verify training-specific notifications
        notifications = [
            call[0][0]
            for call in mock_session_state.add_notification.call_args_list
        ]
        training_notifications = [
            n for n in notifications if "training" in n.lower()
        ]
        assert len(training_notifications) > 0

    def test_all_operation_types_have_messages(self):
        """Test that all expected operation types have contextual messages."""
        operation_types = [
            "config",
            "model",
            "training",
            "results",
            "tensorboard",
            "export",
        ]

        for op_type in operation_types:
            message, subtext, spinner_type = (
                LoadingSpinner.get_contextual_message(op_type)
            )

            # All should return valid strings
            assert isinstance(message, str) and len(message) > 0
            assert isinstance(subtext, str) and len(subtext) > 0
            assert isinstance(spinner_type, str) and len(spinner_type) > 0
            assert spinner_type in LoadingSpinner._SPINNER_STYLES

    def test_spinner_type_consistency(self):
        """Test that spinner types are consistent across the component."""
        # Get all spinner types from styles
        defined_styles = set(LoadingSpinner._SPINNER_STYLES.keys())

        # Get all spinner types used in contextual messages
        operation_types = [
            "config",
            "model",
            "training",
            "results",
            "tensorboard",
            "export",
        ]
        used_styles = set()

        for op_type in operation_types:
            _, _, spinner_type = LoadingSpinner.get_contextual_message(op_type)
            used_styles.add(spinner_type)

        # All used styles should be defined
        assert used_styles.issubset(defined_styles)


# Performance tests (if needed)
class TestLoadingSpinnerPerformance:
    """Performance tests for LoadingSpinner component."""

    @patch("scripts.gui.components.loading_spinner.SessionStateManager")
    @patch("streamlit.spinner")
    @patch("streamlit.empty")
    def test_spinner_performance_overhead(
        self, mock_empty: Mock, mock_spinner: Mock, mock_session_manager: Mock
    ):
        """Test that spinner doesn't add significant overhead."""
        # Setup mocks
        mock_session_state = Mock()
        mock_session_manager.get.return_value = mock_session_state
        mock_placeholder = Mock()
        mock_empty.return_value = mock_placeholder

        start_time = time.time()

        # Test multiple spinner operations
        for i in range(10):
            with LoadingSpinner.spinner(f"Test {i}", show_custom_ui=False):
                pass

        elapsed = time.time() - start_time

        # Should complete quickly (under 1 second for 10 operations)
        assert elapsed < 1.0
