"""
Unit tests for the ProgressBar component.

Tests cover progress bar functionality, step-based progress tracking,
time estimation, brand styling, and integration with session state.
"""

import time
from unittest.mock import MagicMock, patch

from scripts.gui.components.progress_bar import (
    ProgressBar,
    StepBasedProgress,
    create_progress_bar,
    create_step_progress,
)


class TestProgressBar:
    """Test cases for the ProgressBar component."""

    def test_initialization(self) -> None:
        """Test progress bar initialization."""
        # Default initialization
        progress_bar = ProgressBar()
        assert progress_bar.operation_id.startswith("progress_")
        assert progress_bar.start_time is None
        assert not progress_bar._is_active

        # Custom operation ID
        custom_id = "test_operation"
        progress_bar_custom = ProgressBar(custom_id)
        assert progress_bar_custom.operation_id == custom_id

    @patch("streamlit.empty")
    @patch("scripts.gui.utils.session_state.SessionStateManager.get")
    def test_start_operation(
        self, mock_session_state: MagicMock, mock_empty: MagicMock
    ) -> None:
        """Test starting a progress bar operation."""
        mock_state = MagicMock()
        mock_session_state.return_value = mock_state
        mock_placeholder = MagicMock()
        mock_empty.return_value = mock_placeholder

        progress_bar = ProgressBar("test_op")

        with patch.object(progress_bar, "_render_progress") as mock_render:
            progress_bar.start(
                title="Test Operation",
                total_steps=5,
                description="Testing progress bar",
            )

        # Verify state
        assert progress_bar._is_active
        assert progress_bar.title == "Test Operation"
        assert progress_bar.total_steps == 5
        assert progress_bar.description == "Testing progress bar"
        assert progress_bar.current_step == 0
        assert progress_bar.start_time is not None

        # Verify session state notification
        mock_state.add_notification.assert_called_once_with(
            "Started: Test Operation"
        )

        # Verify initial render
        mock_render.assert_called_once_with(0.0, "Initializing...")

    @patch("scripts.gui.utils.session_state.SessionStateManager.get")
    def test_update_progress(self, mock_session_state: MagicMock) -> None:
        """Test updating progress bar state."""
        progress_bar = ProgressBar("test_op")
        progress_bar._is_active = True

        with patch.object(progress_bar, "_render_progress") as mock_render:
            # Test normal update
            progress_bar.update(
                0.5, current_step=3, step_description="Processing data"
            )
            mock_render.assert_called_with(0.5, "Processing data")
            assert progress_bar.current_step == 3

            # Test progress clamping
            progress_bar.update(1.5)  # Should clamp to 1.0
            assert mock_render.call_args[0][0] == 1.0

            progress_bar.update(-0.5)  # Should clamp to 0.0
            assert mock_render.call_args[0][0] == 0.0

        # Test update when not active
        progress_bar._is_active = False
        with patch.object(progress_bar, "_render_progress") as mock_render:
            progress_bar.update(0.8)
            mock_render.assert_not_called()

    @patch("scripts.gui.utils.session_state.SessionStateManager.get")
    @patch("time.sleep")
    def test_finish_operation(
        self, mock_sleep: MagicMock, mock_session_state: MagicMock
    ) -> None:
        """Test finishing a progress bar operation."""
        mock_state = MagicMock()
        mock_session_state.return_value = mock_state

        progress_bar = ProgressBar("test_op")
        progress_bar._is_active = True
        progress_bar.title = "Test Operation"
        progress_bar.start_time = time.time() - 5.0  # 5 seconds ago

        with patch.object(progress_bar, "_render_progress") as mock_render:
            progress_bar.finish("Custom success message")

        # Verify completion render
        mock_render.assert_called_once_with(
            1.0, "Custom success message", state="success"
        )

        # Verify session state notification
        mock_state.add_notification.assert_called_once()
        notification_call = mock_state.add_notification.call_args[0][0]
        assert "Completed: Test Operation" in notification_call
        assert "s)" in notification_call

        # Verify state
        assert not progress_bar._is_active

        # Verify sleep for completion display
        mock_sleep.assert_called_once_with(1)

    def test_format_time(self) -> None:
        """Test time formatting utility."""
        # Test seconds
        assert ProgressBar._format_time(30) == "30s"
        assert ProgressBar._format_time(59) == "59s"

        # Test minutes
        assert ProgressBar._format_time(60) == "1m 0s"
        assert ProgressBar._format_time(125) == "2m 5s"
        assert ProgressBar._format_time(3599) == "59m 59s"

        # Test hours
        assert ProgressBar._format_time(3600) == "1h 0m"
        assert ProgressBar._format_time(3665) == "1h 1m"
        assert ProgressBar._format_time(7325) == "2h 2m"

    @patch("streamlit.markdown")
    def test_render_progress(self, mock_markdown: MagicMock) -> None:
        """Test progress bar rendering."""
        progress_bar = ProgressBar("test_op")
        progress_bar.title = "Test Operation"
        progress_bar.description = "Test description"
        progress_bar.total_steps = 5
        progress_bar.current_step = 2
        progress_bar.start_time = time.time() - 10.0  # 10 seconds ago
        progress_bar._placeholder = MagicMock()

        # Test normal render
        progress_bar._render_progress(0.4, "Processing step", state="normal")

        # Verify CSS injection was called
        assert mock_markdown.call_count >= 1

        # Verify the progress HTML was sent to placeholder
        progress_bar._placeholder.markdown.assert_called_once()
        placeholder_call = progress_bar._placeholder.markdown.call_args

        # Check that it was called with unsafe_allow_html=True
        assert placeholder_call[1]["unsafe_allow_html"] is True

        # Check the HTML content
        progress_html = placeholder_call[0][0]
        assert "Test Operation" in progress_html
        assert "40.0%" in progress_html
        assert "Step 2/5" in progress_html
        assert "Processing step" in progress_html
        assert (
            "crackseg-progress-container crackseg-progress-normal"
            in progress_html
        )

    @patch("streamlit.markdown")
    def test_render_progress_success_state(
        self, mock_markdown: MagicMock
    ) -> None:
        """Test progress bar rendering in success state."""
        progress_bar = ProgressBar("test_op")
        progress_bar.title = "Completed Operation"
        progress_bar.start_time = time.time() - 5.0
        progress_bar._placeholder = MagicMock()

        progress_bar._render_progress(1.0, "Success!", state="success")

        # Verify CSS injection was called
        assert mock_markdown.call_count >= 1

        # Verify the progress HTML was sent to placeholder
        progress_bar._placeholder.markdown.assert_called_once()
        placeholder_call = progress_bar._placeholder.markdown.call_args

        # Check the HTML content
        progress_html = placeholder_call[0][0]
        assert "crackseg-progress-success" in progress_html
        assert "100.0%" in progress_html
        assert "Success!" in progress_html
        assert "Completed Operation" in progress_html

    def test_create_step_based_progress(self) -> None:
        """Test creating step-based progress tracker."""
        steps = ["Step 1", "Step 2", "Step 3"]
        step_progress = ProgressBar.create_step_based_progress(
            "Multi-step Operation", steps, "test_id"
        )

        assert isinstance(step_progress, StepBasedProgress)
        assert step_progress.progress_bar.operation_id == "test_id"
        assert step_progress.steps == steps
        assert step_progress.title == "Multi-step Operation"

    @patch("streamlit.empty")
    @patch("scripts.gui.utils.session_state.SessionStateManager.get")
    def test_brand_colors_consistency(
        self, mock_session_state: MagicMock, mock_empty: MagicMock
    ) -> None:
        """Test that brand colors match LoadingSpinner."""
        from scripts.gui.components.loading_spinner import LoadingSpinner

        # Compare brand colors
        progress_colors = ProgressBar._BRAND_COLORS
        spinner_colors = LoadingSpinner._BRAND_COLORS

        assert progress_colors["primary"] == spinner_colors["primary"]
        assert progress_colors["accent"] == spinner_colors["accent"]
        assert progress_colors["success"] == spinner_colors["success"]
        assert progress_colors["warning"] == spinner_colors["warning"]
        assert progress_colors["error"] == spinner_colors["error"]


class TestStepBasedProgress:
    """Test cases for the StepBasedProgress component."""

    def test_initialization(self) -> None:
        """Test step-based progress initialization."""
        steps = ["Initialize", "Process", "Finalize"]
        step_progress = StepBasedProgress("Test Operation", steps, "test_id")

        assert step_progress.title == "Test Operation"
        assert step_progress.steps == steps
        assert step_progress.current_step_index == 0
        assert step_progress.progress_bar.operation_id == "test_id"

    @patch("scripts.gui.utils.session_state.SessionStateManager.get")
    @patch("streamlit.empty")
    def test_context_manager_success(
        self, mock_empty: MagicMock, mock_session_state: MagicMock
    ) -> None:
        """Test step-based progress as context manager with success."""
        mock_session_state.return_value = MagicMock()
        mock_empty.return_value = MagicMock()

        steps = ["Step 1", "Step 2"]

        with patch("time.sleep"):  # Mock sleep in finish()
            with StepBasedProgress("Test", steps) as progress:
                # Verify initialization
                assert progress.progress_bar._is_active

                # Test next_step
                with patch.object(
                    progress.progress_bar, "update"
                ) as mock_update:
                    progress.next_step()
                    mock_update.assert_called_with(
                        progress=0.5, current_step=1, step_description="Step 1"
                    )
                    assert progress.current_step_index == 1

            # Verify completion
            assert not progress.progress_bar._is_active

    @patch("scripts.gui.utils.session_state.SessionStateManager.get")
    @patch("streamlit.empty")
    def test_context_manager_exception(
        self, mock_empty: MagicMock, mock_session_state: MagicMock
    ) -> None:
        """Test step-based progress context manager with exception."""
        mock_session_state.return_value = MagicMock()
        mock_empty.return_value = MagicMock()

        steps = ["Step 1", "Step 2"]

        with patch("time.sleep"):  # Mock sleep in finish()
            step_progress = StepBasedProgress("Test", steps)

            # Apply mock to the progress_bar before entering context
            with patch.object(
                step_progress.progress_bar, "finish"
            ) as mock_finish:
                try:
                    with step_progress:
                        raise ValueError("Test error")
                except ValueError:
                    pass

                # Verify finish was called with enhanced failure message
                mock_finish.assert_called_once_with(
                    "Operation encountered an error"
                )

    def test_next_step_with_custom_description(self) -> None:
        """Test next_step with custom description."""
        steps = ["Step 1", "Step 2", "Step 3"]
        step_progress = StepBasedProgress("Test", steps)
        step_progress.progress_bar._is_active = True

        with patch.object(step_progress.progress_bar, "update") as mock_update:
            step_progress.next_step("Custom description")

            mock_update.assert_called_with(
                progress=1 / 3,
                current_step=1,
                step_description="Custom description",
            )

    def test_set_step_progress(self) -> None:
        """Test setting progress for specific step."""
        steps = ["Step 1", "Step 2", "Step 3"]
        step_progress = StepBasedProgress("Test", steps)
        step_progress.progress_bar._is_active = True

        with patch.object(step_progress.progress_bar, "update") as mock_update:
            # Test mid-step progress
            step_progress.set_step_progress(1, 0.5)  # 50% through step 2

            expected_progress = (1 + 0.5) / 3  # 1.5/3 = 0.5
            mock_update.assert_called_with(
                progress=expected_progress,
                current_step=2,
                step_description="Step 2",
            )

            # Test invalid step index
            step_progress.set_step_progress(10, 0.5)  # Should be ignored
            # Should still have the previous call count
            assert mock_update.call_count == 1


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_create_progress_bar(self) -> None:
        """Test create_progress_bar convenience function."""
        # Default creation
        progress_bar = create_progress_bar()
        assert isinstance(progress_bar, ProgressBar)
        assert progress_bar.operation_id.startswith("progress_")

        # Custom operation ID
        custom_bar = create_progress_bar("custom_id")
        assert custom_bar.operation_id == "custom_id"

    def test_create_step_progress(self) -> None:
        """Test create_step_progress convenience function."""
        steps = ["Step A", "Step B", "Step C"]

        # Default creation
        step_progress = create_step_progress("Test Operation", steps)
        assert isinstance(step_progress, StepBasedProgress)
        assert step_progress.title == "Test Operation"
        assert step_progress.steps == steps

        # Custom operation ID
        custom_progress = create_step_progress(
            "Custom Operation", steps, "custom_id"
        )
        assert custom_progress.progress_bar.operation_id == "custom_id"


class TestProgressBarIntegration:
    """Integration tests for ProgressBar with Streamlit components."""

    @patch("streamlit.markdown")
    @patch("streamlit.empty")
    @patch("scripts.gui.utils.session_state.SessionStateManager.get")
    def test_full_operation_cycle(
        self,
        mock_session_state: MagicMock,
        mock_empty: MagicMock,
        mock_markdown: MagicMock,
    ) -> None:
        """Test complete progress bar operation cycle."""
        mock_state = MagicMock()
        mock_session_state.return_value = mock_state
        mock_placeholder = MagicMock()
        mock_empty.return_value = mock_placeholder

        progress_bar = ProgressBar("integration_test")

        with patch("time.sleep"):  # Mock sleep in finish()
            # Start operation
            progress_bar.start(
                title="Integration Test",
                total_steps=3,
                description="Testing full cycle",
            )

            # Update progress multiple times
            progress_bar.update(
                0.33, current_step=1, step_description="First step"
            )
            progress_bar.update(
                0.66, current_step=2, step_description="Second step"
            )
            progress_bar.update(
                1.0, current_step=3, step_description="Final step"
            )

            # Finish operation
            progress_bar.finish("Integration test completed")

        # Verify session state calls
        assert mock_state.add_notification.call_count == 2  # Start + finish

        # Verify CSS was injected
        css_calls = [
            call
            for call in mock_markdown.call_args_list
            if "crackseg-progress-container" in str(call)
        ]
        assert len(css_calls) > 0

    @patch("streamlit.empty")
    @patch("scripts.gui.utils.session_state.SessionStateManager.get")
    def test_time_estimation_accuracy(
        self, mock_session_state: MagicMock, mock_empty: MagicMock
    ) -> None:
        """Test time estimation calculations."""
        mock_session_state.return_value = MagicMock()
        mock_empty.return_value = MagicMock()

        progress_bar = ProgressBar("time_test")

        # Mock start time to control calculations
        mock_start_time = time.time() - 10.0  # 10 seconds ago
        progress_bar.start_time = mock_start_time
        progress_bar._placeholder = MagicMock()

        with patch.object(progress_bar, "_format_time") as mock_format:
            mock_format.side_effect = lambda x: f"{x:.1f}s"

            # Test at 50% completion after 10 seconds
            progress_bar._render_progress(0.5, "Halfway")

            # Should estimate 10 more seconds (total 20s, 50% done = 10s
            # remaining)
            format_calls = mock_format.call_args_list
            assert len(format_calls) >= 2  # elapsed + remaining time calls
