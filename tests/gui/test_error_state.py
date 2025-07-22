"""
Tests for error state management system. This module tests the error
handling functionality for GUI components, including error
classification, message generation, and user interaction.
"""

from unittest.mock import MagicMock, patch

import pytest

from gui.utils.error_state import (
    ErrorInfo,
    ErrorMessageFactory,
    ErrorType,
    StandardErrorState,
)


class TestErrorInfo:
    """Test ErrorInfo dataclass functionality."""

    def test_error_info_creation(self) -> None:
        """Test basic ErrorInfo creation."""
        error_info = ErrorInfo(
            error_type=ErrorType.CONFIG_INVALID,
            title="Test Error",
            message="Test message",
        )

        assert error_info.error_type == ErrorType.CONFIG_INVALID
        assert error_info.title == "Test Error"
        assert error_info.message == "Test message"
        assert error_info.retry_possible is True

    def test_error_info_validation(self) -> None:
        """Test ErrorInfo validation."""
        # Empty title should raise error
        with pytest.raises(ValueError, match="Error title cannot be empty"):
            ErrorInfo(
                error_type=ErrorType.UNEXPECTED,
                title="",
                message="Test message",
            )

        # Empty message should raise error
        with pytest.raises(ValueError, match="Error message cannot be empty"):
            ErrorInfo(
                error_type=ErrorType.UNEXPECTED,
                title="Test title",
                message="",
            )


class TestErrorMessageFactory:
    """Test ErrorMessageFactory functionality."""

    def test_create_error_info_config_invalid(self) -> None:
        """Test creating error info for config invalid error."""
        error_info = ErrorMessageFactory.create_error_info(
            error_type=ErrorType.CONFIG_INVALID,
            exception=ValueError("Invalid YAML syntax"),
            context={"file_path": "/path/to/config.yaml"},
        )

        assert error_info.error_type == ErrorType.CONFIG_INVALID
        assert error_info.title == "Configuration Error"
        assert (
            "config" in error_info.message.lower()
            or "configuration" in error_info.message.lower()
        )
        assert error_info.technical_info == "Invalid YAML syntax"
        assert "file_path" in str(error_info.details)
        assert error_info.recovery_suggestions is not None
        assert len(error_info.recovery_suggestions) > 0
        assert error_info.retry_possible is True

    def test_create_error_info_vram_exhausted(self) -> None:
        """Test creating error info for VRAM exhausted error."""
        error_info = ErrorMessageFactory.create_error_info(
            error_type=ErrorType.VRAM_EXHAUSTED,
            exception=RuntimeError("CUDA out of memory"),
            context={"operation": "training"},
        )

        assert error_info.error_type == ErrorType.VRAM_EXHAUSTED
        assert error_info.title == "GPU Memory Insufficient"
        assert "graphics card memory" in error_info.message
        assert error_info.recovery_suggestions is not None
        assert len(error_info.recovery_suggestions) > 0
        assert error_info.retry_possible is True

        # Check technical info includes context
        assert error_info.technical_info is not None
        assert error_info.details is not None
        assert "RTX 3070 Ti" in error_info.details

    def test_create_error_info_unexpected(self) -> None:
        """Test creating error info for unexpected error."""
        error_info = ErrorMessageFactory.create_error_info(
            error_type=ErrorType.UNEXPECTED,
            exception=Exception("Random error"),
        )

        assert error_info.error_type == ErrorType.UNEXPECTED
        assert error_info.title == "Unexpected Error"
        assert "unexpected error occurred" in error_info.message
        assert error_info.technical_info == "Random error"

    def test_create_error_info_without_exception(self) -> None:
        """Test creating error info without exception."""
        error_info = ErrorMessageFactory.create_error_info(
            error_type=ErrorType.TIMEOUT,
        )

        assert error_info.error_type == ErrorType.TIMEOUT
        assert error_info.title == "Operation Timed Out"
        assert error_info.technical_info is None
        assert error_info.retry_possible is True


class TestStandardErrorState:
    """Test StandardErrorState functionality."""

    @patch("streamlit.empty")
    @patch("streamlit.markdown")
    def test_show_error(
        self, mock_markdown: MagicMock, mock_empty: MagicMock
    ) -> None:
        """Test showing error message."""
        # Setup mocks
        mock_placeholder = MagicMock()
        mock_empty.return_value = mock_placeholder

        error_state = StandardErrorState("TestComponent")
        error_info = ErrorInfo(
            error_type=ErrorType.CONFIG_INVALID,
            title="Test Error",
            message="Test message",
        )

        # Show error
        error_state.show_error(error_info)

        # Verify placeholder was created and markdown was called
        mock_empty.assert_called_once()
        mock_placeholder.markdown.assert_called_once()

        # Verify HTML content contains error information
        call_args = mock_placeholder.markdown.call_args[0][0]
        assert "Test Error" in call_args
        assert "Test message" in call_args
        assert "crackseg-error-container" in call_args

    @patch("streamlit.empty")
    @patch("streamlit.columns")
    @patch("streamlit.button")
    def test_show_retry_option(
        self,
        mock_button: MagicMock,
        mock_columns: MagicMock,
        mock_empty: MagicMock,
    ) -> None:
        """Test showing retry option."""
        # Setup mocks
        mock_placeholder = MagicMock()
        mock_empty.return_value = mock_placeholder
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = (mock_col1, mock_col2, mock_col3)
        mock_button.return_value = False

        error_state = StandardErrorState("TestComponent")
        error_info = ErrorInfo(
            error_type=ErrorType.CONFIG_INVALID,
            title="Test Error",
            message="Test message",
            retry_possible=True,
        )

        # Set current error
        error_state._current_error = error_info

        # Mock retry callback
        retry_callback = MagicMock()

        # Show retry option
        error_state.show_retry_option(retry_callback, "Try Again")

        # Verify columns were created and buttons were called
        mock_columns.assert_called_once_with([1, 1, 2])
        assert mock_button.call_count == 2  # Retry and Dismiss buttons

    @patch("streamlit.empty")
    def test_clear_error(self, mock_empty: MagicMock) -> None:
        """Test clearing error state."""
        # Setup mocks
        mock_placeholder = MagicMock()
        mock_empty.return_value = mock_placeholder

        error_state = StandardErrorState("TestComponent")
        error_state._error_placeholder = mock_placeholder
        error_state._current_error = ErrorInfo(
            error_type=ErrorType.UNEXPECTED,
            title="Test",
            message="Test",
        )

        # Clear error
        error_state.clear_error()

        # Verify placeholder was emptied and state was cleared
        mock_placeholder.empty.assert_called_once()
        assert error_state._error_placeholder is None
        assert error_state._current_error is None


class TestErrorClassification:
    """Test error classification functionality."""

    def test_classify_vram_error(self) -> None:
        """Test classification of VRAM/memory errors."""
        from gui.components.loading_spinner import LoadingSpinner

        # CUDA out of memory error
        cuda_error = RuntimeError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )
        assert (
            LoadingSpinner._classify_error(cuda_error)
            == ErrorType.VRAM_EXHAUSTED
        )

        # General memory error
        memory_error = RuntimeError("Not enough memory available")
        assert (
            LoadingSpinner._classify_error(memory_error)
            == ErrorType.MEMORY_INSUFFICIENT
        )

    def test_classify_file_errors(self) -> None:
        """Test classification of file-related errors."""
        from gui.components.loading_spinner import LoadingSpinner

        # Config file not found
        config_error = FileNotFoundError("Config file not found")
        assert (
            LoadingSpinner._classify_error(config_error)
            == ErrorType.CONFIG_NOT_FOUND
        )

        # General file not found
        file_error = FileNotFoundError("Data file missing")
        assert (
            LoadingSpinner._classify_error(file_error)
            == ErrorType.DATA_LOADING
        )

    def test_classify_value_errors(self) -> None:
        """Test classification of value/type errors."""
        from gui.components.loading_spinner import LoadingSpinner

        # Config validation error
        config_error = ValueError("Invalid config parameter")
        assert (
            LoadingSpinner._classify_error(config_error)
            == ErrorType.CONFIG_INVALID
        )

        # Model instantiation error
        model_error = ValueError("Invalid model architecture")
        assert (
            LoadingSpinner._classify_error(model_error)
            == ErrorType.MODEL_INSTANTIATION
        )

    def test_classify_timeout_error(self) -> None:
        """Test classification of timeout errors."""
        from gui.components.loading_spinner import LoadingSpinner

        timeout_error = TimeoutError("Operation timed out")
        assert (
            LoadingSpinner._classify_error(timeout_error) == ErrorType.TIMEOUT
        )

    def test_classify_unexpected_error(self) -> None:
        """Test classification of unexpected errors."""
        from gui.components.loading_spinner import LoadingSpinner

        unexpected_error = Exception("Something random happened")
        assert (
            LoadingSpinner._classify_error(unexpected_error)
            == ErrorType.UNEXPECTED
        )


class TestErrorIntegration:
    """Integration tests for error handling in components."""

    @patch("streamlit.empty")
    @patch("streamlit.info")
    @patch("streamlit.spinner")
    def test_loading_spinner_error_handling(
        self,
        mock_spinner: MagicMock,
        mock_info: MagicMock,
        mock_empty: MagicMock,
    ) -> None:
        """Test LoadingSpinner error handling integration."""
        from gui.components.loading_spinner import LoadingSpinner

        # Setup mocks
        mock_placeholder = MagicMock()
        mock_empty.return_value = mock_placeholder

        # Test error during spinner operation
        with pytest.raises(ValueError):
            with LoadingSpinner.spinner("Test operation"):
                raise ValueError("Test error message")

        # Verify error handling was called
        mock_placeholder.markdown.assert_called()
        mock_info.assert_called()

    @patch("streamlit.empty")
    @patch("streamlit.info")
    def test_progress_bar_error_handling(
        self, mock_info: MagicMock, mock_empty: MagicMock
    ) -> None:
        """Test ProgressBar error handling integration."""
        from gui.components.progress_bar import StepBasedProgress

        # Setup mocks
        mock_placeholder = MagicMock()
        mock_empty.return_value = mock_placeholder

        # Test error during step-based progress
        steps = ["Step 1", "Step 2", "Step 3"]

        with pytest.raises(ValueError):
            with StepBasedProgress("Test Operation", steps) as progress:
                progress.next_step()
                raise ValueError("Test step error")

        # Verify error handling was called
        mock_placeholder.markdown.assert_called()
        mock_info.assert_called()
