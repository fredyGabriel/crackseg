"""Test module for FileUploadComponent.

Tests file upload functionality including validation,
processing, and error handling capabilities.
"""

from unittest.mock import Mock, patch

from scripts.gui.components.file_upload_component import FileUploadComponent
from tests.unit.gui.components.test_component_base import (
    ComponentTestBase,
)


class TestFileUploadComponent(ComponentTestBase):
    """Test suite for FileUploadComponent functionality."""

    @patch("scripts.gui.components.file_upload_component.st")
    def test_render_upload_section_no_file(self, mock_st) -> None:
        """Test render upload section when no file is uploaded."""
        # Setup mocks
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.file_uploader.return_value = None  # No file uploaded

        # Call the method
        result = FileUploadComponent.render_upload_section(
            title="Test Upload",
            help_text="Test help",
            target_directory="test_dir",
            key_suffix="_test",
            show_validation=True,
            show_preview=True,
        )

        # Should return None when no file uploaded
        assert result is None

        # Verify UI elements were called
        mock_st.expander.assert_called_once()
        mock_st.file_uploader.assert_called_once()

    @patch("scripts.gui.components.file_upload_component.st")
    @patch("scripts.gui.components.file_upload_component.get_upload_file_info")
    @patch("scripts.gui.components.file_upload_component.upload_config_file")
    @patch(
        "scripts.gui.components.file_upload_component.create_upload_progress_placeholder"
    )
    @patch(
        "scripts.gui.components.file_upload_component.update_upload_progress"
    )
    @patch("scripts.gui.components.file_upload_component.SessionStateManager")
    def test_render_upload_section_with_valid_file(
        self,
        mock_session,
        mock_progress_update,
        mock_progress_placeholder,
        mock_upload,
        mock_file_info,
        mock_st,
    ) -> None:
        """Test render upload section with a valid file."""
        # Setup mocks
        mock_uploaded_file = Mock()
        mock_uploaded_file.name = "test.yaml"

        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.file_uploader.return_value = mock_uploaded_file
        mock_st.button.return_value = True  # User clicks process button

        # Use comprehensive mocking approach
        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        mock_file_info.return_value = {
            "name": "test.yaml",
            "size_human": "1 KB",
            "extension": ".yaml",
            "is_valid_extension": True,
            "is_valid_size": True,
            "max_size_mb": 10,
        }

        mock_upload.return_value = (
            "/path/to/test.yaml",
            {"model": "test"},
            [],  # No validation errors
        )

        mock_state = Mock()  # noqa: F841
        mock_session.get.return_value = mock_state

        # Call the method
        result = FileUploadComponent.render_upload_section()  # noqa: F841

        # Should return tuple when file is processed
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3

        file_path, config_dict, validation_errors = result
        assert file_path == "/path/to/test.yaml"
        assert config_dict == {"model": "test"}
        assert validation_errors == []

    @patch("scripts.gui.components.file_upload_component.st")
    @patch("scripts.gui.components.file_upload_component.get_upload_file_info")
    def test_render_upload_section_invalid_file(
        self, mock_file_info, mock_st
    ) -> None:
        """Test render upload section with invalid file."""
        # Setup mocks
        mock_uploaded_file = Mock()
        mock_uploaded_file.name = "test.txt"  # Invalid extension

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]
        mock_st.file_uploader.return_value = mock_uploaded_file

        mock_file_info.return_value = {
            "name": "test.txt",
            "size_human": "1 KB",
            "extension": ".txt",
            "is_valid_extension": False,  # Invalid extension
            "is_valid_size": True,
            "max_size_mb": 10,
        }

        # Call the method - this should process the file but return None
        # due to validation
        result = FileUploadComponent.render_upload_section()  # noqa: F841

        # Should return None for invalid files
        assert result is None

        # Verify error was shown
        mock_st.error.assert_called()

    @patch("scripts.gui.components.file_upload_component.st")
    def test_render_upload_section_custom_parameters(self, mock_st) -> None:
        """Test render upload section with custom parameters."""
        # Setup mocks
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.file_uploader.return_value = None

        # Test with custom parameters
        result = FileUploadComponent.render_upload_section(
            title="Custom Upload Title",
            help_text="Custom help text",
            target_directory="custom_dir",
            key_suffix="_custom",
            show_validation=False,
            show_preview=False,
        )

        assert result is None

        # Verify expander was called with custom title
        mock_st.expander.assert_called_with(
            "Custom Upload Title", expanded=False
        )

    @patch("scripts.gui.components.file_upload_component.st")
    def test_render_upload_section_default_help_text(self, mock_st) -> None:
        """Test render upload section shows default help when none provided."""
        # Setup mocks
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.file_uploader.return_value = None

        # Call without help_text (should use default)
        result = FileUploadComponent.render_upload_section(help_text=None)

        assert result is None

        # Verify markdown was called (for default help)
        mock_st.markdown.assert_called()

    @patch("scripts.gui.components.file_upload_component.st")
    @patch("scripts.gui.components.file_upload_component.get_upload_file_info")
    @patch("scripts.gui.components.file_upload_component.upload_config_file")
    @patch(
        "scripts.gui.components.file_upload_component.create_upload_progress_placeholder"
    )
    @patch(
        "scripts.gui.components.file_upload_component.update_upload_progress"
    )
    def test_render_upload_section_upload_error(
        self,
        mock_progress_update,
        mock_progress_placeholder,
        mock_upload,
        mock_file_info,
        mock_st,
    ) -> None:
        """Test render upload section handles upload errors."""
        # Setup mocks
        mock_uploaded_file = Mock()
        mock_uploaded_file.name = "test.yaml"

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]
        mock_st.file_uploader.return_value = mock_uploaded_file
        mock_st.button.return_value = True

        mock_file_info.return_value = {
            "name": "test.yaml",
            "size_human": "1 KB",
            "extension": ".yaml",
            "is_valid_extension": True,
            "is_valid_size": True,
            "max_size_mb": 10,
        }

        # Mock upload to raise an exception
        mock_upload.side_effect = Exception("Upload failed")

        # Call the method
        result = FileUploadComponent.render_upload_section()  # noqa: F841

        # Should return None when upload fails
        assert result is None

        # Verify error handling
        mock_st.error.assert_called()

    def _setup_comprehensive_streamlit_mock(self, mock_st: Mock) -> None:
        """Setup comprehensive streamlit mock for testing."""
        # Session state with item assignment support
        mock_session_state = {}
        mock_st.session_state = mock_session_state

        # Context managers for layout components
        class MockContainer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_st.container = Mock(return_value=MockContainer())
        mock_st.expander = Mock(return_value=MockContainer())

        # Columns that unpack correctly and work as context managers
        def columns_side_effect(spec: int | list[int]) -> list[MockContainer]:
            if isinstance(spec, list):
                return [MockContainer() for _ in spec]
            return [MockContainer() for _ in range(spec)]

        mock_st.columns = Mock(side_effect=columns_side_effect)

        # UI components with sensible defaults - preserve existing mocks if set
        if not hasattr(mock_st, "file_uploader") or not getattr(
            mock_st.file_uploader, "_mock_name", None
        ):  # type: ignore[attr-defined]
            mock_st.file_uploader = Mock(return_value=None)
        if not hasattr(mock_st, "button") or not getattr(
            mock_st.button, "_mock_name", None
        ):  # type: ignore[attr-defined]
            mock_st.button = Mock(return_value=False)

        mock_st.text_input = Mock(return_value="")
        mock_st.selectbox = Mock(return_value="default")

        # Display components
        mock_st.subheader = Mock()
        mock_st.markdown = Mock()
        mock_st.caption = Mock()
        mock_st.info = Mock()
        mock_st.warning = Mock()
        mock_st.success = Mock()
        mock_st.error = Mock()
        mock_st.write = Mock()
        mock_st.text_area = Mock()

        # Progress and feedback
        mock_st.progress = Mock()
        mock_st.spinner = Mock(return_value=MockContainer())
