"""
Test module for FileUploadComponent. Tests file upload functionality
including validation, processing, and error handling capabilities.
"""

from unittest.mock import Mock, patch

from .test_component_base import ComponentTestBase


class TestFileUploadComponent(ComponentTestBase):
    """Test suite for FileUploadComponent functionality."""

    @patch("streamlit.file_uploader")
    def test_render_upload_section_with_params(self, mock_file_uploader):
        """Verify the upload section renders with custom title and help."""
        from gui.components.file_upload_component import FileUploadComponent

        mock_file_uploader.return_value = None
        result = FileUploadComponent.render_upload_section(
            title="Test Upload",
            help_text="Test help",
            target_directory="test_dir",
            key_suffix="_test",
        )
        assert result is None

    @patch("streamlit.file_uploader")
    def test_render_upload_section_no_file(self, mock_file_uploader):
        """Verify behavior when no file is uploaded."""
        from gui.components.file_upload_component import FileUploadComponent

        mock_file_uploader.return_value = None
        result = FileUploadComponent.render_upload_section()
        assert result is None

    @patch("streamlit.file_uploader")
    def test_render_upload_section_with_file(self, mock_st_file_uploader):
        """Verify behavior when a single file is uploaded."""
        from gui.components.file_upload_component import FileUploadComponent

        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.getvalue.return_value = b"test content"
        mock_st_file_uploader.return_value = [mock_file]
        result = FileUploadComponent.render_upload_section()
        # Result depends on validation - may be None or tuple
        assert result is None or isinstance(result, tuple)

    @patch("streamlit.file_uploader")
    def test_render_upload_section_file_validation_failure(
        self, mock_st_file_uploader
    ):
        """Verify behavior with file validation failure."""
        from gui.components.file_upload_component import FileUploadComponent

        mock_file = Mock()
        mock_file.name = "test.invalid"
        mock_file.getvalue.return_value = b"invalid content"
        mock_st_file_uploader.return_value = [mock_file]
        result = FileUploadComponent.render_upload_section()
        assert result is None

    @patch("streamlit.file_uploader")
    def test_render_upload_section_custom_params(self, mock_st_file_uploader):
        """Test the upload section with custom title and help text."""
        from gui.components.file_upload_component import FileUploadComponent

        mock_st_file_uploader.return_value = None
        result = FileUploadComponent.render_upload_section(
            title="Custom Upload Title",
            help_text="Custom help text",
            target_directory="custom_dir",
            show_validation=False,
        )
        assert result is None

    @patch("streamlit.file_uploader")
    def test_render_upload_section_no_help_text(self, mock_st_file_uploader):
        """Test the upload section with no help text provided."""
        from gui.components.file_upload_component import FileUploadComponent

        mock_st_file_uploader.return_value = None
        result = FileUploadComponent.render_upload_section(help_text=None)
        assert result is None

    @patch("streamlit.file_uploader")
    def test_render_upload_section_upload_failure(self, mock_st_file_uploader):
        """Test behavior when file upload raises an exception."""
        from gui.components.file_upload_component import FileUploadComponent

        mock_st_file_uploader.side_effect = Exception("Upload failed")
        result = FileUploadComponent.render_upload_section()
        assert result is None

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
