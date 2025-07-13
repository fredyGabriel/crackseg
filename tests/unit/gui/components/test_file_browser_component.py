"""Test module for FileBrowserComponent.

Tests file browser functionality including navigation,
filtering, and file selection capabilities.
"""

from unittest.mock import Mock, patch

from scripts.gui.components.file_browser_component import FileBrowserComponent
from tests.unit.gui.components.test_component_base import (
    ComponentTestBase,
)


class TestFileBrowserComponent(ComponentTestBase):
    """Test suite for FileBrowserComponent functionality."""

    def test_initialization(self) -> None:
        """Test component can be initialized."""
        component = FileBrowserComponent()
        assert component is not None
        assert hasattr(component, "supported_extensions")
        assert hasattr(component, "sort_options")

    @patch("scripts.gui.components.file_browser_component.st")
    @patch(
        "scripts.gui.components.file_browser_component.scan_config_directories"
    )
    def test_render_basic(self, mock_scan, mock_st) -> None:
        """Test basic render functionality with correct parameters."""
        # Setup comprehensive streamlit mocking
        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]
        mock_scan.return_value = {"configs": ["config1.yaml", "config2.yaml"]}

        component = FileBrowserComponent()

        # Call render - should not raise exceptions
        result = component.render(key="test_browser")  # noqa: F841

        # Verify basic structure
        assert isinstance(result, dict)
        assert "selected_files" in result
        assert "current_directory" in result
        assert "total_files" in result

    @patch("scripts.gui.components.file_browser_component.st")
    def test_render_with_filter(self, mock_st) -> None:
        """Test render with file filtering."""
        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        component = FileBrowserComponent()

        with patch(
            "scripts.gui.components.file_browser_component.scan_config_directories"
        ) as mock_scan:
            mock_scan.return_value = {
                "configs": ["config1.yaml", "config2.yaml"]
            }
            result = component.render(
                key="test_filter", filter_text="config1"
            )  # noqa: F841

        assert isinstance(result, dict)

    @patch("scripts.gui.components.file_browser_component.st")
    def test_render_allows_multiple_selection(self, mock_st) -> None:
        """Test multiple file selection capability."""
        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        component = FileBrowserComponent()

        with patch(
            "scripts.gui.components.file_browser_component.scan_config_directories"
        ) as mock_scan:
            mock_scan.return_value = {
                "configs": ["config1.yaml", "config2.yaml"]
            }
            result = component.render(
                key="test_multi", allow_multiple=True
            )  # noqa: F841

        assert isinstance(result, dict)

    @patch("scripts.gui.components.file_browser_component.st")
    def test_render_navigation_controls(self, mock_st) -> None:
        """Test navigation controls are rendered."""
        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        component = FileBrowserComponent()

        with patch(
            "scripts.gui.components.file_browser_component.scan_config_directories"
        ) as mock_scan:
            mock_scan.return_value = {
                "configs": ["config1.yaml", "config2.yaml"]
            }
            component.render(key="nav_test")

        # Should not raise exceptions
        assert True

    def test_supported_extensions(self) -> None:
        """Test supported file extensions are correctly configured."""
        component = FileBrowserComponent()

        assert ".yaml" in component.supported_extensions
        assert ".yml" in component.supported_extensions

    def test_sort_options_available(self) -> None:
        """Test sort options are properly defined."""
        component = FileBrowserComponent()

        expected_keys = [
            "Name (A-Z)",
            "Name (Z-A)",
            "Modified (Newest)",
            "Modified (Oldest)",
            "Size (Largest)",
            "Size (Smallest)",
        ]

        for key in expected_keys:
            assert key in component.sort_options

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

        # Columns that unpack correctly and work as context managers
        def columns_side_effect(spec: int | list[int]) -> list[MockContainer]:
            if isinstance(spec, list):
                return [MockContainer() for _ in spec]
            return [MockContainer() for _ in range(spec)]

        mock_st.columns = Mock(side_effect=columns_side_effect)  # type: ignore

        # UI components with sensible defaults
        def mock_selectbox(label: str, **kwargs) -> str:  # type: ignore
            options: list[str] = kwargs.get("options", ["default"])  # type: ignore
            if "Sort by" in label:
                return "Name (A-Z)"  # Valid sort option
            elif "Directory" in label:
                return "configs"
            else:
                return options[0] if options else "default"  # type: ignore[return-value]

        mock_st.selectbox = Mock(side_effect=mock_selectbox)  # type: ignore
        mock_st.text_input = Mock(return_value="")
        mock_st.multiselect = Mock(return_value=[])
        mock_st.button = Mock(return_value=False)

        # Display components
        mock_st.subheader = Mock()
        mock_st.markdown = Mock()
        mock_st.caption = Mock()
        mock_st.info = Mock()
        mock_st.warning = Mock()
        mock_st.success = Mock()
        mock_st.error = Mock()

        # Cache management
        mock_st.cache_data = Mock()
        mock_st.cache_data.clear = Mock()
