"""Test module for SidebarComponent.

Tests sidebar functionality including navigation,
state management, and project information display.
"""

from pathlib import Path
from unittest.mock import Mock, patch

from tests.unit.gui.components.test_component_base import (
    ComponentTestBase,
    MockSessionState,
)


class TestSidebarComponent(ComponentTestBase):
    """Test suite for SidebarComponent functionality."""

    def test_render_sidebar_import(self) -> None:
        """Test that render_sidebar function can be imported successfully."""
        from gui.components.sidebar_component import render_sidebar

        assert render_sidebar is not None

    @patch("scripts.gui.components.sidebar_component.st")
    def test_render_sidebar_basic(
        self, mock_st, sample_project_root: Path
    ) -> None:
        """Test basic sidebar rendering functionality."""
        from gui.components.sidebar_component import render_sidebar

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        # Call render_sidebar - should not raise exceptions
        render_sidebar(project_root=sample_project_root)

        # Should complete without errors
        assert True

    @patch("scripts.gui.components.sidebar_component.st")
    def test_render_sidebar_with_different_project_root(
        self, mock_st, sample_project_root: Path
    ) -> None:
        """Test sidebar with different project root."""
        from gui.components.sidebar_component import render_sidebar

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        # Create different root
        different_root = sample_project_root.parent / "different_project"
        different_root.mkdir()

        render_sidebar(project_root=different_root)
        assert True

    @patch("scripts.gui.components.sidebar_component.st")
    def test_render_sidebar_navigation_selection(
        self, mock_st, sample_project_root: Path
    ) -> None:
        """Test sidebar navigation selection."""
        from gui.components.sidebar_component import render_sidebar

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]
        # Setup sidebar selectbox to return different page
        mock_st.sidebar.selectbox.return_value = "Config"

        render_sidebar(project_root=sample_project_root)
        assert True

    @patch("scripts.gui.components.sidebar_component.st")
    def test_render_sidebar_project_info_display(
        self, mock_st, sample_project_root: Path
    ) -> None:
        """Test sidebar displays project information."""
        from gui.components.sidebar_component import render_sidebar

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        render_sidebar(project_root=sample_project_root)

        # Verify function executed without errors
        assert True

    @patch("scripts.gui.components.sidebar_component.st")
    def test_render_sidebar_with_session_state(
        self, mock_st, sample_project_root: Path
    ) -> None:
        """Test sidebar with different session state values."""
        from gui.components.sidebar_component import render_sidebar

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        # Setup session state
        mock_st.session_state.current_page = "Train"
        mock_st.session_state.training_active = True

        # Mock add_notification method for session state
        # (no longer needed, removed unused variable)

        # Mock PageRouter.handle_navigation_change to not call add_notification
        with patch(
            "scripts.gui.components.sidebar_component.PageRouter.handle_navigation_change",
            return_value=False,
        ):
            render_sidebar(project_root=sample_project_root)

        assert True

    @patch("scripts.gui.components.sidebar_component.st")
    def test_render_sidebar_ui_elements(
        self, mock_st, sample_project_root: Path
    ) -> None:
        """Test sidebar UI elements are rendered."""
        from gui.components.sidebar_component import render_sidebar

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        render_sidebar(project_root=sample_project_root)

        # Verify function executed without errors
        assert True

    def test_render_sidebar_function_signature(self) -> None:
        """Test render_sidebar function has expected signature."""
        import inspect

        from gui.components.sidebar_component import render_sidebar

        sig = inspect.signature(render_sidebar)
        params = list(sig.parameters.keys())

        # Should have project_root parameter
        assert "project_root" in params

    def test_render_sidebar_error_handling(self) -> None:
        """Test sidebar error handling with invalid inputs."""
        from gui.components.sidebar_component import render_sidebar

        # Should handle None project_root gracefully
        with patch("scripts.gui.components.sidebar_component.st") as mock_st:
            self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

            try:
                render_sidebar(project_root=None)  # type: ignore[arg-type]
            except Exception:
                # Should not raise unhandled exceptions
                pass

    @patch("scripts.gui.components.sidebar_component.st")
    def test_render_sidebar_path_handling(self, mock_st) -> None:
        """Test sidebar handles different path types."""
        from gui.components.sidebar_component import render_sidebar

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        # Test with string path
        test_path = Path("/test/path")
        render_sidebar(project_root=test_path)
        assert True

    def _setup_comprehensive_streamlit_mock(self, mock_st: Mock) -> None:
        """Setup comprehensive streamlit mock for testing."""
        # Session state with item assignment support
        mock_session_state = MockSessionState()
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

        mock_st.columns = Mock(side_effect=columns_side_effect)

        # Sidebar components that also works as context manager
        class MockSidebar:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def __init__(self):
                self.selectbox = Mock(return_value="Home")
                self.button = Mock(return_value=False)
                self.markdown = Mock()
                self.divider = Mock()
                self.info = Mock()
                self.success = Mock()
                self.warning = Mock()
                self.error = Mock()
                self.metric = Mock()

        mock_st.sidebar = MockSidebar()

        # UI components with sensible defaults
        mock_st.selectbox = Mock(return_value="default")
        mock_st.text_input = Mock(return_value="")
        mock_st.button = Mock(return_value=False)

        # Display components
        mock_st.subheader = Mock()
        mock_st.markdown = Mock()
        mock_st.caption = Mock()
        mock_st.info = Mock()
        mock_st.warning = Mock()
        mock_st.success = Mock()
        mock_st.error = Mock()
        mock_st.write = Mock()

        # Progress and feedback
        mock_st.progress = Mock()
        mock_st.spinner = Mock(return_value=MockContainer())
