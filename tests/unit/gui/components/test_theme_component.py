"""Test module for ThemeComponent.

Tests theme functionality including theme selection,
quick switching, and preview capabilities.
"""

from unittest.mock import Mock, patch

from tests.unit.gui.components.test_component_base import (
    ComponentTestBase,
    MockSessionState,
)


class TestThemeComponent(ComponentTestBase):
    """Test suite for ThemeComponent functionality."""

    def test_theme_component_import(self) -> None:
        """Test that ThemeComponent can be imported successfully."""
        from scripts.gui.components.theme_component import ThemeComponent

        assert ThemeComponent is not None

    @patch("scripts.gui.components.theme_component.st")
    @patch("scripts.gui.components.theme_component.ThemeManager")
    def test_render_theme_selector_basic(
        self, mock_theme_manager, mock_st
    ) -> None:
        """Test basic theme selector rendering."""
        from scripts.gui.components.theme_component import ThemeComponent

        self._setup_comprehensive_streamlit_mock(mock_st)
        self._setup_theme_manager_mock(mock_theme_manager)

        # Call render_theme_selector - should not raise exceptions
        result = ThemeComponent.render_theme_selector()

        # Should return current theme
        assert result == "dark"

    @patch("scripts.gui.components.theme_component.st")
    @patch("scripts.gui.components.theme_component.ThemeManager")
    def test_render_theme_selector_with_location(
        self, mock_theme_manager, mock_st
    ) -> None:
        """Test theme selector with different location."""
        from scripts.gui.components.theme_component import ThemeComponent

        self._setup_comprehensive_streamlit_mock(mock_st)
        self._setup_theme_manager_mock(mock_theme_manager)

        # Call with different location
        result = ThemeComponent.render_theme_selector(location="main")

        # Should work without errors
        assert result == "dark"

    @patch("scripts.gui.components.theme_component.st")
    @patch("scripts.gui.components.theme_component.ThemeManager")
    def test_render_theme_selector_with_expander(
        self, mock_theme_manager, mock_st
    ) -> None:
        """Test theme selector with expander location."""
        from scripts.gui.components.theme_component import ThemeComponent

        self._setup_comprehensive_streamlit_mock(mock_st)
        self._setup_theme_manager_mock(mock_theme_manager)

        # Call with expander location
        result = ThemeComponent.render_theme_selector(location="expander")

        # Should work without errors
        assert result == "dark"

    @patch("scripts.gui.components.theme_component.st")
    @patch("scripts.gui.components.theme_component.ThemeManager")
    def test_render_quick_theme_switcher(
        self, mock_theme_manager, mock_st
    ) -> None:
        """Test quick theme switcher rendering."""
        from scripts.gui.components.theme_component import ThemeComponent

        self._setup_comprehensive_streamlit_mock(mock_st)
        self._setup_theme_manager_mock(mock_theme_manager)

        # Call quick theme switcher - should not raise exceptions
        ThemeComponent.render_quick_theme_switcher()

        # Should complete without errors
        assert True

    def _setup_comprehensive_streamlit_mock(self, mock_st) -> None:
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
        mock_st.expander = Mock(return_value=MockContainer())

        # Columns that unpack correctly and work as context managers
        mock_st.columns = Mock(
            side_effect=lambda spec: [
                MockContainer()
                for _ in (spec if isinstance(spec, list) else range(spec))
            ]
        )

        # UI components with sensible defaults
        def mock_selectbox(label, **kwargs):
            options = kwargs.get("options", ["Dark Theme"])
            # Return first option that makes sense for theme selection
            if "Dark Theme" in options:
                return "Dark Theme"
            elif options:
                return options[0]
            return "Dark Theme"

        mock_st.selectbox = Mock(side_effect=mock_selectbox)
        mock_st.button = Mock(return_value=False)
        mock_st.text_input = Mock(return_value="")

        # Display components
        mock_st.markdown = Mock()
        mock_st.caption = Mock()
        mock_st.info = Mock()
        mock_st.warning = Mock()
        mock_st.success = Mock()
        mock_st.error = Mock()
        mock_st.write = Mock()
        mock_st.rerun = Mock()

        # Progress and feedback
        mock_st.progress = Mock()
        mock_st.spinner = Mock(return_value=MockContainer())

    def _setup_theme_manager_mock(self, mock_theme_manager) -> None:
        """Setup ThemeManager mock with appropriate return values."""
        # Mock theme display options to match what component expects
        mock_theme_manager.get_theme_display_options.return_value = {
            "dark": "Dark Theme",
            "light": "Light Theme",
            "auto": "Auto Theme",
        }

        # Mock current theme
        mock_theme_manager.get_current_theme.return_value = "dark"

        # Mock theme switching
        mock_theme_manager.switch_theme.return_value = True
        mock_theme_manager.apply_theme.return_value = None

        # Mock theme config for previews
        mock_config = Mock()
        mock_config.display_name = "Dark Theme"
        mock_config.description = "Dark color scheme"
        mock_config.colors = Mock()
        mock_config.colors.primary_bg = "#1e1e1e"
        mock_config.colors.secondary_bg = "#2d2d2d"
        mock_config.colors.card_bg = "#3d3d3d"
        mock_config.colors.primary_text = "#ffffff"
        mock_config.colors.secondary_text = "#cccccc"
        mock_config.colors.accent_text = "#4a9eff"
        mock_config.colors.success_color = "#4caf50"
        mock_config.colors.warning_color = "#ff9800"
        mock_config.colors.error_color = "#f44336"

        mock_theme_manager.get_theme_config.return_value = mock_config


class TestThemeIntegration(ComponentTestBase):
    """Test suite for theme integration functionality."""

    @patch("scripts.gui.components.theme_component.st")
    @patch("scripts.gui.components.theme_component.ThemeManager")
    @patch("scripts.gui.components.theme_component.asset_manager")
    def test_theme_selector_integration(
        self, mock_asset_manager, mock_theme_manager, mock_st
    ) -> None:
        """Test theme selector integration with theme manager."""
        from scripts.gui.components.theme_component import ThemeComponent

        # Setup comprehensive mocks
        test_component = TestThemeComponent()
        test_component._setup_comprehensive_streamlit_mock(mock_st)
        test_component._setup_theme_manager_mock(mock_theme_manager)

        # Mock asset manager
        mock_asset_manager.inject_css = Mock()

        # Call apply_current_theme - should not raise exceptions
        ThemeComponent.apply_current_theme()

        # Should complete without errors
        assert True
