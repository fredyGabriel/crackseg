"""Test module for ThemeComponent.

Tests theme functionality including theme selection,
quick switching, and preview capabilities.
"""

from unittest.mock import patch

from .test_component_base import ComponentTestBase


class TestThemeComponent(ComponentTestBase):
    """Test suite for ThemeComponent functionality."""

    @patch("gui.components.theme_component.st")
    def test_render_theme_selector_default(self, mock_st):
        """Verify the theme selector renders with default parameters."""
        from gui.components.theme_component import ThemeComponent

        mock_st.sidebar.selectbox.return_value = "Dark"
        result = ThemeComponent.render_theme_selector()

        # Should return the selected theme
        assert isinstance(result, str)
        mock_st.sidebar.selectbox.assert_called_once()

    @patch("gui.components.theme_component.st")
    def test_render_theme_selector_main_location(self, mock_st):
        """Test theme selector in main area."""
        from gui.components.theme_component import ThemeComponent

        mock_st.selectbox.return_value = "Light"
        result = ThemeComponent.render_theme_selector(location="main")

        assert isinstance(result, str)
        mock_st.selectbox.assert_called_once()

    @patch("gui.components.theme_component.st")
    def test_render_theme_selector_with_info(self, mock_st):
        """Test theme selector with info display."""
        from gui.components.theme_component import ThemeComponent

        mock_st.sidebar.selectbox.return_value = "Auto"
        result = ThemeComponent.render_theme_selector(show_info=True)

        assert isinstance(result, str)
        mock_st.sidebar.selectbox.assert_called_once()

    @patch("gui.components.theme_component.st")
    def test_render_quick_theme_switcher(self, mock_st):
        """Test quick theme switcher functionality."""
        from gui.components.theme_component import ThemeComponent

        # Should not raise any exceptions
        ThemeComponent.render_quick_theme_switcher()

        # Verify some streamlit components were called
        assert mock_st.button.called or mock_st.columns.called

    @patch("gui.components.theme_component.st")
    def test_render_theme_status(self, mock_st):
        """Test theme status display."""
        from gui.components.theme_component import ThemeComponent

        # Should not raise any exceptions
        ThemeComponent.render_theme_status()

        # Verify some display method was called
        assert any(
            [
                mock_st.info.called,
                mock_st.success.called,
                mock_st.markdown.called,
                mock_st.write.called,
            ]
        )


class TestThemeIntegration(ComponentTestBase):
    """Test suite for theme integration functionality."""

    @patch("gui.components.theme_component.st")
    def test_theme_preview_rendering(self, mock_st):
        """Test theme preview functionality."""
        from gui.components.theme_component import ThemeComponent

        # Should not raise exceptions
        ThemeComponent.render_theme_preview("dark")

        # Verify markdown or other display method was called
        assert mock_st.markdown.called or mock_st.write.called
