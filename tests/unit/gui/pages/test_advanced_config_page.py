"""Unit tests for advanced configuration page functionality.

Tests the advanced config page public API without deep implementation details.
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from gui.pages.advanced_config_page import page_advanced_config


class MockSessionState:
    """Mock for streamlit session state."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return self.data.get(name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "data":
            super().__setattr__(name, value)
        else:
            self.data[name] = value


class TestAdvancedConfigPage:
    """Test suite for advanced configuration page."""

    def test_advanced_config_page_import(self) -> None:
        """Test that advanced config page can be imported successfully."""
        from gui.pages.advanced_config_page import page_advanced_config

        assert callable(page_advanced_config)

    @patch("scripts.gui.pages.advanced_config_page.SessionStateManager")
    @patch("streamlit.title")
    @patch("streamlit.tabs")
    @patch("streamlit.markdown")
    @patch("streamlit.info")
    @patch(
        "scripts.gui.components.config_editor_component.ConfigEditorComponent"
    )
    def test_page_advanced_config_basic_mock(
        self,
        mock_editor_component: Mock,
        mock_info: Mock,
        mock_markdown: Mock,
        mock_tabs: Mock,
        mock_title: Mock,
        mock_session_manager: Mock,
    ) -> None:
        """Test basic advanced config page functionality."""
        # Setup minimal state
        mock_state = MagicMock()
        mock_state.config_loaded = False
        mock_session_manager.get.return_value = mock_state

        # Mock tabs to return context managers
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]

        # Setup tab context managers
        for tab in [mock_tab1, mock_tab2, mock_tab3]:
            tab.__enter__ = Mock(return_value=tab)
            tab.__exit__ = Mock(return_value=None)

        # Mock the editor component
        mock_editor_instance = MagicMock()
        mock_editor_component.return_value = mock_editor_instance
        mock_editor_instance.render_editor.return_value = "# test config"

        # This should not raise exceptions - basic smoke test
        try:
            page_advanced_config()
            # Verify basic function calls
            mock_title.assert_called_once()
            mock_session_manager.get.assert_called_once()
            mock_tabs.assert_called_once()
            assert True  # If we get here, test passed
        except Exception as e:
            pytest.fail(f"Advanced config page failed: {e}")


class TestAdvancedConfigPageSmoke:
    """Smoke tests for advanced config page."""

    def test_page_function_exists(self) -> None:
        """Test that page function exists and is callable."""
        assert callable(page_advanced_config)
