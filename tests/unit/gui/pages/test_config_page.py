"""Unit tests for configuration page functionality.

Tests the config page public API without deep implementation details.
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from scripts.gui.pages.config_page import page_config


class MockSessionState:
    """Mock for streamlit session state with iterable support."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return self.data.get(name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "data":
            super().__setattr__(name, value)
        else:
            self.data[name] = value

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for session state."""
        return key in self.data

    def __iter__(self) -> Any:
        """Support iteration over session state."""
        return iter(self.data)

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style get access."""
        return self.data.get(key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style set access."""
        self.data[key] = value

    def keys(self) -> Any:
        """Support keys() method."""
        return self.data.keys()

    def values(self) -> Any:
        """Support values() method."""
        return self.data.values()

    def items(self) -> Any:
        """Support items() method."""
        return self.data.items()


class TestConfigPage:
    """Test suite for configuration page."""

    def test_config_page_import(self) -> None:
        """Test that config page can be imported successfully."""
        from scripts.gui.pages.config_page import page_config

        assert callable(page_config)

    @patch("scripts.gui.pages.config_page.SessionStateManager")
    @patch("scripts.gui.pages.config_page.render_header")
    @patch(
        "scripts.gui.components.config_editor_component.ConfigEditorComponent"
    )
    @patch("scripts.gui.components.file_browser.FileBrowser")
    @patch("scripts.gui.components.file_upload_component.FileUploadComponent")
    @patch("scripts.gui.utils.save_dialog.SaveDialogManager")
    @patch("streamlit.title")
    def test_page_config_basic_mock(
        self,
        mock_title: Mock,
        mock_save_dialog: Mock,
        mock_upload_component: Mock,
        mock_file_browser: Mock,
        mock_config_editor: Mock,
        mock_render_header: Mock,
        mock_session_manager: Mock,
    ) -> None:
        """Test basic page config functionality with comprehensive mocking."""
        # Setup minimal state
        mock_state = MagicMock()
        mock_state.config_loaded = False
        mock_state.config_path = None
        mock_state.output_dir = None
        mock_session_manager.get.return_value = mock_state

        # Mock session state globally with iterable support
        mock_session = MockSessionState()

        with (
            patch("streamlit.session_state", mock_session),
            patch("streamlit.markdown"),
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.info"),
            patch("streamlit.success"),
            patch("streamlit.warning"),
            patch("streamlit.error"),
            patch("streamlit.button"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.subheader"),
            patch("streamlit.caption"),
            patch("streamlit.tabs") as mock_tabs,
            patch("streamlit.text_input", return_value=""),
            patch("streamlit.rerun"),
        ):
            # Setup expander as context manager
            mock_exp_ctx = MagicMock()
            mock_expander.return_value.__enter__ = Mock(
                return_value=mock_exp_ctx
            )
            mock_expander.return_value.__exit__ = Mock(return_value=None)

            # Setup tabs return value
            mock_tab1, mock_tab2 = MagicMock(), MagicMock()
            mock_tabs.return_value = [mock_tab1, mock_tab2]
            mock_tab1.__enter__ = Mock(return_value=mock_tab1)
            mock_tab1.__exit__ = Mock(return_value=None)
            mock_tab2.__enter__ = Mock(return_value=mock_tab2)
            mock_tab2.__exit__ = Mock(return_value=None)

            # Setup columns return value - must match st.columns(3) call
            mock_col1, mock_col2, mock_col3 = (
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
            mock_col1.__enter__ = Mock(return_value=mock_col1)
            mock_col1.__exit__ = Mock(return_value=None)
            mock_col2.__enter__ = Mock(return_value=mock_col2)
            mock_col2.__exit__ = Mock(return_value=None)
            mock_col3.__enter__ = Mock(return_value=mock_col3)
            mock_col3.__exit__ = Mock(return_value=None)

            # Mock component returns
            mock_config_editor.return_value.render.return_value = None
            mock_file_browser.return_value.render.return_value = None
            mock_upload_component.return_value.render.return_value = None

            # This should not raise exceptions
            try:
                page_config()
                assert True  # If we get here, test passed
            except Exception as e:
                pytest.fail(f"Config page failed: {e}")


class TestConfigPageSmoke:
    """Smoke tests for config page."""

    def test_render_theme_controls_import(self) -> None:
        """Test that render_theme_controls can be imported."""
        from scripts.gui.pages.config_page import render_theme_controls

        assert callable(render_theme_controls)
