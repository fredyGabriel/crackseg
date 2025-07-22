"""
Unified Streamlit mocking system for GUI testing. This module provides
a comprehensive mocking framework that combines the best practices from
streamlit_mocking.py and gui_testing_framework.py.
"""

from typing import Any
from unittest.mock import Mock


class UnifiedMockingConfig:
    """Configuration for unified Streamlit mocking."""

    def __init__(
        self,
        enable_session_state: bool = True,
        enable_widget_callbacks: bool = True,
        enable_file_uploads: bool = True,
        track_state_changes: bool = True,
    ) -> None:
        self.enable_session_state = enable_session_state
        self.enable_widget_callbacks = enable_widget_callbacks
        self.enable_file_uploads = enable_file_uploads
        self.track_state_changes = track_state_changes


class UnifiedStreamlitMocker:
    """Unified Streamlit mocker combining all testing capabilities."""

    def __init__(self, config: UnifiedMockingConfig | None = None) -> None:
        """Initialize unified Streamlit mocker.

        Args:
            config: Mocking configuration
        """
        self.config = config or UnifiedMockingConfig()
        self._session_state_history: list[dict[str, Any]] = []

    def create_streamlit_mock(self) -> Mock:
        """Create a comprehensive Streamlit mock with all components."""
        mock_st = Mock()

        # Core session state with unified behavior
        if self.config.enable_session_state:
            mock_st.session_state = self._create_unified_session_state_mock()

        # Widget components with callback support
        if self.config.enable_widget_callbacks:
            self._setup_unified_widget_mocks(mock_st)

        # File handling components
        if self.config.enable_file_uploads:
            self._setup_unified_file_mocks(mock_st)

        # Layout and display components
        self._setup_layout_mocks(mock_st)
        self._setup_display_mocks(mock_st)

        return mock_st

    def _create_unified_session_state_mock(self) -> Any:
        """Create unified session state mock combining best practices."""

        class UnifiedSessionStateMock:
            def __init__(
                self, parent_mocker: "UnifiedStreamlitMocker"
            ) -> None:
                self._parent = parent_mocker
                self._data: dict[str, Any] = {}

            def __getitem__(self, key: str) -> Any:
                return self._data.get(key)

            def __setitem__(self, key: str, value: Any) -> None:
                old_value = self._data.get(key)
                self._data[key] = value

                # Track changes if enabled
                if self._parent.config.track_state_changes:
                    self._parent._session_state_history.append(
                        {
                            "action": "set",
                            "key": key,
                            "old_value": old_value,
                            "new_value": value,
                        }
                    )

            def __contains__(self, key: str) -> bool:
                return key in self._data

            def __delitem__(self, key: str) -> None:
                if key in self._data:
                    old_value = self._data[key]
                    del self._data[key]

                    if self._parent.config.track_state_changes:
                        self._parent._session_state_history.append(
                            {
                                "action": "delete",
                                "key": key,
                                "old_value": old_value,
                            }
                        )

            def get(self, key: str, default: Any = None) -> Any:
                return self._data.get(key, default)

            def setdefault(self, key: str, default: Any = None) -> Any:
                if key not in self._data:
                    self[key] = default
                return self._data[key]

            def keys(self) -> Any:
                return self._data.keys()

            def values(self) -> Any:
                return self._data.values()

            def items(self) -> Any:
                return self._data.items()

            def clear(self) -> None:
                old_data = self._data.copy()
                self._data.clear()

                if self._parent.config.track_state_changes:
                    self._parent._session_state_history.append(
                        {
                            "action": "clear",
                            "old_data": old_data,
                        }
                    )

            def __setattr__(self, name: str, value: Any) -> None:
                if name.startswith("_"):
                    super().__setattr__(name, value)
                else:
                    # Track attribute access for widget state
                    if (
                        hasattr(self, "_parent")
                        and self._parent.config.track_state_changes
                    ):
                        old_value = (
                            getattr(self, name, None)
                            if hasattr(self, name)
                            else None
                        )
                        self._parent._session_state_history.append(
                            {
                                "action": "setattr",
                                "attr": name,
                                "old_value": old_value,
                                "new_value": value,
                            }
                        )
                    super().__setattr__(name, value)

        return UnifiedSessionStateMock(self)

    def _setup_unified_widget_mocks(self, mock_st: Mock) -> None:
        """Setup unified widget mocks consolidating all widget features."""

        # Text input widgets
        def text_input_factory(
            label: str, value: str = "", **kwargs: Any
        ) -> str:
            key = kwargs.get("key", f"text_input_{label}")
            if hasattr(mock_st.session_state, key):
                return getattr(mock_st.session_state, key)
            return value

        def text_area_factory(
            label: str, value: str = "", **kwargs: Any
        ) -> str:
            key = kwargs.get("key", f"text_area_{label}")
            if hasattr(mock_st.session_state, key):
                return getattr(mock_st.session_state, key)
            return value

        # Numeric input widgets
        def number_input_factory(
            label: str, value: float = 0.0, **kwargs: Any
        ) -> float:
            key = kwargs.get("key", f"number_input_{label}")
            if hasattr(mock_st.session_state, key):
                return getattr(mock_st.session_state, key)
            return value

        def slider_factory(
            label: str,
            min_value: float = 0.0,
            max_value: float = 100.0,
            value: float = 50.0,
            **kwargs: Any,
        ) -> float:
            key = kwargs.get("key", f"slider_{label}")
            if hasattr(mock_st.session_state, key):
                return getattr(mock_st.session_state, key)
            return value

        # Selection widgets
        def selectbox_factory(
            label: str, options: list[Any], index: int = 0, **kwargs: Any
        ) -> Any:
            key = kwargs.get("key", f"selectbox_{label}")
            if hasattr(mock_st.session_state, key):
                return getattr(mock_st.session_state, key)
            return options[index] if options else None

        def multiselect_factory(
            label: str,
            options: list[Any],
            default: list[Any] | None = None,
            **kwargs: Any,
        ) -> list[Any]:
            key = kwargs.get("key", f"multiselect_{label}")
            if hasattr(mock_st.session_state, key):
                return getattr(mock_st.session_state, key)
            return default or []

        # Button widgets
        def button_factory(label: str, **kwargs: Any) -> bool:
            kwargs.get("key", f"button_{label}")
            # Buttons typically return False unless explicitly clicked
            return kwargs.get("clicked", False)

        # Assign factory functions to mock
        mock_st.text_input = Mock(side_effect=text_input_factory)
        mock_st.text_area = Mock(side_effect=text_area_factory)
        mock_st.number_input = Mock(side_effect=number_input_factory)
        mock_st.slider = Mock(side_effect=slider_factory)
        mock_st.selectbox = Mock(side_effect=selectbox_factory)
        mock_st.multiselect = Mock(side_effect=multiselect_factory)
        mock_st.button = Mock(side_effect=button_factory)

        # Simple widgets with defaults
        mock_st.radio = Mock(return_value="Option 1")
        mock_st.checkbox = Mock(return_value=False)

    def _setup_unified_file_mocks(self, mock_st: Mock) -> None:
        """Setup unified file handling mocks."""

        def download_button_factory(
            label: str, data: Any, file_name: str, **kwargs: Any
        ) -> bool:
            kwargs.get("key", f"download_{file_name}")
            return kwargs.get("clicked", False)

        def file_uploader_factory(
            label: str, type: str | list[str] | None = None, **kwargs: Any
        ) -> Any:
            kwargs.get("key", f"file_uploader_{label}")
            return kwargs.get("uploaded_file", None)

        mock_st.file_uploader = Mock(side_effect=file_uploader_factory)
        mock_st.download_button = Mock(side_effect=download_button_factory)

    def _setup_layout_mocks(self, mock_st: Mock) -> None:
        """Setup layout component mocks."""
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.container = Mock(return_value=Mock())
        mock_st.sidebar = Mock()

        # Forms
        form_mock = Mock()
        form_mock.__enter__ = Mock(return_value=form_mock)
        form_mock.__exit__ = Mock(return_value=None)
        form_mock.form_submit_button = Mock(return_value=False)
        mock_st.form = Mock(return_value=form_mock)

    def _setup_display_mocks(self, mock_st: Mock) -> None:
        """Setup display component mocks."""
        mock_st.write = Mock()
        mock_st.markdown = Mock()
        mock_st.text = Mock()
        mock_st.json = Mock()
        mock_st.code = Mock()

        # Status displays
        mock_st.success = Mock()
        mock_st.info = Mock()
        mock_st.warning = Mock()
        mock_st.error = Mock()
        mock_st.exception = Mock()

        # Progress indicators
        mock_st.progress = Mock(return_value=Mock())
        mock_st.spinner = Mock(return_value=Mock())

    def get_session_state_history(self) -> list[dict[str, Any]]:
        """Get the history of session state changes."""
        return self._session_state_history.copy()

    def assert_session_state_changed(self, key: str) -> None:
        """Assert that a specific key was changed in session state."""
        for change in self._session_state_history:
            if change.get("key") == key or change.get("attr") == key:
                return
        raise AssertionError(f"Session state key '{key}' was not changed")

    def create_mock_uploaded_file(
        self,
        file_name: str,
        content: str | bytes,
        file_type: str = "text/plain",
    ) -> Mock:
        """Create a mock uploaded file for testing.

        Args:
            file_name: Name of the file
            content: File content
            file_type: MIME type of the file

        Returns:
            Mock uploaded file object
        """
        mock_file = Mock()
        mock_file.name = file_name
        mock_file.type = file_type
        mock_file.size = (
            len(content) if isinstance(content, str | bytes) else 0
        )

        if isinstance(content, str):
            content = content.encode("utf-8")

        mock_file.read = Mock(return_value=content)
        mock_file.getvalue = Mock(return_value=content)

        # Add file-like methods
        mock_file.seek = Mock()
        mock_file.tell = Mock(return_value=0)

        return mock_file
