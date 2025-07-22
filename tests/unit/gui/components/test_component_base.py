"""Base test framework for GUI component unit tests.

Provides shared utilities, fixtures, and mocking patterns for
comprehensive component testing across the GUI test suite.
"""

from collections.abc import ItemsView, Iterator, KeysView
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


class MockStreamlitContainer:
    """Mock container that supports context manager protocol."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


class MockStreamlitColumns:
    """Mock for streamlit columns that can be unpacked and used as context
    managers."""

    def __init__(self, num_cols: int | list[int] = 2):
        # Handle both int and list inputs (like [2, 1, 1])
        if isinstance(num_cols, list):
            num_cols = len(num_cols)
        else:
            num_cols = int(num_cols)
        self.columns: list[MockStreamlitContainer] = [
            MockStreamlitContainer() for _ in range(num_cols)
        ]

    def __iter__(self) -> Iterator[MockStreamlitContainer]:
        return iter(self.columns)

    def __getitem__(self, index: int) -> MockStreamlitContainer:
        return self.columns[index]

    def __len__(self) -> int:
        return len(self.columns)


class MockSessionState:
    """Mock session state with dynamic attribute support."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return self._state.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_state":
            super().__setattr__(name, value)
        else:
            if not hasattr(self, "_state"):
                super().__setattr__("_state", {})
            self._state[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._state

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self._state.get(key, default)

    def update(self, other: dict[str, Any]) -> None:
        """Update with another dict."""
        self._state.update(other)

    def keys(self) -> KeysView[str]:
        """Return keys view."""
        return self._state.keys()

    def items(self) -> ItemsView[str, Any]:
        """Return items view."""
        return self._state.items()

    def clear_notifications(self) -> None:
        """Mock method for clearing notifications."""
        self._state.pop("notifications", None)

    def add_notification(self, message: str) -> None:
        """Mock method for adding notifications."""
        if "notifications" not in self._state:
            self._state["notifications"] = []
        self._state["notifications"].append(message)


class MockEnhancedStreamlit:
    """Enhanced streamlit mock with better context manager support."""

    def __init__(self) -> None:
        self.session_state = MockSessionState()
        self.sidebar = Mock()

        # Setup sidebar context managers
        self.sidebar.selectbox = Mock(return_value="Home")
        self.sidebar.button = Mock(return_value=False)
        self.sidebar.markdown = Mock()
        self.sidebar.divider = Mock()

        # Setup main streamlit mocks
        def columns_factory(x: int | list[int]) -> MockStreamlitColumns:
            return MockStreamlitColumns(x)

        self.container = Mock(return_value=MockStreamlitContainer())
        self.expander = Mock(return_value=MockStreamlitContainer())
        self.columns = Mock(side_effect=columns_factory)

        # Standard streamlit components
        self.write = Mock()
        self.markdown = Mock()
        self.info = Mock()
        self.error = Mock()
        self.warning = Mock()
        self.success = Mock()
        self.selectbox = Mock()
        self.text_input = Mock()
        self.button = Mock()
        self.file_uploader = Mock()
        self.metric = Mock()
        self.image = Mock()
        self.progress = Mock()
        self.spinner = Mock(return_value=MockStreamlitContainer())
        self.dataframe = Mock()
        self.text_area = Mock()
        self.caption = Mock()
        self.subheader = Mock()
        self.header = Mock()


def create_mock_enhanced_streamlit() -> MockEnhancedStreamlit:
    """Create enhanced streamlit mock with context manager support."""
    return MockEnhancedStreamlit()


def create_mock_component_state(**kwargs: object) -> MockSessionState:
    """Create a comprehensive mock session state for testing.

    Args:
        **kwargs: Initial state values to set

    Returns:
        MockSessionState object with specified initial values
    """
    defaults: dict[str, object] = {
        "config_loaded": False,
        "run_directory": None,
        "current_page": "Home",
        "training_active": False,
        "notifications": [],
        "project_root": None,
        "model_ready": False,
    }
    defaults.update(kwargs)
    return MockSessionState()


def assert_streamlit_called_with_pattern(mock_st: Mock, pattern: str) -> None:
    """Assert that any streamlit method was called with a string
    containing pattern.

    Args:
        mock_st: Mock streamlit object
        pattern: String pattern to search for in call arguments
    """
    # Check all method calls on the mock
    called_with_pattern = False

    for call in mock_st.method_calls:
        _, args, kwargs = call  # Remove unused method_name

        # Check args for pattern
        for arg in args:
            if isinstance(arg, str) and pattern in arg:
                called_with_pattern = True
                break

        # Check kwargs for pattern
        for value in kwargs.values():
            if isinstance(value, str) and pattern in value:
                called_with_pattern = True
                break

        if called_with_pattern:
            break

    assert (
        called_with_pattern
    ), f"Expected streamlit to be called with pattern '{pattern}'"


class ComponentTestBase:
    """Base class for GUI component unit tests.

    Provides common setup, utilities, and patterns for testing
    Streamlit GUI components in isolation.
    """

    @pytest.fixture
    def sample_project_root(self, tmp_path: Path) -> Path:
        """Create a sample project root structure for testing.

        Args:
            tmp_path: Pytest temporary directory

        Returns:
            Path to created sample project root
        """
        project_root = tmp_path / "sample_project"
        project_root.mkdir()

        # Create basic project structure
        (project_root / "configs").mkdir()
        (project_root / "data").mkdir()
        (project_root / "models").mkdir()
        (project_root / "assets").mkdir()

        # Create sample config file
        config_file = project_root / "configs" / "sample_config.yaml"
        config_file.write_text("model:\n  type: test\n  name: sample_model\n")

        return project_root

    @pytest.fixture
    def mock_streamlit(self) -> MockEnhancedStreamlit:
        """Provide enhanced streamlit mock for testing.

        Returns:
            Enhanced mock streamlit object with context manager support
        """
        return create_mock_enhanced_streamlit()

    def setup_method(self) -> None:
        """Setup method run before each test."""
        pass

    def teardown_method(self) -> None:
        """Cleanup method run after each test."""
        pass
