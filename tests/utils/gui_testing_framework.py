"""Enhanced GUI Testing Framework for CrackSeg Streamlit Application.

This module provides comprehensive testing capabilities for Streamlit GUI
components, including advanced mocking strategies, automated UI interaction
testing, and performance-oriented test utilities.
Part of subtask 7.6 - GUI Testing Framework Enhancement.
"""

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import Mock, patch

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class StreamlitTestConfig:
    """Configuration for Streamlit testing framework."""

    enable_session_state: bool = True
    enable_widget_callbacks: bool = True
    enable_file_uploads: bool = True
    enable_download_buttons: bool = True
    mock_external_apis: bool = True
    performance_tracking: bool = True
    ui_interaction_timeout: float = 5.0
    widget_interaction_delay: float = 0.1


@dataclass
class TestInteractionResult:
    """Result of an automated UI interaction test."""

    success: bool
    interaction_type: str
    element_id: str | None
    execution_time: float
    error_message: str | None = None
    captured_state: dict[str, Any] = field(default_factory=dict)


class EnhancedStreamlitMocker:
    """Advanced Streamlit mocking system with comprehensive component
    support."""

    def __init__(self, config: StreamlitTestConfig | None = None) -> None:
        self.config = config or StreamlitTestConfig()
        self._mock_session_state: dict[str, Any] = {}
        self._widget_callbacks: dict[str, Callable[..., Any]] = {}
        self._file_uploads: dict[str, Any] = {}
        self._interaction_history: list[dict[str, Any]] = []

    def create_enhanced_streamlit_mock(self) -> Mock:
        """Create a comprehensive Streamlit mock with all components."""
        mock_st = Mock()

        # Session state with advanced features
        if self.config.enable_session_state:
            mock_st.session_state = self._create_session_state_mock()

        # Widget components with callback support
        if self.config.enable_widget_callbacks:
            self._setup_widget_mocks(mock_st)

        # File handling components
        if self.config.enable_file_uploads:
            self._setup_file_mocks(mock_st)

        # Layout and display components
        self._setup_layout_mocks(mock_st)
        self._setup_display_mocks(mock_st)

        return mock_st

    def _create_session_state_mock(self) -> Any:
        """Create advanced session state mock with real dict behavior."""

        class SessionStateMock:
            def __init__(self) -> None:
                self._data = self._parent_mocker._mock_session_state

            def __getitem__(self, key: str) -> Any:
                return self._data[key]

            def __setitem__(self, key: str, value: Any) -> None:
                self._data[key] = value
                self._parent_mocker._track_interaction(
                    "session_state_set", {"key": key, "value": value}
                )

            def __contains__(self, key: str) -> bool:
                return key in self._data

            def __getattr__(self, name: str) -> Any:
                return self._data.get(name)

            def __setattr__(self, name: str, value: Any) -> None:
                if name.startswith("_"):
                    super().__setattr__(name, value)
                else:
                    self._data[name] = value
                    self._parent_mocker._track_interaction(
                        "session_state_attr", {"attr": name, "value": value}
                    )

            def get(self, key: str, default: Any = None) -> Any:
                return self._data.get(key, default)

            def setdefault(self, key: str, default: Any = None) -> Any:
                return self._data.setdefault(key, default)

            def pop(self, key: str, default: Any = None) -> Any:
                return self._data.pop(key, default)

            def keys(self) -> Any:
                return self._data.keys()

            def values(self) -> Any:
                return self._data.values()

            def items(self) -> Any:
                return self._data.items()

        session_mock = SessionStateMock()
        session_mock._parent_mocker = self  # type: ignore[attr-defined]
        return session_mock

    def _setup_widget_mocks(self, mock_st: Mock) -> None:
        """Setup widget mocks with callback support."""

        def button_factory(label: str, **kwargs: Any) -> bool:
            key = kwargs.get("key", f"button_{label}")
            callback = kwargs.get("on_click")
            clicked = kwargs.get("_test_clicked", False)

            if clicked and callback:
                # Simulate callback execution
                self._widget_callbacks[key] = callback
                callback()

            self._track_interaction(
                "button_click", {"label": label, "key": key}
            )
            return clicked

        def selectbox_factory(
            label: str, options: list[Any], **kwargs: Any
        ) -> Any:
            key = kwargs.get("key", f"selectbox_{label}")
            index = kwargs.get("index", 0)
            value = (
                options[index] if options and index < len(options) else None
            )

            self._track_interaction(
                "selectbox_select",
                {"label": label, "key": key, "value": value},
            )
            return value

        def text_input_factory(label: str, **kwargs: Any) -> str:
            key = kwargs.get("key", f"text_input_{label}")
            value = kwargs.get("value", "")

            self._track_interaction(
                "text_input_change",
                {"label": label, "key": key, "value": value},
            )
            return value

        mock_st.button = Mock(side_effect=button_factory)
        mock_st.selectbox = Mock(side_effect=selectbox_factory)
        mock_st.text_input = Mock(side_effect=text_input_factory)
        mock_st.slider = Mock(return_value=1)
        mock_st.checkbox = Mock(return_value=False)
        mock_st.radio = Mock(return_value="option1")
        mock_st.multiselect = Mock(return_value=[])

    def _setup_file_mocks(self, mock_st: Mock) -> None:
        """Setup file handling mocks."""

        def file_uploader_factory(label: str, **kwargs: Any) -> Any:
            key = kwargs.get("key", f"file_uploader_{label}")
            uploaded_file = self._file_uploads.get(key)

            self._track_interaction(
                "file_upload",
                {"label": label, "key": key, "file": uploaded_file},
            )
            return uploaded_file

        def download_button_factory(
            label: str, data: Any, file_name: str, **kwargs: Any
        ) -> bool:
            key = kwargs.get("key", f"download_{file_name}")
            clicked = kwargs.get("_test_clicked", False)

            self._track_interaction(
                "download_button",
                {
                    "label": label,
                    "key": key,
                    "file_name": file_name,
                    "data": data,
                },
            )
            return clicked

        mock_st.file_uploader = Mock(side_effect=file_uploader_factory)
        mock_st.download_button = Mock(side_effect=download_button_factory)

    def _setup_layout_mocks(self, mock_st: Mock) -> None:
        """Setup layout component mocks."""

        class MockContainer:
            def __enter__(self) -> "MockContainer":
                return self

            def __exit__(self, *args: Any) -> None:
                pass

        def columns_factory(spec: int | list[int]) -> list[MockContainer]:
            if isinstance(spec, list):
                return [MockContainer() for _ in spec]
            return [MockContainer() for _ in range(spec)]

        mock_st.container = Mock(return_value=MockContainer())
        mock_st.expander = Mock(return_value=MockContainer())
        mock_st.columns = Mock(side_effect=columns_factory)
        mock_st.sidebar = Mock()

    def _setup_display_mocks(self, mock_st: Mock) -> None:
        """Setup display component mocks."""
        mock_st.write = Mock()
        mock_st.markdown = Mock()
        mock_st.text = Mock()
        mock_st.caption = Mock()
        mock_st.subheader = Mock()
        mock_st.header = Mock()
        mock_st.title = Mock()
        mock_st.info = Mock()
        mock_st.success = Mock()
        mock_st.warning = Mock()
        mock_st.error = Mock()
        mock_st.image = Mock()
        mock_st.dataframe = Mock()
        mock_st.table = Mock()
        mock_st.json = Mock()
        mock_st.progress = Mock()

        class MockContainer:
            def __enter__(self) -> "MockContainer":
                return self

            def __exit__(self, *args: Any) -> None:
                pass

        mock_st.spinner = Mock(return_value=MockContainer())
        mock_st.empty = Mock()
        mock_st.rerun = Mock()

    def _track_interaction(
        self, interaction_type: str, data: dict[str, Any]
    ) -> None:
        """Track UI interactions for testing analysis."""
        if self.config.performance_tracking:
            self._interaction_history.append(
                {
                    "type": interaction_type,
                    "timestamp": time.time(),
                    "data": data,
                }
            )

    def simulate_file_upload(self, key: str, file_data: Any) -> None:
        """Simulate file upload for testing."""
        self._file_uploads[key] = file_data

    def get_interaction_history(self) -> list[dict[str, Any]]:
        """Get history of UI interactions."""
        return self._interaction_history.copy()

    def clear_interaction_history(self) -> None:
        """Clear interaction history."""
        self._interaction_history.clear()


class AutomatedUITester:
    """Automated UI interaction testing system."""

    def __init__(self, mocker: EnhancedStreamlitMocker) -> None:
        self.mocker = mocker
        self._test_scenarios: list[dict[str, Any]] = []

    def add_test_scenario(
        self,
        name: str,
        interactions: list[dict[str, Any]],
        expected_state: dict[str, Any] | None = None,
    ) -> None:
        """Add a test scenario with UI interactions."""
        self._test_scenarios.append(
            {
                "name": name,
                "interactions": interactions,
                "expected_state": expected_state or {},
            }
        )

    def execute_scenario(
        self, scenario_name: str, mock_st: Mock
    ) -> TestInteractionResult:
        """Execute a test scenario and return results."""
        scenario = next(
            (s for s in self._test_scenarios if s["name"] == scenario_name),
            None,
        )

        if not scenario:
            return TestInteractionResult(
                success=False,
                interaction_type="scenario_execution",
                element_id=None,
                execution_time=0.0,
                error_message=f"Scenario '{scenario_name}' not found",
            )

        start_time = time.time()
        try:
            # Execute interactions
            for interaction in scenario["interactions"]:
                self._execute_interaction(interaction, mock_st)

            # Validate expected state
            expected_state = scenario.get("expected_state", {})
            current_state = dict(self.mocker._mock_session_state)

            success = self._validate_state(expected_state, current_state)
            execution_time = time.time() - start_time

            return TestInteractionResult(
                success=success,
                interaction_type="scenario_execution",
                element_id=scenario_name,
                execution_time=execution_time,
                captured_state=current_state,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestInteractionResult(
                success=False,
                interaction_type="scenario_execution",
                element_id=scenario_name,
                execution_time=execution_time,
                error_message=str(e),
            )

    def _execute_interaction(
        self, interaction: dict[str, Any], mock_st: Mock
    ) -> None:
        """Execute a single UI interaction."""
        interaction_type = interaction["type"]
        params = interaction.get("params", {})

        if interaction_type == "button_click":
            # Simulate button click
            label = params["label"]
            mock_st.button(label, _test_clicked=True, **params)

        elif interaction_type == "text_input":
            # Simulate text input
            label = params["label"]
            value = params["value"]
            mock_st.text_input(label, value=value, **params)

        elif interaction_type == "selectbox":
            # Simulate selectbox selection
            label = params["label"]
            options = params["options"]
            index = params.get("index", 0)
            mock_st.selectbox(label, options, index=index, **params)

        elif interaction_type == "file_upload":
            # Simulate file upload
            key = params["key"]
            file_data = params["file_data"]
            self.mocker.simulate_file_upload(key, file_data)

        # Add delay if configured
        if self.mocker.config.widget_interaction_delay > 0:
            time.sleep(self.mocker.config.widget_interaction_delay)

    def _validate_state(
        self, expected: dict[str, Any], actual: dict[str, Any]
    ) -> bool:
        """Validate expected state against actual state."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
        return True


class PerformanceTestingSuite:
    """Performance testing utilities for GUI components."""

    @staticmethod
    def benchmark_component_render(
        component_func: Callable[..., Any], iterations: int = 100
    ) -> dict[str, float]:
        """Benchmark component rendering performance."""
        render_times = []

        for _ in range(iterations):
            start_time = time.time()
            try:
                component_func()
            except Exception:
                pass  # Ignore errors for performance testing
            render_times.append(time.time() - start_time)

        return {
            "mean_time": sum(render_times) / len(render_times),
            "min_time": min(render_times),
            "max_time": max(render_times),
            "total_time": sum(render_times),
        }

    @staticmethod
    def measure_memory_usage(func: Callable[..., Any]) -> dict[str, Any]:
        """Measure memory usage during function execution."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        start_time = time.time()
        result = func()
        execution_time = time.time() - start_time

        memory_after = process.memory_info().rss
        memory_diff = memory_after - memory_before

        return {
            "result": result,
            "execution_time": execution_time,
            "memory_before_mb": memory_before / 1024 / 1024,
            "memory_after_mb": memory_after / 1024 / 1024,
            "memory_diff_mb": memory_diff / 1024 / 1024,
        }


def enhanced_streamlit_test(
    config: StreamlitTestConfig | None = None,
) -> Callable[[F], F]:
    """Decorator for enhanced Streamlit component testing."""

    def decorator(test_func: F) -> F:
        @functools.wraps(test_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            test_config = config or StreamlitTestConfig()
            mocker = EnhancedStreamlitMocker(test_config)

            # Create enhanced mock and inject into test
            mock_st = mocker.create_enhanced_streamlit_mock()

            # Add mocker and mock to test arguments
            kwargs["mock_streamlit"] = mock_st
            kwargs["streamlit_mocker"] = mocker

            return test_func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def streamlit_test_environment(
    config: StreamlitTestConfig | None = None,
) -> Any:
    """Context manager for Streamlit testing environment."""
    test_config = config or StreamlitTestConfig()
    mocker = EnhancedStreamlitMocker(test_config)

    with patch("streamlit", mocker.create_enhanced_streamlit_mock()):
        yield {
            "mocker": mocker,
            "ui_tester": AutomatedUITester(mocker),
            "performance": PerformanceTestingSuite(),
        }


# Utility functions for common testing patterns
def create_sample_project_structure(base_path: Path) -> dict[str, Path]:
    """Create a sample project structure for testing."""
    structure = {
        "config_dir": base_path / "configs",
        "data_dir": base_path / "data",
        "models_dir": base_path / "models",
        "outputs_dir": base_path / "outputs",
    }

    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)

    # Create sample files
    (structure["config_dir"] / "sample.yaml").write_text(
        "model:\n  type: test\n  name: sample\n"
    )

    return structure


def assert_streamlit_interaction(
    mock_st: Mock, interaction_type: str, expected_calls: int = 1
) -> None:
    """Assert that a specific Streamlit interaction occurred."""
    interaction_map = {
        "button": mock_st.button,
        "selectbox": mock_st.selectbox,
        "text_input": mock_st.text_input,
        "file_uploader": mock_st.file_uploader,
        "download_button": mock_st.download_button,
        "write": mock_st.write,
        "markdown": mock_st.markdown,
    }

    if interaction_type not in interaction_map:
        raise ValueError(f"Unknown interaction type: {interaction_type}")

    mock_method = interaction_map[interaction_type]
    assert mock_method.call_count >= expected_calls, (
        f"Expected at least {expected_calls} {interaction_type} calls, "
        f"got {mock_method.call_count}"
    )
