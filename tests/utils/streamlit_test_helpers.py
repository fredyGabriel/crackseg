"""Streamlit-specific test helpers and advanced mocking strategies.

This module provides specialized helpers for Streamlit component testing,
addressing common issues identified in the GUI test coverage analysis.
Part of subtask 7.6 - GUI Testing Framework Enhancement.
"""

import io
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


class StreamlitComponentTestFixture:
    """Reusable test fixture for Streamlit components."""

    def __init__(self, temp_dir: Path | None = None) -> None:
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.mock_files: dict[str, Any] = {}
        self.session_state_history: list[dict[str, Any]] = []

    def create_mock_uploaded_file(
        self,
        file_name: str,
        content: str | bytes,
        mime_type: str = "text/plain",
    ) -> Mock:
        """Create a mock uploaded file for testing file upload components."""
        mock_file = Mock()
        mock_file.name = file_name
        mock_file.type = mime_type
        mock_file.size = len(content)

        # Remember original type for IO behavior
        is_string_content = isinstance(content, str)

        if isinstance(content, str):
            content = content.encode("utf-8")

        mock_file.read = Mock(return_value=content)
        mock_file.getvalue = Mock(return_value=content)
        mock_file.seek = Mock(return_value=None)
        mock_file.tell = Mock(return_value=0)

        # Create StringIO/BytesIO behavior based on original type
        if is_string_content:
            mock_file.file = io.StringIO(content.decode("utf-8"))
        else:
            mock_file.file = io.BytesIO(content)

        return mock_file

    def create_sample_config_file(self, config_data: dict[str, Any]) -> Path:
        """Create a sample configuration file for testing."""
        config_file = self.temp_dir / "test_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        return config_file

    def create_sample_image_files(self, count: int = 3) -> list[Path]:
        """Create sample image files for testing."""
        image_files = []
        for i in range(count):
            # Create a simple 1x1 pixel image file
            image_file = self.temp_dir / f"test_image_{i}.png"
            # Simple PNG header for a 1x1 pixel image
            png_data = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
                b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
                b"\xdd\xb1\x1dB\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            image_file.write_bytes(png_data)
            image_files.append(image_file)
        return image_files

    def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


class StreamlitSessionStateMocker:
    """Advanced session state mocker addressing common session state issues."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._change_history: list[dict[str, Any]] = []

    def create_session_state_mock(self) -> Mock:
        """Create a comprehensive session state mock."""

        class SessionStateMock:
            def __init__(self, parent: "StreamlitSessionStateMocker") -> None:
                self._parent = parent

            def __getitem__(self, key: str) -> Any:
                return self._parent._data[key]

            def __setitem__(self, key: str, value: Any) -> None:
                old_value = self._parent._data.get(key)
                self._parent._data[key] = value
                self._parent._change_history.append(
                    {
                        "action": "set",
                        "key": key,
                        "old_value": old_value,
                        "new_value": value,
                    }
                )

            def __contains__(self, key: str) -> bool:
                return key in self._parent._data

            def __getattr__(self, name: str) -> Any:
                if name.startswith("_"):
                    return object.__getattribute__(self, name)
                return self._parent._data.get(name)

            def __setattr__(self, name: str, value: Any) -> None:
                if name.startswith("_"):
                    object.__setattr__(self, name, value)
                else:
                    old_value = self._parent._data.get(name)
                    self._parent._data[name] = value
                    self._parent._change_history.append(
                        {
                            "action": "setattr",
                            "key": name,
                            "old_value": old_value,
                            "new_value": value,
                        }
                    )

            def get(self, key: str, default: Any = None) -> Any:
                return self._parent._data.get(key, default)

            def setdefault(self, key: str, default: Any = None) -> Any:
                if key not in self._parent._data:
                    self._parent._data[key] = default
                    self._parent._change_history.append(
                        {
                            "action": "setdefault",
                            "key": key,
                            "old_value": None,
                            "new_value": default,
                        }
                    )
                return self._parent._data[key]

            def pop(self, key: str, default: Any = None) -> Any:
                value = self._parent._data.pop(key, default)
                self._parent._change_history.append(
                    {
                        "action": "pop",
                        "key": key,
                        "old_value": value,
                        "new_value": None,
                    }
                )
                return value

            def clear(self) -> None:
                old_data = self._parent._data.copy()
                self._parent._data.clear()
                self._parent._change_history.append(
                    {
                        "action": "clear",
                        "key": None,
                        "old_value": old_data,
                        "new_value": {},
                    }
                )

            def keys(self) -> Any:
                return self._parent._data.keys()

            def values(self) -> Any:
                return self._parent._data.values()

            def items(self) -> Any:
                return self._parent._data.items()

            def update(self, other: dict[str, Any]) -> None:
                old_data = self._parent._data.copy()
                self._parent._data.update(other)
                self._parent._change_history.append(
                    {
                        "action": "update",
                        "key": None,
                        "old_value": old_data,
                        "new_value": self._parent._data.copy(),
                    }
                )

        return SessionStateMock(self)

    def get_change_history(self) -> list[dict[str, Any]]:
        """Get the history of session state changes."""
        return self._change_history.copy()

    def assert_key_changed(self, key: str, expected_value: Any = None) -> None:
        """Assert that a specific key was changed in session state."""
        key_changes = [
            change
            for change in self._change_history
            if change.get("key") == key
        ]
        assert len(key_changes) > 0, f"Key '{key}' was not changed"

        if expected_value is not None:
            final_change = key_changes[-1]
            assert final_change["new_value"] == expected_value, (
                f"Expected '{key}' to be '{expected_value}', "
                f"got '{final_change['new_value']}'"
            )

    def reset_history(self) -> None:
        """Reset the change history."""
        self._change_history.clear()


class StreamlitWidgetMocker:
    """Specialized widget mocker addressing common widget testing issues."""

    @staticmethod
    def create_file_uploader_mock(uploaded_files: list[Mock]) -> Mock:
        """Create a file uploader mock with proper behavior."""

        def file_uploader_side_effect(
            label: str, accept_multiple_files: bool = False, **kwargs: Any
        ) -> Mock | list[Mock] | None:
            if not uploaded_files:
                return None

            if accept_multiple_files:
                return uploaded_files
            else:
                return uploaded_files[0] if uploaded_files else None

        return Mock(side_effect=file_uploader_side_effect)

    @staticmethod
    def create_download_button_mock(simulate_click: bool = False) -> Mock:
        """Create a download button mock with click simulation."""

        def download_button_side_effect(
            label: str, data: Any, file_name: str, **kwargs: Any
        ) -> bool:
            # Simulate validation of download data
            if data is None:
                raise ValueError("Download data cannot be None")
            return simulate_click

        return Mock(side_effect=download_button_side_effect)

    @staticmethod
    def create_form_mock() -> Mock:
        """Create a form mock with proper context manager behavior."""

        class FormMock:
            def __init__(self) -> None:
                self.submitted = False
                self.form_submit_button = Mock(return_value=False)

            def __enter__(self) -> "FormMock":
                return self

            def __exit__(self, *args: Any) -> None:
                pass

        return FormMock()


class StreamlitErrorTestHelper:
    """Helper for testing error handling in Streamlit components."""

    @staticmethod
    def create_error_scenario_mock(error_type: type[Exception]) -> Mock:
        """Create a mock that raises a specific error type."""

        def error_side_effect(*args: Any, **kwargs: Any) -> None:
            raise error_type("Simulated error for testing")

        return Mock(side_effect=error_side_effect)

    @staticmethod
    def assert_error_displayed(
        mock_st: Mock, error_function: str = "error"
    ) -> None:
        """Assert that an error was displayed using Streamlit error
        functions."""
        error_methods = ["error", "exception", "warning"]

        if error_function == "any":
            # Check if any error display method was called
            for method_name in error_methods:
                if hasattr(mock_st, method_name):
                    method = getattr(mock_st, method_name)
                    if method.called:
                        return

            pytest.fail("No error display method was called")
        else:
            # Check specific error method
            if not hasattr(mock_st, error_function):
                pytest.fail(f"Mock doesn't have {error_function} method")

            error_method = getattr(mock_st, error_function)
            assert error_method.called, f"st.{error_function}() was not called"


class StreamlitConfigTestHelper:
    """Helper for testing configuration-related components."""

    @staticmethod
    def create_hydra_config_mock(config_dict: dict[str, Any]) -> Mock:
        """Create a mock for Hydra configuration objects."""
        mock_config = Mock()

        # Handle nested dictionary access
        def deep_getattr(obj: Mock, name: str) -> Any:
            if "." in name:
                parts = name.split(".")
                current = config_dict
                for part in parts:
                    current = current.get(part, {})
                return current
            return config_dict.get(name)

        mock_config.__getattr__ = deep_getattr

        # Support dictionary-style access
        mock_config.__getitem__ = lambda self, key: config_dict[key]
        mock_config.__contains__ = lambda self, key: key in config_dict
        mock_config.get = lambda self, key, default=None: config_dict.get(
            key, default
        )

        return mock_config

    @staticmethod
    def create_yaml_content_sample() -> str:
        """Create sample YAML content for testing."""
        return """
model:
  name: test_model
  type: unet
  encoder: resnet50

training:
  epochs: 10
  batch_size: 4
  learning_rate: 0.001

data:
  image_size: [512, 512]
  dataset_path: /path/to/dataset
"""


# Pytest fixtures for enhanced testing
@pytest.fixture
def streamlit_test_fixture() -> StreamlitComponentTestFixture:
    """Pytest fixture providing a complete Streamlit test environment."""
    fixture = StreamlitComponentTestFixture()
    yield fixture
    fixture.cleanup()


@pytest.fixture
def enhanced_session_state() -> StreamlitSessionStateMocker:
    """Pytest fixture providing enhanced session state testing."""
    return StreamlitSessionStateMocker()


@pytest.fixture
def sample_uploaded_files(
    streamlit_test_fixture: StreamlitComponentTestFixture,
) -> list[Mock]:
    """Pytest fixture providing sample uploaded files."""
    return [
        streamlit_test_fixture.create_mock_uploaded_file(
            "test_config.yaml",
            StreamlitConfigTestHelper.create_yaml_content_sample(),
        ),
        streamlit_test_fixture.create_mock_uploaded_file(
            "test_data.json", '{"key": "value", "number": 42}'
        ),
        streamlit_test_fixture.create_mock_uploaded_file(
            "test_image.png", b"\x89PNG...", "image/png"
        ),
    ]


# Integration with performance testing framework
def performance_test_component(
    component_func: callable, iterations: int = 50
) -> dict[str, float]:
    """Performance test a Streamlit component with error handling."""
    import time

    execution_times = []
    errors = 0

    for _ in range(iterations):
        start_time = time.time()
        try:
            component_func()
        except Exception:
            errors += 1
        execution_times.append(time.time() - start_time)

    if execution_times:
        return {
            "mean_time": sum(execution_times) / len(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "error_rate": errors / iterations,
            "successful_iterations": iterations - errors,
        }
    else:
        return {
            "mean_time": 0.0,
            "min_time": 0.0,
            "max_time": 0.0,
            "error_rate": 1.0,
            "successful_iterations": 0,
        }


# Utility for resolving common test failures
def debug_streamlit_test_failure(
    mock_st: Mock, expected_calls: dict[str, int]
) -> str:
    """Debug common Streamlit test failures and provide helpful error
    messages."""
    issues = []

    for method_name, expected_count in expected_calls.items():
        if hasattr(mock_st, method_name):
            method = getattr(mock_st, method_name)
            actual_count = method.call_count
            if actual_count != expected_count:
                issues.append(
                    f"{method_name}: expected {expected_count} calls, "
                    f"got {actual_count}"
                )
        else:
            issues.append(f"{method_name}: method not found in mock")

    if issues:
        return "Test failures detected:\n" + "\n".join(
            f"- {issue}" for issue in issues
        )
    else:
        return "No issues detected in mock call analysis"
