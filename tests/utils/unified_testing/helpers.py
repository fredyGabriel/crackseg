"""
Unified testing helpers consolidating functionality from various testing
modules. This module provides utility classes and functions for error
testing, configuration management, and common testing patterns.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock


class UnifiedErrorTester:
    """Unified error testing from streamlit_test_helpers.py."""

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
        """Assert that an error was displayed using Streamlit functions."""
        error_method = getattr(mock_st, error_function, None)

        if error_method is None:
            raise AssertionError(
                f"Error method '{error_function}' not found on mock object"
            )

        if not error_method.called:
            raise AssertionError(
                f"Expected {error_function} to be called but it wasn't"
            )

    @staticmethod
    def capture_error_from_exception(
        func: Callable, *args: Any, **kwargs: Any
    ) -> Exception | None:
        """Capture and return exception from function execution."""
        try:
            func(*args, **kwargs)
            return None
        except Exception as e:
            return e

    @staticmethod
    def assert_specific_error_type(
        func: Callable,
        expected_error: type[Exception],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Assert that function raises specific error type."""
        try:
            func(*args, **kwargs)
            raise AssertionError(
                f"Expected {expected_error.__name__} to be raised"
            )
        except expected_error:
            pass  # Expected behavior
        except Exception as e:
            raise AssertionError(
                f"Expected {expected_error.__name__}, "
                f"got {type(e).__name__}: {e}"
            ) from e


class UnifiedConfigHelper:
    """Unified configuration testing from streamlit_test_helpers.py."""

    @staticmethod
    def create_hydra_config_mock(config_dict: dict[str, Any]) -> Mock:
        """Create a mock for Hydra configuration objects."""
        mock_config = Mock()

        # Handle nested dictionary access with proper typing
        def get_nested_value(path: str) -> Any:
            keys = path.split(".")
            value = config_dict
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value

        # Configure mock to return values for nested attributes
        def getattr_side_effect(name: str) -> Any:
            if name in config_dict:
                return config_dict[name]
            # Handle nested attributes like config.model.encoder
            nested_value = get_nested_value(name)
            if nested_value is not None:
                return nested_value
            # Return a new mock for intermediate attributes
            return Mock()

        mock_config.__getattr__ = Mock(side_effect=getattr_side_effect)

        # Handle common dict-like access patterns
        mock_config.__getitem__ = Mock(
            side_effect=lambda key: config_dict[key]
        )
        mock_config.__contains__ = Mock(
            side_effect=lambda key: key in config_dict
        )
        mock_config.get = Mock(
            side_effect=lambda key, default=None: config_dict.get(key, default)
        )

        return mock_config

    @staticmethod
    def create_yaml_content_sample() -> str:
        """Create sample YAML content for testing."""
        return """
model:
  encoder: resnet50
  decoder: unet
training:
  epochs: 100
  batch_size: 16
"""


# Utility functions for common testing patterns
def assert_streamlit_interaction(
    mock_st: Mock, interaction_type: str, expected_calls: int = 1
) -> None:
    """Assert that a specific Streamlit interaction occurred."""
    if not hasattr(mock_st, interaction_type):
        raise AssertionError(
            f"Mock Streamlit object does not have method '{interaction_type}'"
        )

    method = getattr(mock_st, interaction_type)
    if not method.called:
        raise AssertionError(
            f"Expected '{interaction_type}' to be called but it wasn't"
        )

    if method.call_count < expected_calls:
        raise AssertionError(
            f"Expected '{interaction_type}' to be called at least "
            f"{expected_calls} times, "
            f"but it was called {method.call_count} times"
        )


def create_test_session_state(
    initial_state: dict[str, Any] | None = None,
) -> Mock:
    """Create a mock session state for testing."""
    mock_session_state = Mock()
    state_dict = initial_state or {}

    # Configure mock to behave like a dictionary
    mock_session_state.__getitem__ = Mock(
        side_effect=lambda key: state_dict[key]
    )
    mock_session_state.__setitem__ = Mock(
        side_effect=lambda key, value: state_dict.update({key: value})
    )
    mock_session_state.__contains__ = Mock(
        side_effect=lambda key: key in state_dict
    )
    mock_session_state.get = Mock(
        side_effect=lambda key, default=None: state_dict.get(key, default)
    )

    return mock_session_state


def validate_widget_interaction(
    mock_widget: Mock,
    expected_label: str,
    expected_args: tuple[Any, ...] | None = None,
) -> None:
    """Validate that a widget was called with expected parameters."""
    if not mock_widget.called:
        raise AssertionError("Expected widget to be called but it wasn't")

    # Check if any call has the expected label
    found_label = False
    for call in mock_widget.call_args_list:
        args, kwargs = call
        if args and args[0] == expected_label:
            found_label = True
            break
        if "label" in kwargs and kwargs["label"] == expected_label:
            found_label = True
            break

    if not found_label:
        raise AssertionError(
            f"Expected widget to be called with label '{expected_label}'"
        )

    if expected_args is not None:
        # Verify that at least one call matches the expected arguments
        found_args = False
        for call in mock_widget.call_args_list:
            args, _ = call
            if args == expected_args:
                found_args = True
                break

        if not found_args:
            raise AssertionError(
                f"Expected widget to be called with args {expected_args}"
            )
