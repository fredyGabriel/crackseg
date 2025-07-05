"""Basic unit tests for E2E utility modules.

Tests import functionality and basic instantiation of utility functions
to ensure the modules are correctly structured and dependencies are satisfied.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_element_utils_import() -> None:
    """Test that element utilities can be imported."""
    from tests.e2e.utils.element import (
        click_element_safely,
        find_element_with_retry,
        wait_for_element_visible,
    )

    # Test that functions are callable
    assert callable(find_element_with_retry)
    assert callable(wait_for_element_visible)
    assert callable(click_element_safely)


def test_data_utils_import() -> None:
    """Test that data utilities can be imported."""
    from tests.e2e.utils.data import (
        generate_random_email,
        generate_random_string,
        get_sample_config_data,
    )

    # Test that functions are callable
    assert callable(generate_random_string)
    assert callable(generate_random_email)
    assert callable(get_sample_config_data)


def test_file_utils_import() -> None:
    """Test that file utilities can be imported."""
    from tests.e2e.utils.file import (
        ensure_artifacts_dir,
        read_test_config,
        validate_file_exists,
    )

    # Test that functions are callable
    assert callable(ensure_artifacts_dir)
    assert callable(validate_file_exists)
    assert callable(read_test_config)


def test_string_utils_import() -> None:
    """Test that string utilities can be imported."""
    from tests.e2e.utils.string import (
        clean_element_text,
        normalize_whitespace,
        sanitize_filename,
    )

    # Test that functions are callable
    assert callable(normalize_whitespace)
    assert callable(clean_element_text)
    assert callable(sanitize_filename)


def test_time_utils_import() -> None:
    """Test that time utilities can be imported."""
    from tests.e2e.utils.time import (
        format_duration,
        get_current_timestamp,
        wait_with_timeout,
    )

    # Test that functions are callable
    assert callable(get_current_timestamp)
    assert callable(format_duration)
    assert callable(wait_with_timeout)


def test_config_utils_import() -> None:
    """Test that config utilities can be imported."""
    from tests.e2e.utils.config import (
        get_browser_config,
        get_streamlit_test_config,
        merge_configs,
    )

    # Test that functions are callable
    assert callable(get_browser_config)
    assert callable(get_streamlit_test_config)
    assert callable(merge_configs)


def test_streamlit_utils_import() -> None:
    """Test that streamlit utilities can be imported."""
    from tests.e2e.utils.streamlit import (
        click_streamlit_button,
        navigate_to_page,
        wait_for_streamlit_ready,
    )

    # Test that functions are callable
    assert callable(wait_for_streamlit_ready)
    assert callable(navigate_to_page)
    assert callable(click_streamlit_button)


def test_utils_init_import() -> None:
    """Test that the main utils module can be imported."""
    import tests.e2e.utils as e2e_utils

    # Test that key functions are available from main module
    assert hasattr(e2e_utils, "generate_random_string")
    assert hasattr(e2e_utils, "get_browser_config")
    assert hasattr(e2e_utils, "click_element_safely")
    assert hasattr(e2e_utils, "wait_for_streamlit_ready")


def test_data_generation_basic() -> None:
    """Test basic data generation functions."""
    from tests.e2e.utils.data import (
        generate_random_email,
        generate_random_string,
    )

    # Test string generation
    test_string = generate_random_string(length=10)
    assert len(test_string) == 10
    assert isinstance(test_string, str)

    # Test email generation
    test_email = generate_random_email()
    assert "@" in test_email
    assert "." in test_email
    assert isinstance(test_email, str)


def test_config_generation_basic() -> None:
    """Test basic config generation functions."""
    from tests.e2e.utils.config import (
        get_browser_config,
        get_streamlit_test_config,
    )

    # Test browser config
    browser_config = get_browser_config("chrome", headless=True)
    assert isinstance(browser_config, dict)
    assert browser_config["browser"] == "chrome"
    assert browser_config["headless"] is True

    # Test streamlit config
    streamlit_config = get_streamlit_test_config(port=8502)
    assert isinstance(streamlit_config, dict)
    assert streamlit_config["streamlit"]["port"] == 8502


def test_string_manipulation_basic() -> None:
    """Test basic string manipulation functions."""
    from tests.e2e.utils.string import normalize_whitespace, sanitize_filename

    # Test whitespace normalization
    normalized = normalize_whitespace("  hello   world  ")
    assert normalized == "hello world"

    # Test filename sanitization
    sanitized = sanitize_filename("test/file:name*.txt")
    assert "/" not in sanitized
    assert ":" not in sanitized
    assert "*" not in sanitized


def test_time_utilities_basic() -> None:
    """Test basic time utility functions."""
    from tests.e2e.utils.time import format_duration, get_current_timestamp

    # Test timestamp generation
    timestamp = get_current_timestamp()
    assert isinstance(timestamp, str)
    assert len(timestamp) > 0

    # Test duration formatting
    duration = format_duration(3661)  # 1 hour, 1 minute, 1 second
    assert "1h" in duration or "1:01:01" in duration
    assert isinstance(duration, str)


@patch("tests.e2e.utils.file.Path.exists")
def test_file_utilities_basic(mock_exists: MagicMock) -> None:
    """Test basic file utility functions."""
    from tests.e2e.utils.file import ensure_artifacts_dir, validate_file_exists

    # Test file validation
    mock_exists.return_value = True
    assert validate_file_exists("dummy_path.txt") is True

    mock_exists.return_value = False
    assert validate_file_exists("nonexistent.txt") is False

    # Test artifacts directory creation (with mock)
    with patch("tests.e2e.utils.file.Path.mkdir"):
        artifacts_path = ensure_artifacts_dir("test-artifacts")
        assert isinstance(artifacts_path, Path)
        assert str(artifacts_path).endswith("test-artifacts")
