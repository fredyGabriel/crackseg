"""Unit tests for environment variable utilities."""

import os

import pytest

from src.utils import get_env_var, load_env

# --- Fixtures ---


@pytest.fixture
def test_env_setup_teardown():
    # Setup: Store original values if they exist
    original_my_var = os.environ.get("MY_VAR")
    original_other_var = os.environ.get("OTHER_VAR")
    original_existing_var = os.environ.get("EXISTING_VAR")

    # Ensure vars are clean before tests that set them
    if "MY_VAR" in os.environ:
        del os.environ["MY_VAR"]
    if "OTHER_VAR" in os.environ:
        del os.environ["OTHER_VAR"]
    if "EXISTING_VAR" in os.environ:
        del os.environ["EXISTING_VAR"]

    yield  # Run the tests

    # Teardown: Restore original values or remove test vars
    if original_my_var is not None:
        os.environ["MY_VAR"] = original_my_var
    elif "MY_VAR" in os.environ:
        del os.environ["MY_VAR"]

    if original_other_var is not None:
        os.environ["OTHER_VAR"] = original_other_var
    elif "OTHER_VAR" in os.environ:
        del os.environ["OTHER_VAR"]

    if original_existing_var is not None:
        os.environ["EXISTING_VAR"] = original_existing_var
    elif "EXISTING_VAR" in os.environ:
        del os.environ["EXISTING_VAR"]


# --- Test Cases ---


def test_load_env_file(tmp_path, test_env_setup_teardown):
    """Test loading variables from a specific .env file."""
    # Create a dummy .env file
    env_content = "MY_VAR=test_value\nOTHER_VAR=123"
    env_file = tmp_path / ".env_test"
    env_file.write_text(env_content)

    load_env(dotenv_path=str(env_file))

    assert os.getenv("MY_VAR") == "test_value"
    assert os.getenv("OTHER_VAR") == "123"


def test_get_env_var(test_env_setup_teardown):
    """Test getting existing and non-existing env variables."""
    os.environ["EXISTING_VAR"] = "exists"

    assert get_env_var("EXISTING_VAR") == "exists"
    assert get_env_var("NON_EXISTING_VAR") is None
    assert (
        get_env_var("NON_EXISTING_VAR", default="default_val") == "default_val"
    )
