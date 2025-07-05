"""Configuration and fixtures for E2E testing with Selenium WebDriver.

This module provides pytest fixtures for browser initialization, test data
setup, and resource management, leveraging the HybridDriverManager system
from subtask 14.1. Supports both Docker Grid and local WebDriver setups
with automatic cleanup.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from selenium.webdriver.remote.webdriver import WebDriver

from .drivers import (
    DriverConfig,
    HybridDriverManager,
    driver_session,
)

# =============================================================================
# Browser Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def e2e_config() -> DriverConfig:
    """Provide default E2E testing configuration.

    Creates a session-scoped configuration optimized for E2E testing with
    reasonable defaults for headless browser testing, timeouts, and artifact
    management.

    Returns:
        DriverConfig: Default configuration for E2E testing
    """
    return DriverConfig(
        browser="chrome",
        headless=True,
        window_size=(1920, 1080),
        implicit_wait=10.0,
        page_load_timeout=30.0,
        screenshot_on_failure=True,
        artifacts_dir=Path("test-artifacts") / "e2e",
        enable_logging=True,
        log_level="INFO",
    )


@pytest.fixture(scope="session")
def chrome_config(e2e_config: DriverConfig) -> DriverConfig:
    """Provide Chrome-specific configuration for E2E testing.

    Args:
        e2e_config: Base E2E configuration

    Returns:
        DriverConfig: Chrome-optimized configuration
    """
    return DriverConfig(
        browser="chrome",
        **{k: v for k, v in e2e_config.to_dict().items() if k != "browser"},
    )


@pytest.fixture(scope="session")
def firefox_config(e2e_config: DriverConfig) -> DriverConfig:
    """Provide Firefox-specific configuration for E2E testing.

    Args:
        e2e_config: Base E2E configuration

    Returns:
        DriverConfig: Firefox-optimized configuration
    """
    return DriverConfig(
        browser="firefox",
        **{k: v for k, v in e2e_config.to_dict().items() if k != "browser"},
    )


@pytest.fixture(scope="session")
def edge_config(e2e_config: DriverConfig) -> DriverConfig:
    """Provide Edge-specific configuration for E2E testing.

    Args:
        e2e_config: Base E2E configuration

    Returns:
        DriverConfig: Edge-optimized configuration
    """
    return DriverConfig(
        browser="edge",
        **{k: v for k, v in e2e_config.to_dict().items() if k != "browser"},
    )


# =============================================================================
# Driver Manager Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def driver_manager(
    e2e_config: DriverConfig,
) -> Generator[HybridDriverManager, None, None]:
    """Provide session-scoped HybridDriverManager instance.

    Creates and manages a HybridDriverManager instance for the entire test
    session, with automatic cleanup of all managed drivers.

    Args:
        e2e_config: E2E testing configuration

    Yields:
        HybridDriverManager: Configured driver manager
    """
    manager = HybridDriverManager(e2e_config)
    try:
        yield manager
    finally:
        # Cleanup all active drivers
        manager.cleanup_all_drivers()


# =============================================================================
# WebDriver Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def chrome_driver(
    chrome_config: DriverConfig,
) -> Generator[WebDriver, None, None]:
    """Provide Chrome WebDriver instance for individual test functions.

    Creates a fresh Chrome WebDriver for each test function with automatic
    cleanup after test completion.

    Args:
        chrome_config: Chrome-specific configuration

    Yields:
        WebDriver: Chrome WebDriver instance
    """
    with driver_session(browser="chrome", config=chrome_config) as driver:
        yield driver


@pytest.fixture(scope="function")
def firefox_driver(
    firefox_config: DriverConfig,
) -> Generator[WebDriver, None, None]:
    """Provide Firefox WebDriver instance for individual test functions.

    Creates a fresh Firefox WebDriver for each test function with automatic
    cleanup after test completion.

    Args:
        firefox_config: Firefox-specific configuration

    Yields:
        WebDriver: Firefox WebDriver instance
    """
    with driver_session(browser="firefox", config=firefox_config) as driver:
        yield driver


@pytest.fixture(scope="function")
def edge_driver(edge_config: DriverConfig) -> Generator[WebDriver, None, None]:
    """Provide Edge WebDriver instance for individual test functions.

    Creates a fresh Edge WebDriver for each test function with automatic
    cleanup after test completion.

    Args:
        edge_config: Edge-specific configuration

    Yields:
        WebDriver: Edge WebDriver instance
    """
    with driver_session(browser="edge", config=edge_config) as driver:
        yield driver


@pytest.fixture(scope="function")
def webdriver(
    driver_manager: HybridDriverManager,
) -> Generator[WebDriver, None, None]:
    """Provide default WebDriver instance using driver manager.

    Creates a WebDriver using the session-scoped driver manager with the
    default browser configuration. Useful for tests that don't require
    specific browser types.

    Args:
        driver_manager: Session-scoped driver manager

    Yields:
        WebDriver: Default WebDriver instance
    """
    with driver_manager.get_driver() as driver:
        yield driver


# =============================================================================
# Cross-Browser Testing Fixtures
# =============================================================================


@pytest.fixture(
    scope="function",
    params=["chrome", "firefox", "edge"],
    ids=["chrome", "firefox", "edge"],
)
def cross_browser_driver(
    request: pytest.FixtureRequest, e2e_config: DriverConfig
) -> Generator[WebDriver, None, None]:
    """Provide WebDriver for cross-browser testing.

    Parametrized fixture that runs tests across multiple browsers. Useful
    for ensuring Streamlit application compatibility across different browsers.

    Args:
        request: Pytest request object containing browser parameter
        e2e_config: Base E2E configuration

    Yields:
        WebDriver: Browser-specific WebDriver instance
    """
    browser = request.param
    config = DriverConfig(
        browser=browser,  # Controlled by params
        **{k: v for k, v in e2e_config.to_dict().items() if k != "browser"},
    )

    with driver_session(browser=browser, config=config) as driver:
        yield driver


# =============================================================================
# Test Data and Utility Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def streamlit_base_url() -> str:
    """Provide base URL for Streamlit application during testing.

    Returns:
        str: Base URL for accessing the Streamlit application
    """
    return "http://localhost:8501"


@pytest.fixture(scope="session")
def test_artifacts_dir(e2e_config: DriverConfig) -> Path:
    """Provide test artifacts directory with automatic creation.

    Creates and provides the test artifacts directory path for storing
    screenshots, logs, and other test artifacts.

    Args:
        e2e_config: E2E configuration containing artifacts directory

    Returns:
        Path: Test artifacts directory path
    """
    artifacts_dir = e2e_config.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


@pytest.fixture(scope="function")
def test_data() -> dict[str, Any]:
    """Provide test data for Streamlit application testing.

    Creates mock test data suitable for testing crack segmentation
    application functionality.

    Returns:
        dict[str, Any]: Test data dictionary
    """
    return {
        "sample_images": [
            {"name": "crack_sample_1.jpg", "type": "crack"},
            {"name": "crack_sample_2.jpg", "type": "crack"},
            {"name": "no_crack_sample.jpg", "type": "no_crack"},
        ],
        "config_values": {
            "model_name": "test_model",
            "batch_size": 4,
            "confidence_threshold": 0.5,
        },
        "expected_results": {
            "navigation_elements": [
                "Architecture",
                "Configuration",
                "Training",
                "Evaluation",
            ],
            "file_upload_types": [".jpg", ".jpeg", ".png"],
        },
    }


# =============================================================================
# Resource Management and Cleanup Fixtures
# =============================================================================


@pytest.fixture(scope="function", autouse=True)
def cleanup_test_artifacts(
    test_artifacts_dir: Path,
) -> Generator[None, None, None]:
    """Automatically cleanup test artifacts after each test function.

    Autouse fixture that ensures test artifacts are cleaned up after each
    test function to prevent accumulation of test files.

    Args:
        test_artifacts_dir: Test artifacts directory

    Yields:
        None
    """
    yield

    # Cleanup screenshots and logs from individual test
    test_files = test_artifacts_dir.glob("test_*")
    for test_file in test_files:
        if test_file.is_file():
            try:
                test_file.unlink()
            except (OSError, PermissionError):
                # Log but don't fail test due to cleanup issues
                pass


@pytest.fixture(scope="function")
def screenshot_on_failure(
    request: pytest.FixtureRequest, test_artifacts_dir: Path
) -> Generator[None, None, None]:
    """Automatically capture screenshot on test failure.

    Fixture that captures a screenshot when a test fails, useful for
    debugging E2E test failures.

    Args:
        request: Pytest request object
        test_artifacts_dir: Directory for storing artifacts

    Yields:
        None
    """
    yield

    # Check if test failed and there's an active WebDriver
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        # Try to get WebDriver from any of the possible fixtures
        driver = None
        for fixture_name in [
            "webdriver",
            "chrome_driver",
            "firefox_driver",
            "edge_driver",
            "cross_browser_driver",
        ]:
            try:
                driver = request.getfixturevalue(fixture_name)
                break
            except Exception:
                continue

        if driver:
            # Capture screenshot
            test_name = request.node.name
            screenshot_path = test_artifacts_dir / f"failure_{test_name}.png"
            try:
                driver.save_screenshot(str(screenshot_path))
            except Exception:
                # Don't fail test due to screenshot issues
                pass


# =============================================================================
# Pytest Hooks and Configuration
# =============================================================================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Any:
    """Capture test result for screenshot and video capture integration.

    This hook captures the test result and integrates with the capture system
    to automatically take screenshots and save videos on test failures.
    """

    # Execute the test
    outcome = yield
    rep = outcome.get_result()

    # Only handle test failures during the test call phase
    if rep.when == "call" and rep.failed:
        # Get the test instance if it's a method
        test_instance = getattr(item, "instance", None)
        if test_instance and hasattr(test_instance, "on_test_failure"):
            # Extract driver from pytest fixtures
            driver = None
            funcargs = getattr(item, "funcargs", {})
            if isinstance(funcargs, dict) and "webdriver" in funcargs:
                driver = funcargs["webdriver"]
            elif hasattr(test_instance, "_driver"):
                driver = test_instance._driver
            elif hasattr(test_instance, "driver"):
                driver = test_instance.driver

            if driver and hasattr(rep, "longrepr"):
                # Create exception from test failure
                exception = Exception(str(rep.longrepr))

                try:
                    # Call the test instance's failure handler
                    test_instance.on_test_failure(driver, exception)
                except Exception as e:
                    print(f"Warning: Failed to capture failure evidence: {e}")

    # Store result for cleanup decisions
    test_instance = getattr(item, "instance", None)
    if test_instance and hasattr(test_instance, "_pytest_passed"):
        test_instance._pytest_passed = not rep.failed

    return rep


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Setup hook to initialize capture system for each test.

    Automatically starts video recording if configured in test data.
    """
    test_instance = getattr(item, "instance", None)
    if test_instance and hasattr(test_instance, "start_test_recording"):
        # Get driver from pytest fixtures if available
        driver = None
        funcargs = getattr(item, "funcargs", {})
        if isinstance(funcargs, dict) and "webdriver" in funcargs:
            driver = funcargs["webdriver"]
        elif hasattr(test_instance, "_driver"):
            driver = test_instance._driver
        elif hasattr(test_instance, "driver"):
            driver = test_instance.driver

        if driver:
            try:
                # Check if video recording should be enabled for this test
                test_data = getattr(test_instance, "_test_data", {})
                capture_config = test_data.get("capture_config", {})
                video_config = capture_config.get("videos", {})

                if video_config.get("record_all", False):
                    test_instance.start_test_recording(driver)
            except Exception as e:
                print(f"Warning: Failed to start test recording: {e}")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Teardown hook to cleanup capture artifacts after each test.

    Stops video recording and applies retention policies.
    """
    test_instance = getattr(item, "instance", None)
    if test_instance and hasattr(test_instance, "stop_test_recording"):
        try:
            # Stop recording if active
            if (
                hasattr(test_instance, "is_recording")
                and test_instance.is_recording()
            ):
                # Only save video if test failed or configured to save all
                test_passed = not getattr(
                    test_instance, "_pytest_failed", False
                )
                save_video = not test_passed  # Save on failure by default

                # Check configuration for save policy
                test_data = getattr(test_instance, "_test_data", {})
                capture_config = test_data.get("capture_config", {})
                video_config = capture_config.get("videos", {})

                if video_config.get("record_all", False):
                    save_video = (
                        True  # Save all recordings if record_all is enabled
                    )

                test_instance.stop_test_recording(save_video)
        except Exception as e:
            print(f"Warning: Failed to stop test recording: {e}")


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with E2E-specific markers and settings.

    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end integration test"
    )
    config.addinivalue_line("markers", "chrome: mark test as Chrome-specific")
    config.addinivalue_line(
        "markers", "firefox: mark test as Firefox-specific"
    )
    config.addinivalue_line("markers", "edge: mark test as Edge-specific")
    config.addinivalue_line(
        "markers", "cross_browser: mark test for cross-browser execution"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running (requires extended timeout)",
    )
