"""Basic E2E test for Streamlit GUI application.

This test validates that the Docker testing infrastructure works correctly
and that basic Streamlit functionality is accessible through Selenium.
"""

import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


@pytest.fixture(scope="session")
def streamlit_url() -> str:
    """Get Streamlit application URL."""
    return "http://localhost:8501"


@pytest.fixture(scope="session")
def browser_options(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Configure browser options for testing."""
    browser = getattr(request.config.option, "browser", "chrome").lower()
    headless = getattr(request.config.option, "headless", True)
    window_size = getattr(request.config.option, "window_size", "1920,1080")

    width, height = map(int, window_size.split(","))

    return {
        "browser": browser,
        "headless": headless,
        "window_size": (width, height),
        "implicit_wait": getattr(request.config.option, "implicit_wait", 10),
        "page_load_timeout": getattr(
            request.config.option, "page_load_timeout", 30
        ),
    }


@pytest.fixture(scope="function")
def driver(
    browser_options: dict[str, Any],
) -> Generator[webdriver.Remote, None, None]:
    """Create and configure WebDriver instance."""
    browser = browser_options["browser"]
    headless = browser_options["headless"]
    window_size = browser_options["window_size"]

    driver_instance = None

    try:
        if browser == "chrome":
            options = ChromeOptions()
            if headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-plugins")
            options.add_argument(
                f"--window-size={window_size[0]},{window_size[1]}"
            )
            options.add_argument("--remote-debugging-port=9222")

            driver_instance = webdriver.Chrome(options=options)

        elif browser == "firefox":
            options = FirefoxOptions()
            if headless:
                options.add_argument("--headless")
            options.add_argument(f"--width={window_size[0]}")
            options.add_argument(f"--height={window_size[1]}")

            driver_instance = webdriver.Firefox(options=options)

        else:
            raise ValueError(f"Unsupported browser: {browser}")

        # Configure timeouts
        driver_instance.implicitly_wait(browser_options["implicit_wait"])
        driver_instance.set_page_load_timeout(
            browser_options["page_load_timeout"]
        )

        # Set window size
        driver_instance.set_window_size(*window_size)

        yield driver_instance

    except WebDriverException as e:
        pytest.fail(f"Failed to create WebDriver for {browser}: {e}")

    finally:
        if driver_instance:
            try:
                driver_instance.quit()
            except Exception as e:
                print(f"Error closing WebDriver: {e}")


@pytest.fixture(scope="function")
def take_screenshot(
    request: pytest.FixtureRequest, driver: webdriver.Remote
) -> Callable[[str], Path | None]:
    """Fixture to take screenshots on test failure."""

    def _take_screenshot(name: str = "screenshot") -> Path | None:
        """Take a screenshot and save it to test results."""
        timestamp = int(time.time())
        test_name = request.node.name
        filename = f"{test_name}_{name}_{timestamp}.png"

        screenshot_dir = Path("test-results/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        screenshot_path = screenshot_dir / filename

        try:
            driver.save_screenshot(str(screenshot_path))
            print(f"Screenshot saved: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            print(f"Failed to take screenshot: {e}")
            return None

    # Take screenshot on test failure
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        _take_screenshot("failure")

    return _take_screenshot


class TestStreamlitBasic:
    """Basic E2E tests for Streamlit application."""

    @pytest.mark.e2e
    @pytest.mark.smoke
    @pytest.mark.docker
    def test_application_loads(
        self,
        driver: webdriver.Remote,
        streamlit_url: str,
        take_screenshot: Callable[[str], Path | None],
    ) -> None:
        """Test that Streamlit application loads successfully."""
        try:
            # Navigate to Streamlit application
            driver.get(streamlit_url)

            # Wait for the page to load
            wait = WebDriverWait(driver, 30)

            # Check if Streamlit is loaded by looking for the main container
            main_container = wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='stApp']")
                )
            )

            assert (
                main_container is not None
            ), "Streamlit main container not found"

            # Take a screenshot for verification
            take_screenshot("application_loaded")

            # Verify page title contains expected content
            page_title = driver.title
            assert page_title, "Page title is empty"

        except TimeoutException:
            take_screenshot("timeout_error")
            pytest.fail("Timeout waiting for Streamlit application to load")

    @pytest.mark.e2e
    @pytest.mark.smoke
    @pytest.mark.docker
    def test_sidebar_navigation(
        self,
        driver: webdriver.Remote,
        streamlit_url: str,
        take_screenshot: Callable[[str], Path | None],
    ) -> None:
        """Test that sidebar navigation is present and functional."""
        driver.get(streamlit_url)

        wait = WebDriverWait(driver, 30)

        try:
            # Wait for sidebar to be present
            sidebar = wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='stSidebar']")
                )
            )

            assert sidebar is not None, "Sidebar not found"

            # Check if navigation elements are present
            # This might need adjustment based on actual implementation
            navigation_elements = driver.find_elements(
                By.CSS_SELECTOR, "[data-testid='stSidebar'] .stSelectbox"
            )

            # If no selectbox, try radio buttons or other navigation
            if not navigation_elements:
                navigation_elements = driver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stSidebar'] .stRadio"
                )

            if not navigation_elements:
                navigation_elements = driver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stSidebar'] button"
                )

            take_screenshot("sidebar_navigation")

            # For now, just verify sidebar exists
            # More specific navigation tests can be added later
            assert sidebar.is_displayed(), "Sidebar is not visible"

        except TimeoutException:
            take_screenshot("sidebar_timeout")
            pytest.fail("Timeout waiting for sidebar to load")

    @pytest.mark.e2e
    @pytest.mark.critical
    @pytest.mark.docker
    def test_health_endpoint(
        self, driver: webdriver.Remote, streamlit_url: str
    ) -> None:
        """Test that Streamlit health endpoint is accessible."""
        health_url = f"{streamlit_url}/_stcore/health"

        try:
            driver.get(health_url)

            # Health endpoint should return some content
            page_source = driver.page_source
            assert page_source, "Health endpoint returned empty response"

            # The health endpoint typically returns JSON or simple text
            # We just verify it's accessible and returns content

        except Exception as e:
            pytest.fail(f"Health endpoint check failed: {e}")

    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.docker
    def test_page_load_performance(
        self, driver: webdriver.Remote, streamlit_url: str
    ) -> None:
        """Test that page loads within acceptable time limits."""
        start_time = time.time()

        try:
            driver.get(streamlit_url)

            # Wait for main content to load
            wait = WebDriverWait(driver, 30)
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='stApp']")
                )
            )

            load_time = time.time() - start_time

            # Assert page loads within 10 seconds (generous for container)
            assert (
                load_time < 10.0
            ), f"Page load time too slow: {load_time:.2f}s"

            print(f"Page load time: {load_time:.2f}s")

        except TimeoutException:
            load_time = time.time() - start_time
            pytest.fail(
                f"Page failed to load within timeout. "
                f"Load time: {load_time:.2f}s"
            )


# Pytest hooks for custom command line options
def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options for E2E testing."""
    parser.addoption(
        "--browser",
        action="store",
        default="chrome",
        help="Browser to use for testing (chrome, firefox)",
    )
    parser.addoption(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode",
    )
    parser.addoption(
        "--window-size",
        action="store",
        default="1920,1080",
        help="Browser window size (width,height)",
    )
    parser.addoption(
        "--implicit-wait",
        action="store",
        type=int,
        default=10,
        help="Implicit wait timeout in seconds",
    )
    parser.addoption(
        "--page-load-timeout",
        action="store",
        type=int,
        default=30,
        help="Page load timeout in seconds",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, None, None]:
    """Hook to capture test results for screenshot on failure."""
    outcome = yield
    rep = outcome.get_result() if outcome else None

    # Add report to the test item for access in fixtures
    if rep:
        setattr(item, f"rep_{rep.when}", rep)
