"""Integration test for hybrid WebDriver management system.

This test demonstrates the new driver management system integrating with
the existing CrackSeg E2E testing infrastructure, showing both Docker Grid
and WebDriverManager fallback capabilities.
"""

import pytest
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from tests.e2e.drivers import (
    DriverConfig,
    HybridDriverManager,
    create_driver,
    driver_session,
)


@pytest.fixture(scope="session")
def streamlit_url() -> str:
    """Get Streamlit application URL for testing."""
    return "http://localhost:8501"


class TestHybridDriverManagement:
    """Test suite for hybrid WebDriver management system."""

    @pytest.mark.e2e
    @pytest.mark.smoke
    def test_driver_manager_initialization(self) -> None:
        """Test that HybridDriverManager initializes correctly."""
        config = DriverConfig.from_environment(browser="chrome", headless=True)
        manager = HybridDriverManager(config)

        assert manager.config.browser == "chrome"
        assert manager.config.headless is True
        assert manager.get_active_driver_count() == 0

    @pytest.mark.e2e
    @pytest.mark.smoke
    def test_configuration_validation(self) -> None:
        """Test driver configuration validation and environment checking."""
        manager = HybridDriverManager()
        validation = manager.validate_configuration()

        assert isinstance(validation, dict)
        assert "configuration_valid" in validation
        assert "supported_browsers" in validation
        assert "chrome" in validation["supported_browsers"]
        assert "recommendations" in validation
        assert "warnings" in validation

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.parametrize("browser", ["chrome", "firefox"])
    def test_driver_creation_with_fallback(self, browser: str) -> None:
        """Test driver creation with automatic fallback between methods."""
        config = DriverConfig.from_environment(
            browser=browser, headless=True, page_load_timeout=15.0
        )

        manager = HybridDriverManager(config)

        # Test driver creation with automatic method selection
        driver = manager.create_driver(browser=browser, method="auto")

        try:
            assert driver is not None
            assert driver.session_id is not None
            assert manager.get_active_driver_count() == 1

            # Basic functionality test
            driver.get("https://httpbin.org/html")
            assert "Herman Melville" in driver.page_source

        finally:
            manager.cleanup_driver(driver)
            assert manager.get_active_driver_count() == 0

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_context_manager_usage(self) -> None:
        """Test driver management using context managers."""
        config = DriverConfig.from_environment(headless=True)

        with HybridDriverManager(config) as manager:
            assert manager.get_active_driver_count() == 0

            with manager.get_driver("chrome") as driver:
                assert isinstance(driver, WebDriver)
                assert driver.session_id is not None
                assert manager.get_active_driver_count() == 1

                # Test basic WebDriver functionality
                driver.get("https://httpbin.org/html")
                assert driver.title

            # Driver should be cleaned up automatically
            assert manager.get_active_driver_count() == 0

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_convenience_functions(self) -> None:
        """Test convenience functions for quick driver creation."""
        # Test create_driver convenience function
        driver = create_driver(browser="chrome", headless=True)

        try:
            assert driver is not None
            driver.get("https://httpbin.org/html")
            assert "Herman Melville" in driver.page_source
        finally:
            driver.quit()

        # Test driver_session convenience function
        with driver_session(browser="chrome", headless=True) as driver:
            assert driver is not None
            driver.get("https://httpbin.org/html")
            assert driver.title

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_streamlit_application_compatibility(
        self, streamlit_url: str
    ) -> None:
        """Test new driver system with actual Streamlit application.

        This test verifies that the new driver management system works
        correctly with the CrackSeg Streamlit application, maintaining
        compatibility with existing test infrastructure.
        """
        # Skip if Streamlit is not running (development environment)
        pytest.importorskip("requests")

        import requests

        try:
            response = requests.get(streamlit_url, timeout=5)
            if response.status_code != 200:
                pytest.skip("Streamlit application not available")
        except requests.RequestException:
            pytest.skip("Streamlit application not available")

        # Test with new driver management system
        config = DriverConfig.from_environment(
            browser="chrome",
            headless=True,
            implicit_wait=10.0,
            page_load_timeout=30.0,
        )

        with HybridDriverManager(config) as manager:
            with manager.get_driver() as driver:
                try:
                    # Navigate to Streamlit application
                    driver.get(streamlit_url)

                    # Wait for Streamlit to load
                    wait = WebDriverWait(driver, 30)

                    # Check for Streamlit main container
                    main_container = wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "[data-testid='stApp']")
                        )
                    )

                    assert main_container is not None
                    assert driver.title  # Page should have a title

                except TimeoutException:
                    pytest.fail(
                        "Timeout waiting for Streamlit application to load"
                    )

    @pytest.mark.e2e
    @pytest.mark.error_handling
    def test_error_handling_and_recovery(self) -> None:
        """Test error handling and recovery mechanisms."""
        config = DriverConfig.from_environment(
            browser="chrome",
            headless=True,
            selenium_hub_host="nonexistent-host",  # Force Docker failure
            selenium_hub_port=9999,
        )

        manager = HybridDriverManager(config)

        # Should fall back to WebDriverManager or local when Docker fails
        driver = manager.create_driver(method="auto", retry_count=2)

        try:
            assert driver is not None
            # Test that fallback driver works
            driver.get("https://httpbin.org/html")
            assert "Herman Melville" in driver.page_source
        finally:
            manager.cleanup_driver(driver)

    @pytest.mark.e2e
    @pytest.mark.performance
    def test_multiple_driver_management(self) -> None:
        """Test managing multiple drivers simultaneously."""
        config = DriverConfig.from_environment(headless=True)
        manager = HybridDriverManager(config)

        drivers = []
        try:
            # Create multiple drivers
            for browser in ["chrome", "firefox"]:
                if browser == "firefox":
                    # Skip Firefox if not available in environment
                    try:
                        driver = manager.create_driver(browser=browser)
                        drivers.append(driver)
                    except Exception:
                        pytest.skip(
                            f"{browser} not available in test environment"
                        )
                else:
                    driver = manager.create_driver(browser=browser)
                    drivers.append(driver)

            assert len(drivers) > 0
            assert manager.get_active_driver_count() == len(drivers)

            # Test that all drivers work independently
            for _i, driver in enumerate(drivers):
                driver.get("https://httpbin.org/html")
                assert "Herman Melville" in driver.page_source

        finally:
            # Cleanup all drivers
            manager.cleanup_all_drivers()
            assert manager.get_active_driver_count() == 0


@pytest.mark.e2e
@pytest.mark.integration
class TestDockerGridIntegration:
    """Test Docker Grid integration specifically."""

    def test_docker_grid_availability_check(self) -> None:
        """Test Docker Grid availability detection."""
        manager = HybridDriverManager()

        # This should not raise an exception
        is_available = manager.is_docker_grid_available()
        assert isinstance(is_available, bool)

    @pytest.mark.docker
    @pytest.mark.skipif_no_docker
    def test_docker_grid_driver_creation(self) -> None:
        """Test driver creation specifically using Docker Grid.

        This test requires Docker infrastructure to be running.
        It will be skipped if Docker Grid is not available.
        """
        manager = HybridDriverManager()

        if not manager.is_docker_grid_available():
            pytest.skip("Docker Grid not available")

        # Force Docker method
        driver = manager.create_driver(method="docker")

        try:
            assert driver is not None

            # Test Docker-specific capabilities
            caps = driver.capabilities
            assert "browserName" in caps
            assert caps["browserName"] in [
                "chrome",
                "firefox",
                "MicrosoftEdge",
            ]

            # Test basic functionality
            driver.get("https://httpbin.org/html")
            assert "Herman Melville" in driver.page_source

        finally:
            manager.cleanup_driver(driver)


# Pytest configuration for test parametrization
def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options for driver testing."""
    parser.addoption(
        "--test-browser",
        action="store",
        default="chrome",
        help="Browser to use for testing (chrome, firefox, edge)",
    )
    parser.addoption(
        "--test-method",
        action="store",
        default="auto",
        help="Driver creation method (auto, docker, local, webdriver-manager)",
    )


@pytest.fixture(scope="session")
def test_browser(request: pytest.FixtureRequest) -> str:
    """Get browser from command line option."""
    return request.config.getoption("--test-browser")


@pytest.fixture(scope="session")
def test_method(request: pytest.FixtureRequest) -> str:
    """Get driver method from command line option."""
    return request.config.getoption("--test-method")
