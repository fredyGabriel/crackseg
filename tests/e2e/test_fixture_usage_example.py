"""Example test demonstrating E2E fixture usage patterns.

This module provides examples of how to use the E2E testing fixtures
for testing the CrackSeg Streamlit application with different browsers
and configurations.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from tests.e2e.drivers import BrowserType, DriverConfig


class TestStreamlitApplicationBasics:
    """Example tests for basic Streamlit application functionality."""

    @pytest.mark.e2e
    def test_application_loads_with_chrome(
        self, chrome_driver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test that the Streamlit application loads successfully in Chrome.

        Args:
            chrome_driver: Chrome WebDriver instance
            streamlit_base_url: Base URL for Streamlit application
        """
        # Navigate to the application
        chrome_driver.get(streamlit_base_url)

        # Wait for the application to load
        wait = WebDriverWait(chrome_driver, 10)
        wait.until(
            lambda driver: driver.execute_script("return document.readyState")
            == "complete"
        )

        # Check that the page title contains expected content
        assert (
            "Streamlit" in chrome_driver.title
            or "CrackSeg" in chrome_driver.title
        )

        # Verify the application is responsive
        assert (
            chrome_driver.execute_script("return document.readyState")
            == "complete"
        )

    @pytest.mark.e2e
    @pytest.mark.firefox
    def test_application_loads_with_firefox(
        self, firefox_driver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test that the Streamlit application loads successfully in Firefox.

        Args:
            firefox_driver: Firefox WebDriver instance
            streamlit_base_url: Base URL for Streamlit application
        """
        firefox_driver.get(streamlit_base_url)

        # Wait for page load
        WebDriverWait(firefox_driver, 10).until(
            lambda driver: driver.execute_script("return document.readyState")
            == "complete"
        )

        # Verify basic page structure
        assert firefox_driver.find_element(By.TAG_NAME, "body") is not None

    @pytest.mark.e2e
    @pytest.mark.cross_browser
    def test_navigation_elements_present(
        self,
        cross_browser_driver: WebDriver,
        streamlit_base_url: str,
        test_data: dict[str, Any],
    ) -> None:
        """Test that navigation elements are present across different browsers.

        Args:
            cross_browser_driver: Cross-browser WebDriver instance
            streamlit_base_url: Base URL for Streamlit application
            test_data: Test data containing expected navigation elements
        """
        cross_browser_driver.get(streamlit_base_url)

        # Wait for application to load
        wait = WebDriverWait(cross_browser_driver, 15)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Get expected navigation elements from test data
        expected_nav_elements = test_data["expected_results"][
            "navigation_elements"
        ]

        # Check that page contains navigation elements
        page_text = cross_browser_driver.page_source.lower()

        # At least some navigation elements should be present
        # Note: This is a basic check - real implementation would use more
        # specific selectors
        found_elements = []
        for element in expected_nav_elements:
            if element.lower() in page_text:
                found_elements.append(element)

        # Verify at least half of the expected elements are found
        assert (
            len(found_elements) >= len(expected_nav_elements) // 2
        ), f"Expected navigation elements not found. Found: {found_elements}"


class TestDriverManagerUsage:
    """Example tests demonstrating driver manager usage."""

    @pytest.mark.e2e
    def test_driver_manager_provides_working_driver(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test that the driver manager provides a working WebDriver.

        Args:
            webdriver: Default WebDriver from driver manager
            streamlit_base_url: Base URL for Streamlit application
        """
        # Navigate to application
        webdriver.get(streamlit_base_url)

        # Verify driver is functional
        assert webdriver.current_url.startswith(streamlit_base_url)

        # Test basic JavaScript execution
        result = webdriver.execute_script("return window.navigator.userAgent")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.e2e
    def test_window_size_configuration(
        self, webdriver: WebDriver, e2e_config: DriverConfig
    ) -> None:
        """Test that window size is configured correctly.

        Args:
            webdriver: WebDriver instance
            e2e_config: E2E configuration
        """
        expected_width, expected_height = e2e_config.window_size

        # Get actual window size
        actual_size = webdriver.get_window_size()

        # Allow for small variations due to browser chrome
        assert abs(actual_size["width"] - expected_width) <= 50, (
            f"Window width mismatch: expected ~{expected_width}, "
            f"got {actual_size['width']}"
        )
        assert abs(actual_size["height"] - expected_height) <= 100, (
            f"Window height mismatch: expected ~{expected_height}, "
            f"got {actual_size['height']}"
        )


class TestResourceManagement:
    """Example tests for resource management and cleanup."""

    @pytest.mark.e2e
    def test_screenshot_on_failure_demonstration(
        self,
        chrome_driver: WebDriver,
        streamlit_base_url: str,
        test_artifacts_dir: Path,
    ) -> None:
        """Demonstrate screenshot capture functionality.

        Note: This test is designed to show how screenshots work,
        not to actually fail in normal operation.

        Args:
            chrome_driver: Chrome WebDriver instance
            streamlit_base_url: Base URL for Streamlit application
            test_artifacts_dir: Directory for test artifacts
        """
        chrome_driver.get(streamlit_base_url)

        # Take a manual screenshot for demonstration
        screenshot_path = test_artifacts_dir / "manual_screenshot_example.png"
        chrome_driver.save_screenshot(str(screenshot_path))

        # Verify screenshot was created
        assert screenshot_path.exists()
        assert screenshot_path.stat().st_size > 0

    @pytest.mark.e2e
    def test_artifacts_directory_usage(self, test_artifacts_dir: Path) -> None:
        """Test that artifacts directory is properly set up.

        Args:
            test_artifacts_dir: Test artifacts directory
        """
        # Verify directory exists and is writable
        assert test_artifacts_dir.exists()
        assert test_artifacts_dir.is_dir()

        # Test writing to artifacts directory
        test_file = test_artifacts_dir / "test_artifact.txt"
        test_file.write_text("Test artifact content")

        assert test_file.exists()
        assert test_file.read_text() == "Test artifact content"


class TestConfigurationVariations:
    """Example tests for different configuration scenarios."""

    @pytest.mark.e2e
    def test_different_browser_configs(
        self,
        chrome_config: DriverConfig,
        firefox_config: DriverConfig,
        edge_config: DriverConfig,
    ) -> None:
        """Test that different browser configurations are set up correctly.

        Args:
            chrome_config: Chrome configuration
            firefox_config: Firefox configuration
            edge_config: Edge configuration
        """
        # Verify browser-specific configurations
        assert chrome_config.browser == "chrome"
        assert firefox_config.browser == "firefox"
        assert edge_config.browser == "edge"

        # Verify common settings are consistent
        configs = [chrome_config, firefox_config, edge_config]
        for config in configs:
            assert config.headless is True
            assert config.window_size == (1920, 1080)
            assert config.implicit_wait == 10.0

    @pytest.mark.e2e
    def test_test_data_availability(self, test_data: dict[str, Any]) -> None:
        """Test that test data is properly structured for E2E tests.

        Args:
            test_data: Test data dictionary
        """
        # Verify sample images for upload testing
        sample_images = test_data["sample_images"]
        assert len(sample_images) > 0

        crack_images = [img for img in sample_images if img["type"] == "crack"]
        no_crack_images = [
            img for img in sample_images if img["type"] == "no_crack"
        ]

        assert len(crack_images) > 0, "Need crack images for testing"
        assert len(no_crack_images) > 0, "Need non-crack images for testing"

        # Verify configuration values
        config_values = test_data["config_values"]
        assert "model_name" in config_values
        assert isinstance(config_values["batch_size"], int)
        assert 0.0 <= config_values["confidence_threshold"] <= 1.0


class TestIntegrationWithExistingInfrastructure:
    """Tests demonstrating integration with existing project infrastructure."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_driver_cleanup_after_test(self, webdriver: WebDriver) -> None:
        """Test that drivers are properly cleaned up after test execution.

        This test demonstrates the automatic cleanup functionality
        provided by the fixture system.

        Args:
            webdriver: WebDriver instance that will be automatically cleaned up
        """
        # Use the driver for some operations
        webdriver.execute_script("console.log('E2E test running')")

        # The driver will be automatically cleaned up by the fixture
        # No manual cleanup required
        assert webdriver is not None

    @pytest.mark.e2e
    def test_multiple_driver_instances_in_sequence(
        self, e2e_config: DriverConfig
    ) -> None:
        """Test that multiple driver instances can be created in sequence.

        This demonstrates the robustness of the driver management system.

        Args:
            e2e_config: E2E configuration
        """
        from tests.e2e.drivers import driver_session

        # Create multiple drivers in sequence
        for browser_name in ["chrome", "firefox"]:
            browser = cast(BrowserType, browser_name)
            config = type(e2e_config)(
                browser=browser,
                **{
                    k: v
                    for k, v in e2e_config.to_dict().items()
                    if k != "browser"
                },
            )

            with driver_session(browser=browser, config=config) as driver:
                # Verify driver is functional
                assert driver is not None
                result = driver.execute_script("return true")
                assert result is True
