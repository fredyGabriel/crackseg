"""E2E Workflow and Regression Tests for Subtask 4.4.

This module provides simplified E2E tests specifically designed for subtask
4.4: End-to-End Workflow and Regression Testing. Tests are designed to work
without constructor issues and validate critical user workflows.
"""

import time

import pytest
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class WorkflowRegressionTests:
    """Simplified E2E tests for workflow validation and regression testing."""

    def setup_method(self) -> None:
        """Setup method for each test."""
        self.base_url = "http://localhost:8501"
        self.timeout = 30

    def create_chrome_driver(self) -> WebDriver:
        """Create Chrome WebDriver with appropriate options."""
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        return webdriver.Chrome(options=chrome_options)

    def create_firefox_driver(self) -> WebDriver:
        """Create Firefox WebDriver with appropriate options."""
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("--headless")
        firefox_options.add_argument("--width=1920")
        firefox_options.add_argument("--height=1080")
        return webdriver.Firefox(options=firefox_options)

    def wait_for_streamlit_ready(self, driver: WebDriver) -> None:
        """Wait for Streamlit application to be fully loaded."""
        try:
            # Wait for the Streamlit app to initialize
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)  # Additional wait for Streamlit initialization

            # Check if we can find typical Streamlit elements
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.readyState")
                == "complete"
            )
        except TimeoutException as e:
            raise AssertionError(
                "Streamlit application failed to load within timeout"
            ) from e

    def navigate_to_page(self, driver: WebDriver, page_name: str) -> bool:
        """Navigate to a specific page using sidebar navigation."""
        try:
            # Wait for sidebar to be present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='stSidebar']")
                )
            )

            # Find and click the page button
            sidebar_buttons = driver.find_elements(
                By.CSS_SELECTOR, "[data-testid='stSidebar'] button"
            )

            for button in sidebar_buttons:
                if page_name.lower() in button.text.lower():
                    button.click()
                    time.sleep(2)  # Wait for page transition
                    return True

            return False
        except Exception:
            return False

    def check_page_elements(
        self, driver: WebDriver, expected_elements: list[str]
    ) -> dict[str, bool]:
        """Check if expected elements are present on the page."""
        results = {}
        for element_text in expected_elements:
            try:
                # Try to find element by text content
                element_found = (
                    len(
                        driver.find_elements(
                            By.XPATH,
                            f"//*[contains(text(), '{element_text}')]",
                        )
                    )
                    > 0
                )
                results[element_text] = element_found
            except Exception:
                results[element_text] = False
        return results

    @pytest.mark.e2e
    def test_streamlit_application_loads(self) -> None:
        """Test that the Streamlit application loads successfully."""
        driver = self.create_chrome_driver()
        try:
            driver.get(self.base_url)
            self.wait_for_streamlit_ready(driver)

            # Verify page title is present (can be "app", "Streamlit", or
            # "CrackSeg")
            assert len(driver.title) > 0, (
                f"Page title is empty: '{driver.title}'"
            )

            # Verify basic page structure
            body = driver.find_element(By.TAG_NAME, "body")
            assert body is not None

            # Verify we can find some Streamlit-specific elements or content
            page_source = driver.page_source.lower()
            streamlit_indicators = [
                "streamlit",
                "stapp",
                "data-testid",
                "sidebar",
            ]
            has_streamlit_content = any(
                indicator in page_source for indicator in streamlit_indicators
            )
            assert has_streamlit_content, (
                "Page does not appear to be a Streamlit application"
            )

        finally:
            driver.quit()

    @pytest.mark.e2e
    def test_core_navigation_workflow(self) -> None:
        """Test core navigation between pages as validated in subtask 4.3."""
        driver = self.create_chrome_driver()
        try:
            driver.get(self.base_url)
            self.wait_for_streamlit_ready(driver)

            # Test navigation to all 6 pages as confirmed working in 4.3
            pages_to_test = [
                ("Home", ["Dashboard", "Dataset Statistics"]),
                ("Config", ["Configuration", "Load Configuration"]),
                ("Advanced Config", ["YAML Editor", "Templates"]),
                ("Architecture", ["Model Architecture", "Device"]),
                ("Train", ["Training Controls", "Start Training"]),
                ("Results", ["Results", "Gallery"]),
            ]

            navigation_results = {}

            for page_name, expected_elements in pages_to_test:
                navigation_success = self.navigate_to_page(driver, page_name)
                if navigation_success:
                    time.sleep(1)  # Wait for page load
                    element_check = self.check_page_elements(
                        driver, expected_elements
                    )
                    navigation_results[page_name] = {
                        "navigation": True,
                        "elements": element_check,
                    }
                else:
                    navigation_results[page_name] = {
                        "navigation": False,
                        "elements": {},
                    }

            # Assert that at least 5 out of 6 pages loaded successfully
            successful_navigations = sum(
                1
                for result in navigation_results.values()
                if result["navigation"]
            )
            assert successful_navigations >= 5, (
                f"Only {successful_navigations}/6 pages loaded successfully"
            )

        finally:
            driver.quit()

    @pytest.mark.e2e
    def test_regression_areas_from_4_3(self) -> None:
        """Test regression areas specifically validated in subtask 4.3."""
        driver = self.create_chrome_driver()
        try:
            driver.get(self.base_url)
            self.wait_for_streamlit_ready(driver)

            # Test Config page file browser (major functionality from 4.3)
            config_nav = self.navigate_to_page(driver, "Config")
            assert config_nav, "Failed to navigate to Config page"

            # Check for config file browser elements
            config_elements = self.check_page_elements(
                driver, ["YAML", "Load", "Configuration"]
            )
            assert any(config_elements.values()), (
                "Config page missing critical elements"
            )

            # Test Architecture page model information (validated in 4.3)
            arch_nav = self.navigate_to_page(driver, "Architecture")
            assert arch_nav, "Failed to navigate to Architecture page"

            # Check for architecture elements
            arch_elements = self.check_page_elements(
                driver, ["Model", "Architecture", "Device"]
            )
            assert any(arch_elements.values()), (
                "Architecture page missing critical elements"
            )

            # Test Train page controls (confirmed working in 4.3)
            train_nav = self.navigate_to_page(driver, "Train")
            assert train_nav, "Failed to navigate to Train page"

            # Check for training controls
            train_elements = self.check_page_elements(
                driver, ["Training", "Start", "Controls"]
            )
            assert any(train_elements.values()), (
                "Train page missing critical elements"
            )

        finally:
            driver.quit()

    @pytest.mark.e2e
    @pytest.mark.cross_browser
    def test_firefox_compatibility(self) -> None:
        """Test basic functionality in Firefox browser."""
        driver = self.create_firefox_driver()
        try:
            driver.get(self.base_url)
            self.wait_for_streamlit_ready(driver)

            # Basic compatibility test
            assert "Streamlit" in driver.title or len(driver.title) > 0

            # Test navigation to at least 2 core pages
            home_nav = self.navigate_to_page(driver, "Home")
            config_nav = self.navigate_to_page(driver, "Config")

            # At least one navigation should work
            assert home_nav or config_nav, (
                "Firefox failed basic navigation test"
            )

        finally:
            driver.quit()

    @pytest.mark.e2e
    def test_session_persistence_regression(self) -> None:
        """
        Test session state persistence across navigation (regression from 4.3).
        """
        driver = self.create_chrome_driver()
        try:
            driver.get(self.base_url)
            self.wait_for_streamlit_ready(driver)

            # Navigate through multiple pages to test session persistence
            page_sequence = ["Home", "Config", "Architecture", "Train"]

            for page in page_sequence:
                nav_success = self.navigate_to_page(driver, page)
                if nav_success:
                    time.sleep(1)
                    # Check that page loaded without errors
                    assert "error" not in driver.page_source.lower()
                    assert "exception" not in driver.page_source.lower()

        finally:
            driver.quit()

    @pytest.mark.e2e
    def test_performance_basic_load_times(self) -> None:
        """Test basic page load performance requirements."""
        driver = self.create_chrome_driver()
        try:
            start_time = time.time()
            driver.get(self.base_url)
            self.wait_for_streamlit_ready(driver)
            load_time = time.time() - start_time

            # Page should load within 30 seconds (generous for E2E)
            assert load_time < 30, (
                f"Page load time {load_time:.2f}s exceeds 30s limit"
            )

            # Test navigation performance
            nav_start = time.time()
            self.navigate_to_page(driver, "Config")
            nav_time = time.time() - nav_start

            # Navigation should be under 10 seconds
            assert nav_time < 10, (
                f"Navigation time {nav_time:.2f}s exceeds 10s limit"
            )

        finally:
            driver.quit()


# Standalone test functions for simpler execution
def test_workflow_basic_functionality() -> None:
    """Standalone test for basic workflow functionality."""
    test_instance = WorkflowRegressionTests()
    test_instance.setup_method()
    test_instance.test_streamlit_application_loads()


def test_workflow_navigation() -> None:
    """Standalone test for navigation workflow."""
    test_instance = WorkflowRegressionTests()
    test_instance.setup_method()
    test_instance.test_core_navigation_workflow()


def test_workflow_regression() -> None:
    """Standalone test for regression validation."""
    test_instance = WorkflowRegressionTests()
    test_instance.setup_method()
    test_instance.test_regression_areas_from_4_3()
