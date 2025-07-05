"""Cross-browser testing base class for comprehensive browser compatibility
testing.

This module provides a framework for executing E2E tests across multiple
browsers and devices, supporting both sequential and parallel execution modes.
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

from selenium.webdriver.remote.webdriver import WebDriver

from ..base_test import BaseE2ETest
from ..drivers.config import BrowserType
from .browser_capabilities import (
    BrowserCapabilities,
    ChromeCapabilities,
    EdgeCapabilities,
    FirefoxCapabilities,
    MobileBrowserCapabilities,
)
from .browser_config_manager import BrowserConfigManager, BrowserMatrix

logger = logging.getLogger(__name__)


class CrossBrowserTest(BaseE2ETest, ABC):
    """Base class for cross-browser E2E testing with Streamlit applications.

    This class extends BaseE2ETest to support testing across multiple browsers
    and devices using the BrowserConfigManager infrastructure.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize cross-browser test with configuration manager."""
        super().__init__(*args, **kwargs)
        self.config_manager = BrowserConfigManager()
        self.test_results: dict[str, dict[str, Any]] = {}

    def setup_cross_browser_matrix(
        self, browsers: list[str] | None = None, include_mobile: bool = False
    ) -> BrowserMatrix:
        """Setup browser testing matrix for cross-browser execution.

        Args:
            browsers: List of browser names to test (defaults to
                ["chrome", "firefox", "edge"])
            include_mobile: Whether to include mobile device testing

        Returns:
            Configured BrowserMatrix instance
        """
        # Convert string browser names to BrowserType literals
        browser_types: list[BrowserType] = []
        default_browsers = ["chrome", "firefox", "edge"]
        target_browsers = browsers or default_browsers

        for browser_name in target_browsers:
            # Cast string to BrowserType literal for type safety
            browser_type = cast(BrowserType, browser_name)
            browser_types.append(browser_type)

        return BrowserMatrix(
            browsers=browser_types,
            include_mobile=include_mobile,
            headless_modes=[True, False],
        )

    def set_browser_capabilities(
        self, browser: str, capabilities: BrowserCapabilities
    ) -> None:
        """Set custom capabilities for a specific browser.

        Args:
            browser: Browser name
            capabilities: Browser-specific capabilities configuration
        """
        # Cast string to BrowserType for manager compatibility
        browser_type = cast(BrowserType, browser)
        self.config_manager.set_browser_capabilities(
            browser_type, capabilities
        )

    def get_default_capabilities(self, browser: str) -> BrowserCapabilities:
        """Get default capabilities for a browser type.

        Args:
            browser: Browser name

        Returns:
            Default capabilities configuration for the browser
        """
        browser_type = cast(BrowserType, browser)
        return self.config_manager.get_browser_capabilities(browser_type)

    @abstractmethod
    def run_browser_specific_test(
        self, driver: WebDriver, browser: str, test_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run test logic specific to each browser.

        This method must be implemented by concrete test classes to define
        the actual test steps to be executed across all browsers.

        Args:
            driver: WebDriver instance for the current browser
            browser: Browser name being tested
            test_data: Test data for the current test execution

        Returns:
            Test results dictionary with metrics and status
        """
        pass

    def execute_cross_browser_test(
        self,
        browser_matrix: BrowserMatrix | None = None,
        parallel: bool = False,
        max_workers: int = 3,
    ) -> dict[str, dict[str, Any]]:
        """Execute test across multiple browsers.

        Args:
            browser_matrix: Browser testing matrix (created if None)
            parallel: Whether to run browsers in parallel
            max_workers: Maximum parallel workers for concurrent execution

        Returns:
            Dictionary of test results keyed by browser configuration
        """
        if browser_matrix is None:
            browser_matrix = self.setup_cross_browser_matrix()

        test_data = self.get_test_data()

        if parallel:
            return self._execute_parallel_tests(
                browser_matrix, test_data, max_workers
            )
        else:
            return self._execute_sequential_tests(browser_matrix, test_data)

    def _execute_sequential_tests(
        self, browser_matrix: BrowserMatrix, test_data: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Execute tests sequentially across browsers.

        Args:
            browser_matrix: Browser testing configuration matrix
            test_data: Test data for execution

        Returns:
            Dictionary of test results
        """
        results: dict[str, dict[str, Any]] = {}

        for config in browser_matrix.generate_configurations():
            config_key = (
                f"{config['browser']}_{config.get('window_width', 1920)}x"
                f"{config.get('window_height', 1080)}"
            )

            self.log_test_step(
                f"Starting test for {config_key}",
                f"Browser: {config['browser']}, Window: "
                f"{config.get('window_width', 1920)}x"
                f"{config.get('window_height', 1080)}",
            )

            try:
                # Create driver for current configuration using the manager
                driver = self.config_manager._create_driver_from_config(
                    config_key, config
                )

                try:
                    # Navigate to application and verify load
                    self.navigate_and_verify(
                        driver, "http://localhost:8501", timeout=30.0
                    )

                    # Wait for Streamlit to be ready
                    self.assert_streamlit_loaded(driver, timeout=30.0)

                    # Execute browser-specific test logic
                    test_result = self.run_browser_specific_test(
                        driver, config["browser"], test_data
                    )

                    results[config_key] = {
                        **test_result,
                        "status": "passed",
                        "browser": config["browser"],
                    }

                    self.log_test_step(
                        f"Test completed for {config_key}", "Status: PASSED"
                    )

                finally:
                    driver.quit()

            except Exception as e:
                results[config_key] = {
                    "status": "failed",
                    "error": str(e),
                    "browser": config["browser"],
                }
                self.log_test_step(
                    f"Test failed for {config_key}", f"Error: {e}"
                )

        return results

    def _execute_parallel_tests(
        self,
        browser_matrix: BrowserMatrix,
        test_data: dict[str, Any],
        max_workers: int,
    ) -> dict[str, dict[str, Any]]:
        """Execute tests in parallel across browsers.

        Args:
            browser_matrix: Browser testing configuration matrix
            test_data: Test data for execution
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary of test results
        """
        results: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test tasks
            future_to_config = {}

            for config in browser_matrix.generate_configurations():
                future = executor.submit(
                    self._run_single_browser_test, config, test_data
                )
                future_to_config[future] = config

            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                config_key = (
                    f"{config['browser']}_{config.get('window_width', 1920)}x"
                    f"{config.get('window_height', 1080)}"
                )

                try:
                    result = future.result()
                    results[config_key] = result
                    self.log_test_step(
                        f"Parallel test completed for {config_key}",
                        f"Status: {result.get('status', 'unknown')}",
                    )

                except Exception as e:
                    results[config_key] = {
                        "status": "failed",
                        "error": str(e),
                        "browser": config["browser"],
                    }
                    self.log_test_step(
                        f"Parallel test failed for {config_key}", f"Error: {e}"
                    )

        return results

    def _run_single_browser_test(
        self, config: dict[str, Any], test_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run test for a single browser configuration.

        Args:
            config: Browser configuration dictionary
            test_data: Test data for execution

        Returns:
            Test result dictionary
        """
        driver = None
        config_key = (
            f"{config['browser']}_{config.get('window_width', 1920)}x"
            f"{config.get('window_height', 1080)}"
        )

        try:
            # Create driver for current configuration
            driver = self.config_manager._create_driver_from_config(
                config_key, config
            )

            # Navigate to application and verify load
            self.navigate_and_verify(
                driver, "http://localhost:8501", timeout=30.0
            )

            # Wait for Streamlit to be ready
            self.assert_streamlit_loaded(driver, timeout=30.0)

            # Execute browser-specific test logic
            test_result = self.run_browser_specific_test(
                driver, config["browser"], test_data
            )

            return {
                **test_result,
                "status": "passed",
                "browser": config["browser"],
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "browser": config["browser"],
            }
        finally:
            if driver:
                driver.quit()


def create_performance_optimized_capabilities(
    browser: str,
) -> BrowserCapabilities:
    """Create performance-optimized capabilities for CI/CD environments.

    Args:
        browser: Browser name

    Returns:
        Performance-optimized browser capabilities
    """
    browser_type = cast(BrowserType, browser)

    if browser_type == "chrome":
        return ChromeCapabilities(
            headless_mode="new",
            disable_gpu=True,
            no_sandbox=True,
            disable_dev_shm_usage=True,
            disable_extensions=True,
            additional_arguments=[
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-features=TranslateUI,VizDisplayCompositor",
            ],
        )
    elif browser_type == "firefox":
        return FirefoxCapabilities(
            headless_mode="new",
            disable_gpu=True,
            preferences={
                "browser.cache.disk.enable": False,
                "browser.cache.memory.enable": False,
                "browser.cache.offline.enable": False,
                "network.cookie.cookieBehavior": 0,
            },
        )
    elif browser_type == "edge":
        return EdgeCapabilities(
            headless_mode="new",
            disable_gpu=True,
            no_sandbox=True,
            edge_performance_mode=True,
            additional_arguments=[
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
            ],
        )
    else:
        # Default fallback
        return ChromeCapabilities(headless_mode="new", disable_gpu=True)


def create_mobile_testing_capabilities(
    device: str = "iPhone 12",
) -> MobileBrowserCapabilities:
    """Create mobile device testing capabilities.

    Args:
        device: Mobile device name for emulation

    Returns:
        Mobile browser capabilities configuration
    """
    return MobileBrowserCapabilities(
        device_name=device,
        headless_mode="new",
        enable_touch_events=True,
        force_device_scale_factor=True,
        additional_arguments=[
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
        ],
    )
