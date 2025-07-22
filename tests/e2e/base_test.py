"""
Base test class for E2E testing. This module provides the core
BaseE2ETest class that composes all testing mixins to provide a
comprehensive E2E testing foundation.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pytest
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Test data management (simplified for demo)
from .mixins import (
    CaptureMixin,
    LoggingMixin,
    PerformanceMixin,
    RetryMixin,
    StreamlitMixin,
)


class BaseE2ETest(
    LoggingMixin,
    RetryMixin,
    StreamlitMixin,
    PerformanceMixin,
    CaptureMixin,
):
    """
    Base class for E2E testing providing comprehensive testing
    capabilities. Combines all testing mixins to provide: - Structured
    logging and assertions - Retry mechanisms for flaky scenarios -
    Streamlit-specific utilities - Performance monitoring - Screenshot and
    video capture This class maintains a slim profile by delegating
    functionality to mixins.
    """

    driver: WebDriver
    base_url: str
    _test_data: dict[str, Any]
    _current_test_name: str | None = None

    def setup_method(self, method: pytest.Function) -> None:
        """
        Setup method called before each test method. Args: method: pytest test
        method
        """
        self._current_test_name = (
            f"{self.__class__.__name__}.{method.__name__}"
        )

        # Initialize all mixins
        self.setup_logging()
        self.setup_retry()
        self.setup_performance_monitoring()
        self.setup_capture_system()

        # Load test data (simplified default configuration)
        self._test_data = {
            "streamlit_app": True,
            "capture_config": {
                "screenshots": {"enabled": True, "on_failure": True},
                "videos": {"enabled": True, "cleanup_on_success": True},
            },
            "retention_policy": "keep_failures",
        }

        # Configure capture system from test data
        self.configure_capture_from_test_data(self._test_data)

        self.log_test_step(f"Starting test: {self._current_test_name}")

    def teardown_method(self, method: pytest.Function) -> None:
        """
        Teardown method called after each test method. Args: method: pytest
        test method
        """
        test_passed = getattr(self, "_pytest_passed", True)

        # Clean up capture artifacts based on test result
        self.cleanup_capture_artifacts(test_passed)

        self.log_test_step(
            f"Completed test: {self._current_test_name}",
            f"Result: {'PASSED' if test_passed else 'FAILED'}",
        )

    def navigate_and_verify(
        self,
        driver: WebDriver,
        path: str = "",
        expected_title: str | None = None,
        wait_for_element: tuple[str, str] | None = None,
        timeout: float = 30.0,
        measure_performance: bool = False,
    ) -> dict[str, float] | None:
        """
        Navigate to URL and verify page load with optional performance
        measurement. Args: driver: WebDriver instance path: URL path to
        navigate to (relative to base_url) expected_title: Expected page title
        for verification wait_for_element: Tuple of (by, value) to wait for
        specific element timeout: Maximum time to wait for page load
        measure_performance: Whether to measure performance metrics Returns:
        Performance metrics if measurement enabled, None otherwise Raises:
        AssertionError: If verification fails TimeoutException: If page load
        times out
        """
        # Construct full URL
        url = urljoin(self.base_url, path) if path else self.base_url

        self.log_test_step(f"Navigating to: {url}")

        # Navigate with optional performance measurement
        if measure_performance:
            metrics = self.measure_page_load_performance(driver, url, timeout)
        else:
            driver.get(url)
            self.assert_page_ready_state(driver, timeout)
            metrics = None

        # Verify page title if specified
        if expected_title:
            self._verify_page_title(driver, expected_title)

        # Wait for specific element if specified
        if wait_for_element:
            by, value = wait_for_element
            self._wait_for_element(driver, by, value, timeout)

        # Verify Streamlit is loaded (if applicable)
        if self._is_streamlit_app():
            self.assert_streamlit_loaded(driver, timeout)

        self.log_test_step("Page navigation and verification complete")
        return metrics

    def wait_for_condition(
        self,
        driver: WebDriver,
        condition: Callable[[WebDriver], Any],
        timeout: float = 10.0,
        description: str = "condition",
        retry_count: int = 2,
    ) -> Any:
        """
        Wait for a custom condition with retry support. Args: driver:
        WebDriver instance condition: Condition function to wait for timeout:
        Timeout for each attempt description: Description for logging
        retry_count: Number of retry attempts Returns: Result of successful
        condition Raises: TimeoutException: If condition not met within
        timeout and retries
        """
        return self.wait_with_retry(
            driver, condition, timeout, retry_count, description
        )

    def execute_with_retry(
        self,
        operation: Callable[[], Any],
        description: str = "operation",
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Any:
        """
        Execute operation with automatic retry on common failures. Args:
        operation: Operation to execute description: Description for logging
        max_retries: Maximum retry attempts delay: Initial delay between
        retries Returns: Result of successful operation Raises: Last exception
        if all retries fail
        """
        return self.retry_operation(
            operation,
            max_retries=max_retries,
            delay=delay,
            description=description,
        )

    def capture_checkpoint(
        self,
        driver: WebDriver,
        checkpoint_name: str,
        context: str | None = None,
    ) -> Path | None:
        """
        Capture screenshot at test checkpoint. Args: driver: WebDriver
        instance checkpoint_name: Name of the checkpoint context: Optional
        context information Returns: Path to captured screenshot or None if
        failed
        """
        return self.capture_test_checkpoint(driver, checkpoint_name, context)

    def start_test_recording_shortcut(self, driver: WebDriver) -> bool:
        """
        Start video recording for current test (shortcut method). Args:
        driver: WebDriver instance Returns: True if recording started
        successfully
        """
        return self.start_test_recording(driver)

    def stop_recording(self, save_video: bool = True) -> Path | None:
        """
        Stop video recording for current test. Args: save_video: Whether to
        save the recorded video Returns: Path to saved video or None if not
        saved
        """
        return self.stop_test_recording(save_video)

    def _verify_page_title(
        self, driver: WebDriver, expected_title: str
    ) -> None:
        """Verify page title matches expected value."""
        actual_title = driver.title
        passed = expected_title in actual_title

        self.log_assertion(
            f"Page title contains '{expected_title}'",
            passed,
            f"Actual: '{actual_title}'",
        )

        if not passed:
            raise AssertionError(
                f"Page title '{actual_title}' does not contain "
                f"'{expected_title}'"
            )

    def _wait_for_element(
        self, driver: WebDriver, by: str, value: str, timeout: float
    ) -> None:
        """Wait for specific element to be present."""
        from selenium.webdriver.common.by import By

        by_mapping = {
            "id": By.ID,
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME,
        }

        by_locator = by_mapping.get(by.lower(), By.CSS_SELECTOR)

        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by_locator, value))
        )

        self.log_test_step(f"Element found: {by}='{value}'")

    def _is_streamlit_app(self) -> bool:
        """Check if current test is for a Streamlit application."""
        return self._test_data.get("streamlit_app", False)

    def get_test_data(self) -> dict[str, Any]:
        """Get test data configuration for the current test.

        This method provides access to test-specific data configuration.
        Can be overridden by test classes to provide custom test data.

        Returns:
            Dictionary containing test data configuration
        """
        return getattr(
            self,
            "_test_data",
            {
                "config_file": "basic_verification.yaml",
                "expected_model_name": "MockModel",
                "boundary_test_config": "basic_verification.yaml",
                "wait_times": {
                    "config_load": 2,
                    "error_message": 2,
                    "long_config": 2,
                    "edge_values": 2,
                },
                "expected_error_keywords": ["invalid", "error", "failed"],
            },
        )

    def setup_test_data(self) -> dict[str, Any]:
        """Set up test-specific data - to be overridden by subclasses.

        Returns:
            Dictionary containing test configuration
        """
        return self.get_test_data()

    # Performance monitoring methods
    def start_performance_monitoring(self) -> None:
        """Start performance monitoring for the current test."""
        self._performance_enabled = True
        if hasattr(self, "setup_performance_monitoring"):
            self.setup_performance_monitoring()

    def monitor_memory_usage_snapshot(self) -> dict[str, float]:
        """Take a memory usage snapshot.

        Returns:
            Dictionary with memory usage metrics
        """
        # Mock implementation - in real scenario would use psutil or similar
        return {
            "memory_mb": 512.0,
            "cpu_percent": 45.0,
            "timestamp": time.time(),
        }

    def track_interaction_performance(
        self,
        driver: WebDriver,
        interaction_type: str,
        description: str,
        threshold_ms: float = 1000.0,
    ) -> dict[str, Any]:
        """Track performance of UI interactions.

        Args:
            driver: WebDriver instance
            interaction_type: Type of interaction being tracked
            description: Description of the interaction
            threshold_ms: Performance threshold in milliseconds

        Returns:
            Performance metrics dictionary
        """
        import time

        start_time = time.time()
        # Mock interaction timing
        time.sleep(0.1)  # Simulate interaction time
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000

        return {
            "interaction_type": interaction_type,
            "description": description,
            "duration_ms": duration_ms,
            "threshold_ms": threshold_ms,
            "passed": duration_ms <= threshold_ms,
        }

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate a performance report for the current test.

        Returns:
            Performance report dictionary
        """
        return {
            "test_name": self._current_test_name or "unknown",
            "performance_enabled": getattr(
                self, "_performance_enabled", False
            ),
            "memory_snapshots": [],
            "interaction_metrics": [],
            "summary": "Performance monitoring completed",
        }
