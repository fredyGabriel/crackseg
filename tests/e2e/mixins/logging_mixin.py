"""Logging mixin for E2E testing.

This module provides structured logging capabilities for E2E tests,
including test step logging, assertion logging, and performance metrics.
"""

import logging
from typing import Protocol

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver


class HasLogging(Protocol):
    """Protocol for classes that have logging capabilities."""

    _test_logger: logging.Logger

    def log_test_step(self, step: str, details: str | None = None) -> None:
        """Log a test step with optional details."""
        ...

    def log_assertion(
        self, assertion: str, passed: bool, details: str | None = None
    ) -> None:
        """Log assertion results."""
        ...


class LoggingMixin:
    """Mixin providing structured logging capabilities for E2E tests."""

    _test_logger: logging.Logger

    def setup_logging(self) -> None:
        """Initialize logging for the test class."""
        self._test_logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def log_test_step(self, step: str, details: str | None = None) -> None:
        """Log a test step with optional details.

        Args:
            step: Description of the test step
            details: Additional details about the step
        """
        message = f"TEST STEP: {step}"
        if details:
            message += f" - {details}"
        self._test_logger.info(message)

    def log_assertion(
        self, assertion: str, passed: bool, details: str | None = None
    ) -> None:
        """Log assertion results.

        Args:
            assertion: Description of the assertion
            passed: Whether assertion passed
            details: Additional details about the assertion
        """
        status = "PASSED" if passed else "FAILED"
        message = f"ASSERTION {status}: {assertion}"
        if details:
            message += f" - {details}"

        if passed:
            self._test_logger.info(message)
        else:
            self._test_logger.error(message)

    def log_performance_metric(
        self, metric_name: str, value: float, unit: str = "seconds"
    ) -> None:
        """Log performance metrics.

        Args:
            metric_name: Name of the performance metric
            value: Measured value
            unit: Unit of measurement
        """
        self._test_logger.info(
            f"PERFORMANCE: {metric_name} = {value:.3f} {unit}"
        )

    def log_browser_info(self, driver: WebDriver) -> None:
        """Log browser and environment information.

        Args:
            driver: WebDriver instance
        """
        try:
            user_agent = driver.execute_script("return navigator.userAgent")
            window_size = driver.get_window_size()

            self._test_logger.info(f"BROWSER INFO: {user_agent}")
            self._test_logger.info(
                f"WINDOW SIZE: {window_size['width']}x{window_size['height']}"
            )
        except WebDriverException as e:
            self._test_logger.warning(f"Could not retrieve browser info: {e}")
