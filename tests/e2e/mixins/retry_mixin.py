"""Retry mixin for E2E testing.

This module provides retry mechanisms for handling flaky test scenarios
with configurable retry policies and exponential backoff.
"""

import time
from collections.abc import Callable
from typing import Any, TypeVar

from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait

# Type variable for flexible return types
T = TypeVar("T")


class RetryMixin:
    """Mixin providing retry mechanisms for flaky test scenarios."""

    _default_retry_count: int
    _default_retry_delay: float

    def setup_retry(self) -> None:
        """Initialize retry mixin."""
        self._default_retry_count = 3
        self._default_retry_delay = 1.0

    def retry_operation(
        self,
        operation: Callable[[], T],
        max_retries: int | None = None,
        delay: float | None = None,
        exceptions: tuple[type[Exception], ...] = (WebDriverException,),
        description: str = "operation",
    ) -> T:
        """Retry an operation with configurable parameters.

        Args:
            operation: Function to retry
            max_retries: Maximum number of retry attempts
            delay: Delay between attempts in seconds
            exceptions: Exception types to catch and retry
            description: Description for logging

        Returns:
            Result of successful operation

        Raises:
            Last exception if all retries fail
        """
        max_retries = max_retries or self._default_retry_count
        delay = delay or self._default_retry_delay

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if hasattr(self, "log_test_step"):
                    if attempt > 0:
                        self.log_test_step(  # type: ignore[attr-defined]
                            f"Retrying {description}",
                            f"Attempt {attempt + 1}/{max_retries + 1}",
                        )
                    else:
                        self.log_test_step(  # type: ignore[attr-defined]
                            f"Executing {description}"
                        )

                result = operation()

                if hasattr(self, "log_test_step") and attempt > 0:
                    self.log_test_step(  # type: ignore[attr-defined]
                        f"{description} succeeded",
                        f"After {attempt + 1} attempts",
                    )

                return result

            except exceptions as e:
                last_exception = e
                if hasattr(self, "_test_logger"):
                    self._test_logger.warning(  # type: ignore[attr-defined]
                        f"{description} failed (attempt {attempt + 1}): {e}"
                    )

                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 1.5  # Exponential backoff
                else:
                    if hasattr(self, "_test_logger"):
                        self._test_logger.error(  # type: ignore[attr-defined]
                            f"{description} failed after "
                            f"{max_retries + 1} attempts"
                        )

        # All retries failed
        if last_exception:
            raise last_exception

        # Should never reach here, but just in case
        raise RuntimeError(f"Operation {description} failed without exception")

    def wait_with_retry(
        self,
        driver: WebDriver,
        condition: Callable[[WebDriver], Any],
        timeout: float = 10.0,
        retry_count: int = 2,
        description: str = "condition",
    ) -> Any:
        """Wait for condition with retry mechanism.

        Args:
            driver: WebDriver instance
            condition: Condition function to wait for
            timeout: Timeout for each wait attempt
            retry_count: Number of retry attempts
            description: Description for logging

        Returns:
            Result of successful condition

        Raises:
            TimeoutException: If all retries fail
        """

        def wait_operation() -> Any:
            return WebDriverWait(driver, timeout).until(condition)

        return self.retry_operation(
            wait_operation,
            max_retries=retry_count,
            exceptions=(TimeoutException,),
            description=f"waiting for {description}",
        )
