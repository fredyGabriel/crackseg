"""Streamlit mixin for E2E testing.

This module provides Streamlit-specific assertions and utility methods
for testing Streamlit applications.
"""

from typing import Any, Protocol

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class HasLogging(Protocol):
    """Protocol for classes that have logging capabilities."""

    def log_assertion(
        self, assertion: str, passed: bool, details: str | None = None
    ) -> None:
        """Log assertion results."""
        ...

    def log_test_step(self, step: str, details: str | None = None) -> None:
        """Log a test step with optional details."""
        ...


class StreamlitMixin:
    """Mixin providing Streamlit-specific capabilities for E2E tests.

    Expected to be composed with LoggingMixin for full functionality.
    """

    def assert_streamlit_loaded(
        self, driver: WebDriver, timeout: float = 30.0
    ) -> None:
        """Assert that Streamlit application is fully loaded.

        Args:
            driver: WebDriver instance
            timeout: Maximum time to wait for Streamlit to load

        Raises:
            AssertionError: If Streamlit is not loaded within timeout
        """
        try:
            # Wait for Streamlit running indicator to disappear
            WebDriverWait(driver, timeout).until_not(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '[data-testid="stStatusWidget"]')
                )
            )
            if hasattr(self, "log_assertion"):
                # Type checker doesn't know about composed mixins
                self.log_assertion(  # type: ignore[attr-defined]
                    "Streamlit application loaded", True
                )
        except TimeoutException as e:
            if hasattr(self, "log_assertion"):
                self.log_assertion(  # type: ignore[attr-defined]
                    "Streamlit application loaded", False
                )
            raise AssertionError(
                f"Streamlit application did not load within {timeout} seconds"
            ) from e

    def assert_page_ready_state(
        self, driver: WebDriver, timeout: float = 30.0
    ) -> None:
        """Assert that page is in ready state.

        Args:
            driver: WebDriver instance
            timeout: Maximum time to wait for ready state

        Raises:
            AssertionError: If page is not ready within timeout
        """

        def page_ready() -> str:
            return driver.execute_script("return document.readyState")

        try:
            WebDriverWait(driver, timeout).until(
                lambda d: page_ready() == "complete"
            )
            if hasattr(self, "log_assertion"):
                self.log_assertion(  # type: ignore[attr-defined]
                    "Page ready state complete", True
                )
        except TimeoutException as e:
            ready_state = page_ready()
            if hasattr(self, "log_assertion"):
                self.log_assertion(  # type: ignore[attr-defined]
                    "Page ready state complete", False, f"State: {ready_state}"
                )
            raise AssertionError(
                f"Page did not reach ready state within {timeout} seconds. "
                f"Current state: {ready_state}"
            ) from e

    def wait_for_streamlit_rerun(
        self, driver: WebDriver, timeout: float = 10.0
    ) -> None:
        """Wait for Streamlit rerun to complete.

        Args:
            driver: WebDriver instance
            timeout: Maximum time to wait for rerun

        Raises:
            TimeoutException: If rerun does not complete within timeout
        """
        try:
            # Wait for running indicator to disappear
            WebDriverWait(driver, timeout).until_not(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '[data-testid="stStatusWidget"]')
                )
            )
            if hasattr(self, "log_test_step"):
                self.log_test_step("Streamlit rerun completed")  # type: ignore[attr-defined]
        except TimeoutException:
            if hasattr(self, "log_test_step"):
                self.log_test_step(  # type: ignore[attr-defined]
                    "Streamlit rerun timeout", f"After {timeout}s"
                )
            raise

    def get_streamlit_element(
        self, driver: WebDriver, test_id: str, timeout: float = 10.0
    ) -> Any:
        """Get Streamlit element by test ID.

        Args:
            driver: WebDriver instance
            test_id: Streamlit test ID
            timeout: Maximum time to wait for element

        Returns:
            WebElement if found

        Raises:
            TimeoutException: If element not found within timeout
        """
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, f'[data-testid="{test_id}"]')
            )
        )

    def wait_for_streamlit_sidebar(
        self, driver: WebDriver, timeout: float = 10.0
    ) -> None:
        """Wait for Streamlit sidebar to be available.

        Args:
            driver: WebDriver instance
            timeout: Maximum time to wait for sidebar

        Raises:
            TimeoutException: If sidebar not available within timeout
        """
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, '[data-testid="stSidebar"]')
            )
        )
        if hasattr(self, "log_test_step"):
            self.log_test_step("Streamlit sidebar loaded")  # type: ignore[attr-defined]
