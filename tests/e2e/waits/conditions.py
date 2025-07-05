"""Custom wait conditions for E2E testing.

This module provides custom wait conditions specifically designed for
Streamlit applications and general web testing scenarios. Includes
both individual condition functions and condition factory classes.
"""

import re
import time
from collections.abc import Callable
from typing import Any

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement


def element_text_matches(
    locator: tuple[str, str], pattern: str, regex: bool = False
) -> Callable[[WebDriver], WebElement | bool]:
    """Wait condition for element text to match pattern.

    Args:
        locator: Selenium locator tuple
        pattern: Text pattern to match
        regex: Whether to use regex matching

    Returns:
        Condition function for WebDriverWait
    """

    def _predicate(driver: WebDriver) -> WebElement | bool:
        try:
            element = driver.find_element(*locator)
            text = element.text

            if regex:
                return element if re.search(pattern, text) else False
            else:
                return element if pattern in text else False

        except (NoSuchElementException, StaleElementReferenceException):
            return False

    return _predicate


def element_count_equals(
    locator: tuple[str, str], expected_count: int
) -> Callable[[WebDriver], list[WebElement] | bool]:
    """Wait condition for specific number of elements.

    Args:
        locator: Selenium locator tuple
        expected_count: Expected number of elements

    Returns:
        Condition function returning elements if count matches
    """

    def _predicate(driver: WebDriver) -> list[WebElement] | bool:
        try:
            elements = driver.find_elements(*locator)
            return elements if len(elements) == expected_count else False
        except WebDriverException:
            return False

    return _predicate


def element_attribute_contains(
    locator: tuple[str, str], attribute: str, value: str
) -> Callable[[WebDriver], WebElement | bool]:
    """Wait condition for element attribute to contain value.

    Args:
        locator: Selenium locator tuple
        attribute: HTML attribute name
        value: Expected value in attribute

    Returns:
        Condition function for WebDriverWait
    """

    def _predicate(driver: WebDriver) -> WebElement | bool:
        try:
            element = driver.find_element(*locator)
            attr_value = element.get_attribute(attribute)

            if attr_value and value in attr_value:
                return element
            return False

        except (NoSuchElementException, StaleElementReferenceException):
            return False

    return _predicate


def text_to_be_present_in_element_value(
    locator: tuple[str, str], text: str
) -> Callable[[WebDriver], bool]:
    """Wait condition for text in element value attribute.

    Args:
        locator: Selenium locator tuple
        text: Text to check for in value

    Returns:
        Condition function for WebDriverWait
    """

    def _predicate(driver: WebDriver) -> bool:
        try:
            element = driver.find_element(*locator)
            value = element.get_attribute("value")
            return text in (value or "")
        except (NoSuchElementException, StaleElementReferenceException):
            return False

    return _predicate


class CustomConditions:
    """Factory class for custom wait conditions.

    Provides reusable wait conditions for common testing scenarios
    that are not covered by Selenium's built-in expected_conditions.
    """

    @staticmethod
    def element_to_be_stale(
        element: WebElement,
    ) -> Callable[[WebDriver], bool]:
        """Wait for element to become stale (removed from DOM).

        Args:
            element: WebElement to check for staleness

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> bool:
            try:
                # Try to access element properties
                element.is_enabled()
                return False
            except StaleElementReferenceException:
                return True

        return _predicate

    @staticmethod
    def element_attribute_to_be_updated(
        locator: tuple[str, str], attribute: str, initial_value: str
    ) -> Callable[[WebDriver], WebElement | bool]:
        """Wait for element attribute value to change.

        Args:
            locator: Selenium locator tuple
            attribute: Attribute name to monitor
            initial_value: Initial value to compare against

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> WebElement | bool:
            try:
                element = driver.find_element(*locator)
                current_value = element.get_attribute(attribute)
                return element if current_value != initial_value else False
            except (NoSuchElementException, StaleElementReferenceException):
                return False

        return _predicate

    @staticmethod
    def page_title_matches(
        pattern: str, regex: bool = False
    ) -> Callable[[WebDriver], bool]:
        """Wait for page title to match pattern.

        Args:
            pattern: Title pattern to match
            regex: Whether to use regex matching

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> bool:
            try:
                title = driver.title
                if regex:
                    return bool(re.search(pattern, title))
                else:
                    return pattern in title
            except WebDriverException:
                return False

        return _predicate


class StreamlitConditions:
    """Streamlit-specific wait conditions.

    Provides wait conditions specifically designed for testing
    Streamlit applications, handling app lifecycle, widgets,
    and dynamic content loading.
    """

    @staticmethod
    def app_ready(
        check_sidebar: bool = True,
        additional_elements: list[str] | None = None,
    ) -> Callable[[WebDriver], bool]:
        """Wait for Streamlit app to be fully ready.

        Args:
            check_sidebar: Whether to check for sidebar presence
            additional_elements: Additional CSS selectors to check

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> bool:
            try:
                # Check basic Streamlit app structure
                app_element = driver.find_element(
                    By.CSS_SELECTOR, "[data-testid='stApp']"
                )
                if not app_element:
                    return False

                # Check no loading spinners
                spinners = driver.find_elements(By.CLASS_NAME, "stSpinner")
                if spinners:
                    return False

                # Check sidebar if requested
                if check_sidebar:
                    sidebar = driver.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stSidebar']"
                    )
                    if not sidebar:
                        return False

                # Check additional elements
                if additional_elements:
                    for selector in additional_elements:
                        elements = driver.find_elements(
                            By.CSS_SELECTOR, selector
                        )
                        if not elements:
                            return False

                return True

            except WebDriverException:
                return False

        return _predicate

    @staticmethod
    def sidebar_loaded() -> Callable[[WebDriver], bool]:
        """Wait for Streamlit sidebar to be loaded and visible.

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> bool:
            try:
                sidebar = driver.find_element(
                    By.CSS_SELECTOR, "[data-testid='stSidebar']"
                )
                return sidebar.is_displayed()
            except (NoSuchElementException, WebDriverException):
                return False

        return _predicate

    @staticmethod
    def no_spinners_present() -> Callable[[WebDriver], bool]:
        """Wait for all Streamlit loading spinners to disappear.

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> bool:
            try:
                spinners = driver.find_elements(By.CLASS_NAME, "stSpinner")
                return len(spinners) == 0
            except WebDriverException:
                return False

        return _predicate

    @staticmethod
    def file_upload_complete(
        uploader_locator: tuple[str, str] | None = None,
    ) -> Callable[[WebDriver], bool]:
        """Wait for file upload to complete in Streamlit.

        Args:
            uploader_locator: Optional specific file uploader locator

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> bool:
            try:
                # Check for upload progress indicators
                progress_indicators = driver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stProgress']"
                )
                if progress_indicators:
                    return False

                # Check specific uploader if provided
                if uploader_locator:
                    uploader = driver.find_element(*uploader_locator)
                    # Look for success indicators in the uploader
                    success_elements = uploader.find_elements(
                        By.CSS_SELECTOR, ".uploadedFile, .uploaded-file"
                    )
                    return len(success_elements) > 0

                # General check for upload completion
                upload_elements = driver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stFileUploader']"
                )
                for element in upload_elements:
                    # Check if any uploads are in progress
                    progress = element.find_elements(
                        By.CSS_SELECTOR, ".stProgress"
                    )
                    if progress:
                        return False

                return True

            except WebDriverException:
                return False

        return _predicate

    @staticmethod
    def rerun_complete(
        max_wait_for_spinner: float = 2.0,
    ) -> Callable[[WebDriver], bool]:
        """Wait for Streamlit rerun to complete.

        Args:
            max_wait_for_spinner: Max time to wait for spinner to appear

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> bool:
            try:
                start_time = time.time()

                # First, wait a short time for spinners to potentially appear
                while time.time() - start_time < max_wait_for_spinner:
                    spinners = driver.find_elements(By.CLASS_NAME, "stSpinner")
                    if spinners:
                        break
                    time.sleep(0.1)

                # Then wait for all spinners to disappear
                spinners = driver.find_elements(By.CLASS_NAME, "stSpinner")
                return len(spinners) == 0

            except WebDriverException:
                return False

        return _predicate

    @staticmethod
    def session_state_contains(
        key: str, expected_value: Any = None
    ) -> Callable[[WebDriver], bool]:
        """Wait for session state to contain specific key/value.

        Args:
            key: Session state key to check
            expected_value: Expected value (None means just check key exists)

        Returns:
            Condition function for WebDriverWait
        """

        def _predicate(driver: WebDriver) -> bool:
            try:
                # Execute JavaScript to check session state
                script = f"""
                return window.streamlitSessionState &&
                       window.streamlitSessionState.hasOwnProperty('{key}');
                """

                if expected_value is not None:
                    script = f"""
                    return window.streamlitSessionState &&
                           window.streamlitSessionState['{key}'] ===
                           arguments[0];
                    """
                    return driver.execute_script(script, expected_value)
                else:
                    return driver.execute_script(script)

            except WebDriverException:
                return False

        return _predicate
