"""
Element interaction utilities for Selenium E2E testing. This module
provides robust utilities for common Selenium element interactions
with built-in error handling, retry mechanisms, and enhanced
reliability for testing Streamlit applications.
"""

import logging
import time

from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


def find_element_with_retry(
    driver: WebDriver,
    locator: tuple[str, str],
    timeout: float = 10.0,
    retry_count: int = 3,
) -> WebElement | None:
    """Find element with retry mechanism for flaky scenarios.

    Args:
        driver: WebDriver instance
        locator: Selenium locator tuple (By.ID, "element-id")
        timeout: Timeout for each attempt
        retry_count: Number of retry attempts

    Returns:
        WebElement if found, None otherwise

    Example:
        >>> element = find_element_with_retry(
        ...     driver, (By.ID, "submit-button"), timeout=5.0
        ... )
    """
    for attempt in range(retry_count + 1):
        try:
            wait = WebDriverWait(driver, timeout)
            element = wait.until(EC.presence_of_element_located(locator))
            logger.debug(f"Element found on attempt {attempt + 1}: {locator}")
            return element

        except (TimeoutException, NoSuchElementException) as e:
            if attempt < retry_count:
                logger.warning(
                    f"Element not found (attempt {attempt + 1}): {locator}. "
                    f"Retrying... Error: {e}"
                )
                time.sleep(1.0)
            else:
                logger.error(
                    f"Element not found after {retry_count + 1} attempts: "
                    f"{locator}"
                )

        except WebDriverException as e:
            logger.error(f"WebDriver error finding element {locator}: {e}")
            break

    return None


def wait_for_element_visible(
    driver: WebDriver,
    locator: tuple[str, str],
    timeout: float = 10.0,
) -> WebElement | None:
    """
    Wait for element to be visible and return it. Args: driver: WebDriver
    instance locator: Selenium locator tuple timeout: Maximum time to wait
    Returns: WebElement if visible, None otherwise
    """
    try:
        wait = WebDriverWait(driver, timeout)
        element = wait.until(EC.visibility_of_element_located(locator))
        logger.debug(f"Element became visible: {locator}")
        return element
    except TimeoutException:
        logger.warning(f"Element not visible within {timeout}s: {locator}")
        return None


def wait_for_element_to_be_clickable(
    driver: WebDriver,
    locator: tuple[str, str],
    timeout: float = 10.0,
) -> WebElement | None:
    """
    Wait for element to be clickable and return it. Args: driver:
    WebDriver instance locator: Selenium locator tuple timeout: Maximum
    time to wait Returns: WebElement if clickable, None otherwise
    """
    try:
        wait = WebDriverWait(driver, timeout)
        element = wait.until(EC.element_to_be_clickable(locator))
        logger.debug(f"Element became clickable: {locator}")
        return element
    except TimeoutException:
        logger.warning(f"Element not clickable within {timeout}s: {locator}")
        return None


def wait_for_element_to_disappear(
    driver: WebDriver,
    locator: tuple[str, str],
    timeout: float = 10.0,
) -> bool:
    """
    Wait for element to disappear from DOM. Args: driver: WebDriver
    instance locator: Selenium locator tuple timeout: Maximum time to wait
    Returns: True if element disappeared, False otherwise
    """
    try:
        wait = WebDriverWait(driver, timeout)
        wait.until_not(EC.presence_of_element_located(locator))
        logger.debug(f"Element disappeared: {locator}")
        return True
    except TimeoutException:
        logger.warning(f"Element still present after {timeout}s: {locator}")
        return False


def wait_for_elements_count(
    driver: WebDriver,
    locator: tuple[str, str],
    expected_count: int,
    timeout: float = 10.0,
) -> list[WebElement]:
    """
    Wait for specific number of elements to be present. Args: driver:
    WebDriver instance locator: Selenium locator tuple expected_count:
    Expected number of elements timeout: Maximum time to wait Returns:
    List of WebElements if count matches, empty list otherwise
    """
    try:
        wait = WebDriverWait(driver, timeout)

        def elements_count_matches(
            driver: WebDriver,
        ) -> list[WebElement] | bool:
            elements = driver.find_elements(*locator)
            return elements if len(elements) == expected_count else False

        elements = wait.until(elements_count_matches)
        logger.debug(f"Found {expected_count} elements matching {locator}")
        return elements if isinstance(elements, list) else []
    except TimeoutException:
        current_count = len(driver.find_elements(*locator))
        logger.warning(
            f"Expected {expected_count} elements, found {current_count}: "
            f"{locator}"
        )
        return []


def click_element_safely(
    driver: WebDriver,
    element: WebElement,
    retry_count: int = 3,
) -> bool:
    """
    Click element safely with retry mechanism. Args: driver: WebDriver
    instance element: WebElement to click retry_count: Number of retry
    attempts Returns: True if click successful, False otherwise
    """
    for attempt in range(retry_count + 1):
        try:
            # Scroll to element first
            scroll_to_element(driver, element)
            time.sleep(0.5)  # Brief pause after scrolling

            element.click()
            logger.debug(
                f"Element clicked successfully on attempt {attempt + 1}"
            )
            return True

        except ElementClickInterceptedException:
            # Try using JavaScript click
            try:
                driver.execute_script("arguments[0].click();", element)
                logger.debug("Element clicked using JavaScript")
                return True
            except WebDriverException as e:
                logger.warning(f"JavaScript click failed: {e}")

        except (
            StaleElementReferenceException,
            ElementNotInteractableException,
        ) as e:
            if attempt < retry_count:
                logger.warning(
                    f"Click failed (attempt {attempt + 1}): {e}. Retrying..."
                )
                time.sleep(1.0)
            else:
                logger.error(
                    f"Click failed after {retry_count + 1} attempts: {e}"
                )

        except WebDriverException as e:
            logger.error(f"WebDriver error clicking element: {e}")
            break

    return False


def scroll_to_element(driver: WebDriver, element: WebElement) -> bool:
    """
    Scroll to make element visible in viewport. Args: driver: WebDriver
    instance element: WebElement to scroll to Returns: True if scroll
    successful, False otherwise
    """
    try:
        # Use JavaScript to scroll element into view
        driver.execute_script(
            "arguments[0].scrollIntoView("
            "{behavior: 'smooth', block: 'center'});",
            element,
        )
        time.sleep(0.5)  # Allow time for smooth scrolling
        logger.debug("Successfully scrolled to element")
        return True
    except WebDriverException as e:
        logger.error(f"Failed to scroll to element: {e}")
        return False


def is_element_present(driver: WebDriver, locator: tuple[str, str]) -> bool:
    """
    Check if element is present in DOM. Args: driver: WebDriver instance
    locator: Selenium locator tuple Returns: True if element present,
    False otherwise
    """
    try:
        driver.find_element(*locator)
        return True
    except NoSuchElementException:
        return False


def is_element_visible(driver: WebDriver, locator: tuple[str, str]) -> bool:
    """
    Check if element is visible on page. Args: driver: WebDriver instance
    locator: Selenium locator tuple Returns: True if element visible,
    False otherwise
    """
    try:
        element = driver.find_element(*locator)
        return element.is_displayed()
    except NoSuchElementException:
        return False


def get_element_text_safely(
    driver: WebDriver, locator: tuple[str, str]
) -> str:
    """
    Get element text with safe error handling. Args: driver: WebDriver
    instance locator: Selenium locator tuple Returns: Element text or
    empty string if not found
    """
    try:
        element = driver.find_element(*locator)
        return element.text.strip()
    except NoSuchElementException:
        logger.warning(f"Element not found for text extraction: {locator}")
        return ""
    except WebDriverException as e:
        logger.error(f"Error getting element text: {e}")
        return ""


def get_element_attribute_safely(
    driver: WebDriver,
    locator: tuple[str, str],
    attribute: str,
) -> str:
    """
    Get element attribute with safe error handling. Args: driver:
    WebDriver instance locator: Selenium locator tuple attribute:
    Attribute name to retrieve Returns: Attribute value or empty string if
    not found
    """
    try:
        element = driver.find_element(*locator)
        value = element.get_attribute(attribute)
        return value or ""
    except NoSuchElementException:
        logger.warning(
            f"Element not found for attribute extraction: {locator}"
        )
        return ""
    except WebDriverException as e:
        logger.error(f"Error getting element attribute: {e}")
        return ""
