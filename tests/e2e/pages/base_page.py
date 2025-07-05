"""Base page class for Page Object Model implementation.

This module provides the foundational BasePage class that implements common
functionality shared across all Streamlit page objects, including navigation,
validation, wait strategies, and fluent interface patterns.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..utils import (
    is_element_present,
    wait_for_element_visible,
    wait_for_streamlit_ready,
    wait_for_streamlit_rerun,
)
from .locators import BaseLocators, SidebarLocators


class BasePage(ABC):
    """Abstract base class for all page objects.

    Provides common functionality including navigation, validation,
    element interactions, and Streamlit-specific utilities. All page
    objects should inherit from this class.
    """

    def __init__(self, driver: WebDriver, timeout: float = 10.0) -> None:
        """Initialize base page object.

        Args:
            driver: Selenium WebDriver instance
            timeout: Default timeout for element operations
        """
        self.driver = driver
        self.timeout = timeout
        self.wait = WebDriverWait(driver, timeout)

    @property
    @abstractmethod
    def page_name(self) -> str:
        """Return the name of the page for navigation and validation."""
        pass

    @property
    @abstractmethod
    def page_title_locator(self) -> tuple[str, str]:
        """Return the locator for the page title element."""
        pass

    @property
    @abstractmethod
    def expected_url_fragment(self) -> str | None:
        """Return expected URL fragment for this page (if applicable)."""
        pass

    def is_page_loaded(self, timeout: float | None = None) -> bool:
        """Check if the page is properly loaded.

        Args:
            timeout: Custom timeout for this check

        Returns:
            True if page is loaded, False otherwise
        """
        timeout = timeout or self.timeout

        try:
            # Check for Streamlit app structure
            self.wait_for_element(BaseLocators.STREAMLIT_APP, timeout)

            # Check for page-specific title
            self.wait_for_element(self.page_title_locator, timeout)

            # Wait for Streamlit to be ready
            wait_for_streamlit_ready(self.driver, int(timeout))

            return True

        except (TimeoutException, WebDriverException):
            return False

    def navigate_to_page(self, wait_for_load: bool = True) -> "BasePage":
        """Navigate to this page using sidebar navigation.

        Uses a multi-strategy approach to locate and click navigation buttons,
        providing robust fallback mechanisms for different Streamlit rendering
        scenarios.

        Args:
            wait_for_load: Whether to wait for page to load after navigation

        Returns:
            Self for method chaining (fluent interface)

        Raises:
            WebDriverException: If navigation fails with all strategies
        """
        try:
            # Define navigation strategies with descriptive names
            strategies = [
                (
                    "CSS Multi-Strategy",
                    SidebarLocators.nav_button(self.page_name),
                ),
                (
                    "XPath Fallback",
                    SidebarLocators.nav_button_xpath_fallback(self.page_name),
                ),
                (
                    "Key-Based Direct",
                    SidebarLocators.nav_button_by_key(self.page_name),
                ),
            ]

            nav_button = None
            last_exception = None

            # Try each navigation strategy
            for strategy_name, locator_tuple in strategies:
                try:
                    print(f"🔍 Trying {strategy_name} for '{self.page_name}'")
                    by_method, selector = locator_tuple
                    nav_button = self.wait_for_element(
                        (by_method, selector), timeout=5.0
                    )
                    if (
                        nav_button
                        and nav_button.is_displayed()
                        and nav_button.is_enabled()
                    ):
                        # Additional verification: check if button text matches
                        button_text = nav_button.text.strip().lower()
                        page_name_lower = self.page_name.lower()

                        if (
                            page_name_lower in button_text
                            or button_text in page_name_lower
                        ):
                            print(
                                f"✅ Navigation strategy '{strategy_name}' "
                                f"successful for '{self.page_name}'"
                            )
                            break
                        else:
                            print(
                                f"⚠️ Button found but text mismatch: expected"
                                f" '{self.page_name}', got '{nav_button.text}'"
                            )
                            nav_button = None
                    else:
                        nav_button = None

                except (TimeoutException, NoSuchElementException) as e:
                    last_exception = e
                    print(
                        f"❌ Navigation strategy '{strategy_name}' failed: "
                        f"{str(e)[:100]}"
                    )
                    continue

            if nav_button is None:
                # Final fallback: try to find any button in sidebar with
                # matching text
                try:
                    all_buttons = self.driver.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stSidebar'] button"
                    )

                    for button in all_buttons:
                        if (
                            button.is_displayed()
                            and self.page_name.lower() in button.text.lower()
                        ):
                            nav_button = button
                            print(
                                f"✅ Fallback text search successful for "
                                f"'{self.page_name}'"
                            )
                            break

                except Exception as e:
                    print(f"❌ Fallback strategy also failed: {e}")

            if nav_button is None:
                raise NoSuchElementException(
                    f"Navigation button for '{self.page_name}' not found with "
                    f"any strategy. Last exception: {last_exception}"
                )

            # Scroll button into view and click
            self.driver.execute_script(
                "arguments[0].scrollIntoView(true);", nav_button
            )
            time.sleep(0.5)  # Small delay after scroll

            # Try click with JavaScript if regular click fails
            try:
                nav_button.click()
            except WebDriverException:
                self.driver.execute_script("arguments[0].click();", nav_button)
                print(f"⚡ Used JavaScript click for '{self.page_name}'")

            if wait_for_load:
                # Wait for page transition and rerun to complete
                time.sleep(1)
                wait_for_streamlit_rerun(self.driver, int(self.timeout))

                # Verify page loaded
                if not self.is_page_loaded():
                    raise WebDriverException(
                        f"Page '{self.page_name}' did not load properly "
                        "after navigation"
                    )

            return self

        except (TimeoutException, NoSuchElementException) as e:
            raise WebDriverException(
                f"Failed to navigate to {self.page_name}: {e}"
            ) from e

    def wait_for_element(
        self,
        locator: tuple[str, str],
        timeout: float | None = None,
        visible: bool = True,
    ) -> Any | None:
        """Wait for element to be present and optionally visible.

        Args:
            locator: Tuple of (By strategy, selector)
            timeout: Custom timeout for this operation
            visible: Whether element should be visible

        Returns:
            WebElement if found, None otherwise
        """
        timeout = timeout or self.timeout

        try:
            if visible:
                return wait_for_element_visible(self.driver, locator, timeout)
            else:
                wait = WebDriverWait(self.driver, timeout)
                return wait.until(EC.presence_of_element_located(locator))

        except TimeoutException:
            return None

    def click_element(
        self,
        locator: tuple[str, str],
        wait_for_clickable: bool = True,
        wait_for_rerun: bool = False,
    ) -> "BasePage":
        """Click an element with optional wait strategies.

        Args:
            locator: Tuple of (By strategy, selector)
            wait_for_clickable: Wait for element to be clickable
            wait_for_rerun: Wait for Streamlit rerun after click

        Returns:
            Self for method chaining

        Raises:
            WebDriverException: If click operation fails
        """
        try:
            if wait_for_clickable:
                element = self.wait.until(EC.element_to_be_clickable(locator))
                element.click()
            else:
                element = self.wait_for_element(locator)
                if element:
                    element.click()

            if wait_for_rerun:
                # Correctly call the utility function
                from ..utils import wait_for_streamlit_rerun

                wait_for_streamlit_rerun(self.driver, int(self.timeout))

            return self

        except (TimeoutException, WebDriverException) as e:
            raise WebDriverException(
                f"Failed to click element {locator}: {e}"
            ) from e

    def get_element_text(self, locator: tuple[str, str]) -> str | None:
        """Get text content of an element.

        Args:
            locator: Tuple of (By strategy, selector)

        Returns:
            Element text content, or None if element not found
        """
        try:
            element = self.wait_for_element(locator)
            return element.text if element else None
        except WebDriverException:
            return None

    def is_element_displayed(self, locator: tuple[str, str]) -> bool:
        """Check if element is present and visible.

        Args:
            locator: Tuple of (By strategy, selector)

        Returns:
            True if element is displayed, False otherwise
        """
        return (
            is_element_present(self.driver, locator)
            and self.wait_for_element(locator) is not None
        )

    def wait_for_page_ready(self, timeout: float | None = None) -> "BasePage":
        """Wait for page to be fully ready for interaction.

        Args:
            timeout: Custom timeout for this operation

        Returns:
            Self for method chaining

        Raises:
            TimeoutException: If page doesn't become ready within timeout
        """
        timeout = timeout or self.timeout

        try:
            # Wait for basic Streamlit readiness
            wait_for_streamlit_ready(self.driver, int(timeout))

            # Wait for page-specific elements
            self.wait_for_element(self.page_title_locator, timeout)

            # Allow time for dynamic content
            time.sleep(0.5)

            return self

        except TimeoutException as e:
            raise TimeoutException(
                f"Page {self.page_name} not ready within {timeout} seconds"
            ) from e

    def get_page_title(self) -> str | None:
        """Get the current page title text.

        Returns:
            Page title text, or None if not found
        """
        return self.get_element_text(self.page_title_locator)

    def validate_page_loaded(self) -> bool:
        """Validate that the correct page is loaded.

        Returns:
            True if correct page is loaded, False otherwise
        """
        return self.is_page_loaded()

    def take_screenshot(self, filename: str | None = None) -> bool:
        """Take screenshot of current page state.

        Args:
            filename: Optional custom filename

        Returns:
            True if screenshot was saved successfully, False otherwise
        """
        try:
            if filename is None:
                timestamp = int(time.time())
                filename = f"{self.page_name.lower()}_page_{timestamp}.png"

            return self.driver.save_screenshot(filename)

        except WebDriverException:
            return False

    def wait_for_no_spinners(self, timeout: float | None = None) -> "BasePage":
        """Wait for all Streamlit spinners to disappear.

        Args:
            timeout: Custom timeout for this operation

        Returns:
            Self for method chaining
        """
        timeout = timeout or self.timeout

        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until_not(
                EC.presence_of_element_located(BaseLocators.SPINNER)
            )
        except TimeoutException:
            # Spinners might not be present, which is fine
            pass

        return self

    def scroll_to_element(self, locator: tuple[str, str]) -> "BasePage":
        """Scroll element into view.

        Args:
            locator: Tuple of (By strategy, selector)

        Returns:
            Self for method chaining
        """
        try:
            element = self.wait_for_element(locator)
            if element:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView(true);", element
                )
                time.sleep(0.5)  # Allow scroll to complete

        except WebDriverException:
            pass  # Ignore scroll failures

        return self

    def refresh_page(self) -> "BasePage":
        """Refresh the current page and wait for it to load.

        Returns:
            Self for method chaining
        """
        self.driver.refresh()
        self.wait_for_page_ready()
        return self

    def __str__(self) -> str:
        """String representation of the page object."""
        return f"{self.__class__.__name__}(page_name='{self.page_name}')"

    def __repr__(self) -> str:
        """Developer representation of the page object."""
        return self.__str__()
