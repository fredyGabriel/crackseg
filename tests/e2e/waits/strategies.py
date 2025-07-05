"""Wait strategy implementations for reliable E2E testing.

This module provides the main wait strategy classes that orchestrate
different waiting approaches, from basic explicit waits to intelligent
context-aware waiting strategies.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..utils.time import retry_with_backoff
from .conditions import StreamlitConditions

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WaitContext(Enum):
    """Context types for smart wait strategy selection."""

    GENERAL = "general"
    STREAMLIT_APP = "streamlit_app"
    FILE_UPLOAD = "file_upload"
    FORM_INTERACTION = "form_interaction"
    NAVIGATION = "navigation"
    AJAX_CONTENT = "ajax_content"


@dataclass
class FluentWaitConfig:
    """Configuration for fluent wait behavior.

    Provides configurable polling intervals, timeout settings,
    and exception handling for fluent wait strategies.
    """

    timeout: float = 30.0
    poll_frequency: float = 0.5
    ignored_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (WebDriverException,)
    )
    message: str = ""

    def with_timeout(self, timeout: float) -> "FluentWaitConfig":
        """Create new config with different timeout.

        Args:
            timeout: New timeout value

        Returns:
            New FluentWaitConfig instance
        """
        return FluentWaitConfig(
            timeout=timeout,
            poll_frequency=self.poll_frequency,
            ignored_exceptions=self.ignored_exceptions,
            message=self.message,
        )

    def with_polling(self, poll_frequency: float) -> "FluentWaitConfig":
        """Create new config with different polling frequency.

        Args:
            poll_frequency: New polling frequency

        Returns:
            New FluentWaitConfig instance
        """
        return FluentWaitConfig(
            timeout=self.timeout,
            poll_frequency=poll_frequency,
            ignored_exceptions=self.ignored_exceptions,
            message=self.message,
        )

    def with_message(self, message: str) -> "FluentWaitConfig":
        """Create new config with custom timeout message.

        Args:
            message: Custom timeout message

        Returns:
            New FluentWaitConfig instance
        """
        return FluentWaitConfig(
            timeout=self.timeout,
            poll_frequency=self.poll_frequency,
            ignored_exceptions=self.ignored_exceptions,
            message=message,
        )


class WaitStrategy:
    """Main wait strategy orchestrator.

    Provides a unified interface for different waiting strategies
    including explicit waits, fluent waits, and integration with
    existing utility functions.
    """

    def __init__(
        self,
        driver: WebDriver,
        timeout: float = 30.0,
        poll_frequency: float = 0.5,
    ) -> None:
        """Initialize wait strategy.

        Args:
            driver: WebDriver instance
            timeout: Default timeout for wait operations
            poll_frequency: Default polling frequency
        """
        self.driver = driver
        self.timeout = timeout
        self.poll_frequency = poll_frequency
        self._wait = WebDriverWait(driver, timeout, poll_frequency)

    def until(
        self,
        condition: Callable[[WebDriver], T],
        timeout: float | None = None,
        message: str = "",
    ) -> T:
        """Wait until condition is met.

        Args:
            condition: Condition function to wait for
            timeout: Custom timeout (uses default if None)
            message: Custom timeout message

        Returns:
            Result of condition function

        Raises:
            TimeoutException: If condition not met within timeout
        """
        timeout = timeout or self.timeout
        wait = WebDriverWait(self.driver, timeout, self.poll_frequency)

        try:
            result = wait.until(condition, message)
            logger.debug(f"Wait condition met: {condition}")
            return result
        except TimeoutException as e:
            logger.error(f"Wait timeout after {timeout}s: {condition}")
            raise e

    def until_not(
        self,
        condition: Callable[[WebDriver], Any],
        timeout: float | None = None,
        message: str = "",
    ) -> bool:
        """Wait until condition is no longer met.

        Args:
            condition: Condition function to wait for absence
            timeout: Custom timeout (uses default if None)
            message: Custom timeout message

        Returns:
            True when condition is no longer met

        Raises:
            TimeoutException: If condition still met after timeout
        """
        timeout = timeout or self.timeout
        wait = WebDriverWait(self.driver, timeout, self.poll_frequency)

        try:
            result = wait.until_not(condition, message)
            logger.debug(f"Wait condition no longer met: {condition}")
            return result
        except TimeoutException as e:
            logger.error(f"Wait timeout after {timeout}s: {condition}")
            raise e

    def for_element_present(
        self, locator: tuple[str, str], timeout: float | None = None
    ) -> WebElement:
        """Wait for element to be present in DOM.

        Args:
            locator: Selenium locator tuple
            timeout: Custom timeout

        Returns:
            Located WebElement

        Raises:
            TimeoutException: If element not found within timeout
        """
        condition = EC.presence_of_element_located(locator)
        result = self.until(condition, timeout, f"Element present: {locator}")
        # Type narrowing: until() ensures WebElement or raises exception
        assert isinstance(result, WebElement)
        return result

    def for_element_visible(
        self, locator: tuple[str, str], timeout: float | None = None
    ) -> WebElement:
        """Wait for element to be visible.

        Args:
            locator: Selenium locator tuple
            timeout: Custom timeout

        Returns:
            Visible WebElement

        Raises:
            TimeoutException: If element not visible within timeout
        """
        condition = EC.visibility_of_element_located(locator)
        result = self.until(condition, timeout, f"Element visible: {locator}")
        # Type narrowing: until() ensures WebElement or raises exception
        assert isinstance(result, WebElement)
        return result

    def for_element_clickable(
        self, locator: tuple[str, str], timeout: float | None = None
    ) -> WebElement:
        """Wait for element to be clickable.

        Args:
            locator: Selenium locator tuple
            timeout: Custom timeout

        Returns:
            Clickable WebElement

        Raises:
            TimeoutException: If element not clickable within timeout
        """
        condition = EC.element_to_be_clickable(locator)
        result = self.until(
            condition, timeout, f"Element clickable: {locator}"
        )
        # Type narrowing: until() ensures WebElement or raises exception
        assert isinstance(result, WebElement)
        return result

    def for_elements_count(
        self,
        locator: tuple[str, str],
        count: int,
        timeout: float | None = None,
    ) -> list[WebElement]:
        """Wait for specific number of elements.

        Args:
            locator: Selenium locator tuple
            count: Expected number of elements
            timeout: Custom timeout

        Returns:
            List of WebElements

        Raises:
            TimeoutException: If element count not met within timeout
        """

        def _elements_count_matches(
            driver: WebDriver,
        ) -> list[WebElement] | bool:
            elements = driver.find_elements(*locator)
            return elements if len(elements) == count else False

        result = self.until(
            _elements_count_matches,
            timeout,
            f"Elements count {count}: {locator}",
        )
        # Type narrowing: until() ensures we get the truthy result
        assert isinstance(result, list)
        return result

    def for_text_in_element(
        self,
        locator: tuple[str, str],
        text: str,
        timeout: float | None = None,
    ) -> bool:
        """Wait for text to be present in element.

        Args:
            locator: Selenium locator tuple
            text: Text to wait for
            timeout: Custom timeout

        Returns:
            True when text is present

        Raises:
            TimeoutException: If text not found within timeout
        """
        condition = EC.text_to_be_present_in_element(locator, text)
        return self.until(
            condition, timeout, f"Text '{text}' in element: {locator}"
        )

    def for_streamlit_ready(
        self,
        check_sidebar: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Wait for Streamlit app to be ready.

        Args:
            check_sidebar: Whether to check sidebar presence
            timeout: Custom timeout

        Returns:
            True when Streamlit app is ready

        Raises:
            TimeoutException: If app not ready within timeout
        """
        condition = StreamlitConditions.app_ready(check_sidebar=check_sidebar)
        return self.until(condition, timeout, "Streamlit app ready")

    def with_retry(
        self,
        operation: Callable[[], T],
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> T:
        """Execute operation with retry mechanism.

        Args:
            operation: Operation to retry
            max_retries: Maximum number of retries
            delay: Initial delay between retries
            backoff_factor: Exponential backoff factor

        Returns:
            Result of successful operation

        Raises:
            Last exception if all retries fail
        """
        return retry_with_backoff(
            operation,
            max_attempts=max_retries + 1,
            initial_delay=delay,
            backoff_factor=backoff_factor,
        )

    def with_fluent_config(self, config: FluentWaitConfig) -> "WaitStrategy":
        """Create new wait strategy with fluent configuration.

        Args:
            config: FluentWaitConfig instance

        Returns:
            New WaitStrategy with updated configuration
        """
        return WaitStrategy(
            self.driver,
            timeout=config.timeout,
            poll_frequency=config.poll_frequency,
        )


class SmartWait:
    """Context-aware wait strategy selector.

    Analyzes the testing context and selects optimal wait strategies
    for different scenarios such as Streamlit app loading, file uploads,
    form interactions, etc.
    """

    def __init__(self, driver: WebDriver) -> None:
        """Initialize smart wait strategy.

        Args:
            driver: WebDriver instance
        """
        self.driver = driver
        self._context_strategies: dict[WaitContext, FluentWaitConfig] = {
            WaitContext.GENERAL: FluentWaitConfig(
                timeout=10.0, poll_frequency=0.5
            ),
            WaitContext.STREAMLIT_APP: FluentWaitConfig(
                timeout=30.0, poll_frequency=1.0
            ),
            WaitContext.FILE_UPLOAD: FluentWaitConfig(
                timeout=60.0, poll_frequency=2.0
            ),
            WaitContext.FORM_INTERACTION: FluentWaitConfig(
                timeout=15.0, poll_frequency=0.5
            ),
            WaitContext.NAVIGATION: FluentWaitConfig(
                timeout=20.0, poll_frequency=1.0
            ),
            WaitContext.AJAX_CONTENT: FluentWaitConfig(
                timeout=25.0, poll_frequency=0.8
            ),
        }

    def for_context(self, context: WaitContext) -> WaitStrategy:
        """Get wait strategy optimized for specific context.

        Args:
            context: WaitContext enum value

        Returns:
            WaitStrategy configured for the context
        """
        config = self._context_strategies.get(
            context, self._context_strategies[WaitContext.GENERAL]
        )

        strategy = WaitStrategy(
            self.driver,
            timeout=config.timeout,
            poll_frequency=config.poll_frequency,
        )

        logger.debug(f"Using smart wait strategy for context: {context.value}")
        return strategy

    def detect_context(
        self, locator: tuple[str, str] | None = None
    ) -> WaitContext:
        """Automatically detect wait context based on page state.

        Args:
            locator: Optional element locator for context hints

        Returns:
            Detected WaitContext
        """
        try:
            # Check if we're in a Streamlit app
            streamlit_elements = self.driver.find_elements(
                By.CSS_SELECTOR, "[data-testid='stApp']"
            )
            if streamlit_elements:
                # Check for file upload context
                upload_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stFileUploader']"
                )
                if upload_elements:
                    return WaitContext.FILE_UPLOAD

                # Check for form context
                form_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, "form, [data-testid='stForm']"
                )
                if form_elements:
                    return WaitContext.FORM_INTERACTION

                return WaitContext.STREAMLIT_APP

            # Check for AJAX indicators
            ajax_indicators = self.driver.find_elements(
                By.CSS_SELECTOR, ".loading, .spinner, [data-loading='true']"
            )
            if ajax_indicators:
                return WaitContext.AJAX_CONTENT

            # Default context
            return WaitContext.GENERAL

        except WebDriverException:
            logger.warning("Failed to detect wait context, using general")
            return WaitContext.GENERAL

    def auto_wait(
        self,
        condition: Callable[[WebDriver], T],
        locator: tuple[str, str] | None = None,
        hint_context: WaitContext | None = None,
    ) -> T:
        """Automatically select and execute optimal wait strategy.

        Args:
            condition: Condition function to wait for
            locator: Optional locator for context detection
            hint_context: Optional context hint

        Returns:
            Result of condition function

        Raises:
            TimeoutException: If condition not met within timeout
        """
        context = hint_context or self.detect_context(locator)
        strategy = self.for_context(context)

        logger.debug(f"Auto-wait using context: {context.value}")
        return strategy.until(condition)

    def customize_context(
        self, context: WaitContext, config: FluentWaitConfig
    ) -> None:
        """Customize wait configuration for specific context.

        Args:
            context: WaitContext to customize
            config: New FluentWaitConfig for the context
        """
        self._context_strategies[context] = config
        logger.debug(f"Customized wait strategy for context: {context.value}")

    def get_context_config(self, context: WaitContext) -> FluentWaitConfig:
        """Get current configuration for a context.

        Args:
            context: WaitContext to get config for

        Returns:
            Current FluentWaitConfig for the context
        """
        return self._context_strategies.get(
            context, self._context_strategies[WaitContext.GENERAL]
        )
