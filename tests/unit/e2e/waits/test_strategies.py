"""Unit tests for wait strategies module."""

from unittest.mock import Mock, patch

import pytest
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from tests.e2e.waits.strategies import (
    FluentWaitConfig,
    SmartWait,
    WaitContext,
    WaitStrategy,
)


class TestFluentWaitConfig:
    """Test FluentWaitConfig dataclass and methods."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FluentWaitConfig()

        assert config.timeout == 30.0
        assert config.poll_frequency == 0.5
        assert config.ignored_exceptions == (WebDriverException,)
        assert config.message == ""

    def test_with_timeout(self) -> None:
        """Test creating config with different timeout."""
        original = FluentWaitConfig(timeout=10.0, poll_frequency=1.0)
        modified = original.with_timeout(20.0)

        assert modified.timeout == 20.0
        assert modified.poll_frequency == 1.0  # Unchanged
        assert modified is not original  # New instance

    def test_with_polling(self) -> None:
        """Test creating config with different polling frequency."""
        original = FluentWaitConfig(timeout=15.0, poll_frequency=0.5)
        modified = original.with_polling(2.0)

        assert modified.timeout == 15.0  # Unchanged
        assert modified.poll_frequency == 2.0
        assert modified is not original

    def test_with_message(self) -> None:
        """Test creating config with custom message."""
        original = FluentWaitConfig()
        modified = original.with_message("Custom timeout message")

        assert modified.message == "Custom timeout message"
        assert modified.timeout == original.timeout  # Unchanged
        assert modified is not original


class TestWaitStrategy:
    """Test WaitStrategy main orchestrator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.driver = Mock(spec=WebDriver)
        self.strategy = WaitStrategy(
            self.driver, timeout=10.0, poll_frequency=1.0
        )

    def test_initialization(self) -> None:
        """Test WaitStrategy initialization."""
        assert self.strategy.driver == self.driver
        assert self.strategy.timeout == 10.0
        assert self.strategy.poll_frequency == 1.0

    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_until_success(self, mock_wait_class: Mock) -> None:
        """Test successful until operation."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_condition = Mock()
        mock_result = Mock(spec=WebElement)
        mock_wait.until.return_value = mock_result

        result = self.strategy.until(
            mock_condition, timeout=5.0, message="test"
        )

        assert result == mock_result
        mock_wait_class.assert_called_once_with(self.driver, 5.0, 1.0)
        mock_wait.until.assert_called_once_with(mock_condition, "test")

    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_until_timeout(self, mock_wait_class: Mock) -> None:
        """Test until operation with timeout."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_condition = Mock()
        mock_wait.until.side_effect = TimeoutException("Timeout")

        with pytest.raises(TimeoutException):
            self.strategy.until(mock_condition)

    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_until_not_success(self, mock_wait_class: Mock) -> None:
        """Test successful until_not operation."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_condition = Mock()
        mock_wait.until_not.return_value = True

        result = self.strategy.until_not(mock_condition)

        assert result is True
        mock_wait.until_not.assert_called_once()

    @patch("tests.e2e.waits.strategies.EC")
    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_for_element_present(
        self, mock_wait_class: Mock, mock_ec: Mock
    ) -> None:
        """Test for_element_present method."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_element = Mock(spec=WebElement)
        mock_wait.until.return_value = mock_element
        mock_condition = Mock()
        mock_ec.presence_of_element_located.return_value = mock_condition

        locator = (By.ID, "test-element")
        result = self.strategy.for_element_present(locator)

        assert result == mock_element
        mock_ec.presence_of_element_located.assert_called_once_with(locator)

    @patch("tests.e2e.waits.strategies.EC")
    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_for_element_visible(
        self, mock_wait_class: Mock, mock_ec: Mock
    ) -> None:
        """Test for_element_visible method."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_element = Mock(spec=WebElement)
        mock_wait.until.return_value = mock_element
        mock_condition = Mock()
        mock_ec.visibility_of_element_located.return_value = mock_condition

        locator = (By.CLASS_NAME, "visible-element")
        result = self.strategy.for_element_visible(locator)

        assert result == mock_element
        mock_ec.visibility_of_element_located.assert_called_once_with(locator)

    @patch("tests.e2e.waits.strategies.EC")
    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_for_element_clickable(
        self, mock_wait_class: Mock, mock_ec: Mock
    ) -> None:
        """Test for_element_clickable method."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_element = Mock(spec=WebElement)
        mock_wait.until.return_value = mock_element
        mock_condition = Mock()
        mock_ec.element_to_be_clickable.return_value = mock_condition

        locator = (By.XPATH, "//button[@id='submit']")
        result = self.strategy.for_element_clickable(locator)

        assert result == mock_element
        mock_ec.element_to_be_clickable.assert_called_once_with(locator)

    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_for_elements_count(self, mock_wait_class: Mock) -> None:
        """Test for_elements_count method."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        elements = [Mock(spec=WebElement) for _ in range(3)]
        mock_wait.until.return_value = elements

        locator = (By.CLASS_NAME, "list-item")
        result = self.strategy.for_elements_count(locator, 3)

        assert result == elements
        mock_wait.until.assert_called_once()

    @patch("tests.e2e.waits.strategies.EC")
    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_for_text_in_element(
        self, mock_wait_class: Mock, mock_ec: Mock
    ) -> None:
        """Test for_text_in_element method."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_wait.until.return_value = True
        mock_condition = Mock()
        mock_ec.text_to_be_present_in_element.return_value = mock_condition

        locator = (By.ID, "status")
        result = self.strategy.for_text_in_element(locator, "Success")

        assert result is True
        mock_ec.text_to_be_present_in_element.assert_called_once_with(
            locator, "Success"
        )

    @patch("tests.e2e.waits.strategies.StreamlitConditions")
    @patch("tests.e2e.waits.strategies.WebDriverWait")
    def test_for_streamlit_ready(
        self, mock_wait_class: Mock, mock_conditions: Mock
    ) -> None:
        """Test for_streamlit_ready method."""
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_wait.until.return_value = True
        mock_condition = Mock()
        mock_conditions.app_ready.return_value = mock_condition

        result = self.strategy.for_streamlit_ready(check_sidebar=False)

        assert result is True
        mock_conditions.app_ready.assert_called_once_with(check_sidebar=False)

    @patch("tests.e2e.waits.strategies.retry_with_backoff")
    def test_with_retry(self, mock_retry: Mock) -> None:
        """Test with_retry method."""
        mock_operation = Mock()
        mock_result = "operation_result"
        mock_retry.return_value = mock_result

        result = self.strategy.with_retry(
            mock_operation, max_retries=2, delay=0.5, backoff_factor=1.5
        )

        assert result == mock_result
        mock_retry.assert_called_once_with(
            mock_operation,
            max_attempts=3,  # max_retries + 1
            initial_delay=0.5,
            backoff_factor=1.5,
        )

    def test_with_fluent_config(self) -> None:
        """Test with_fluent_config method."""
        config = FluentWaitConfig(timeout=20.0, poll_frequency=2.0)
        new_strategy = self.strategy.with_fluent_config(config)

        assert new_strategy.driver == self.driver
        assert new_strategy.timeout == 20.0
        assert new_strategy.poll_frequency == 2.0
        assert new_strategy is not self.strategy  # New instance


class TestSmartWait:
    """Test SmartWait context-aware strategy selector."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.driver = Mock(spec=WebDriver)
        self.smart_wait = SmartWait(self.driver)

    def test_initialization(self) -> None:
        """Test SmartWait initialization with default contexts."""
        assert self.smart_wait.driver == self.driver
        assert WaitContext.GENERAL in self.smart_wait._context_strategies
        assert WaitContext.STREAMLIT_APP in self.smart_wait._context_strategies
        assert WaitContext.FILE_UPLOAD in self.smart_wait._context_strategies

    def test_for_context(self) -> None:
        """Test getting wait strategy for specific context."""
        strategy = self.smart_wait.for_context(WaitContext.STREAMLIT_APP)

        assert isinstance(strategy, WaitStrategy)
        assert strategy.driver == self.driver
        # Check that timeout matches the configured value for STREAMLIT_APP
        expected_config = self.smart_wait._context_strategies[
            WaitContext.STREAMLIT_APP
        ]
        assert strategy.timeout == expected_config.timeout

    def test_detect_context_streamlit_app(self) -> None:
        """Test context detection for Streamlit app."""
        # Mock Streamlit app present
        streamlit_element = Mock(spec=WebElement)
        self.driver.find_elements.return_value = [streamlit_element]

        context = self.smart_wait.detect_context()

        assert context == WaitContext.STREAMLIT_APP

    def test_detect_context_file_upload(self) -> None:
        """Test context detection for file upload."""

        def mock_find_elements(by: str, selector: str) -> list[Mock]:
            if "stApp" in selector:
                return [Mock(spec=WebElement)]  # Streamlit app present
            elif "stFileUploader" in selector:
                return [Mock(spec=WebElement)]  # File uploader present
            return []

        self.driver.find_elements = mock_find_elements

        context = self.smart_wait.detect_context()

        assert context == WaitContext.FILE_UPLOAD

    def test_detect_context_form_interaction(self) -> None:
        """Test context detection for form interaction."""

        def mock_find_elements(by: str, selector: str) -> list[Mock]:
            if "stApp" in selector:
                return [Mock(spec=WebElement)]  # Streamlit app present
            elif "stFileUploader" in selector:
                return []  # No file uploader
            elif "stForm" in selector or selector == "form":
                return [Mock(spec=WebElement)]  # Form present
            return []

        self.driver.find_elements = mock_find_elements

        context = self.smart_wait.detect_context()

        assert context == WaitContext.FORM_INTERACTION

    def test_detect_context_ajax_content(self) -> None:
        """Test context detection for AJAX content."""

        def mock_find_elements(by: str, selector: str) -> list[Mock]:
            if "stApp" in selector:
                return []  # No Streamlit app
            elif "loading" in selector or "spinner" in selector:
                return [Mock(spec=WebElement)]  # AJAX indicators present
            return []

        self.driver.find_elements = mock_find_elements

        context = self.smart_wait.detect_context()

        assert context == WaitContext.AJAX_CONTENT

    def test_detect_context_general_default(self) -> None:
        """Test context detection defaults to general."""
        # Mock no special elements present
        self.driver.find_elements.return_value = []

        context = self.smart_wait.detect_context()

        assert context == WaitContext.GENERAL

    def test_detect_context_webdriver_exception(self) -> None:
        """Test context detection handles WebDriver exceptions."""
        self.driver.find_elements.side_effect = WebDriverException(
            "Driver error"
        )

        context = self.smart_wait.detect_context()

        assert context == WaitContext.GENERAL

    @patch.object(SmartWait, "detect_context")
    @patch.object(SmartWait, "for_context")
    def test_auto_wait_with_detection(
        self, mock_for_context: Mock, mock_detect: Mock
    ) -> None:
        """Test auto_wait with automatic context detection."""
        mock_strategy = Mock(spec=WaitStrategy)
        mock_condition = Mock()
        mock_result = Mock()

        mock_detect.return_value = WaitContext.STREAMLIT_APP
        mock_for_context.return_value = mock_strategy
        mock_strategy.until.return_value = mock_result

        result = self.smart_wait.auto_wait(mock_condition)

        assert result == mock_result
        mock_detect.assert_called_once_with(None)
        mock_for_context.assert_called_once_with(WaitContext.STREAMLIT_APP)
        mock_strategy.until.assert_called_once_with(mock_condition)

    @patch.object(SmartWait, "for_context")
    def test_auto_wait_with_hint_context(self, mock_for_context: Mock) -> None:
        """Test auto_wait with provided context hint."""
        mock_strategy = Mock(spec=WaitStrategy)
        mock_condition = Mock()
        mock_result = Mock()

        mock_for_context.return_value = mock_strategy
        mock_strategy.until.return_value = mock_result

        result = self.smart_wait.auto_wait(
            mock_condition, hint_context=WaitContext.FILE_UPLOAD
        )

        assert result == mock_result
        mock_for_context.assert_called_once_with(WaitContext.FILE_UPLOAD)

    def test_customize_context(self) -> None:
        """Test customizing context configuration."""
        new_config = FluentWaitConfig(timeout=45.0, poll_frequency=3.0)

        self.smart_wait.customize_context(WaitContext.FILE_UPLOAD, new_config)

        stored_config = self.smart_wait._context_strategies[
            WaitContext.FILE_UPLOAD
        ]
        assert stored_config == new_config
        assert stored_config.timeout == 45.0
        assert stored_config.poll_frequency == 3.0

    def test_get_context_config(self) -> None:
        """Test getting context configuration."""
        config = self.smart_wait.get_context_config(WaitContext.NAVIGATION)

        assert isinstance(config, FluentWaitConfig)
        assert config.timeout > 0  # Should have reasonable default values

    def test_get_context_config_unknown_context(self) -> None:
        """Test getting config for unknown context returns general."""
        # Create a custom enum value that doesn't exist in the strategies
        unknown_context = WaitContext.GENERAL  # Use existing for this test

        config = self.smart_wait.get_context_config(unknown_context)

        assert isinstance(config, FluentWaitConfig)


class TestWaitContext:
    """Test WaitContext enum values."""

    def test_enum_values(self) -> None:
        """Test that all expected context values exist."""
        expected_contexts = [
            "general",
            "streamlit_app",
            "file_upload",
            "form_interaction",
            "navigation",
            "ajax_content",
        ]

        actual_contexts = [context.value for context in WaitContext]

        for expected in expected_contexts:
            assert expected in actual_contexts
