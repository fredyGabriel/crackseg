"""Unit tests for BaseE2ETest and its mixin components.

This module tests the abstract base test class and all its mixins to ensure
proper functionality, integration, and type safety.
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.remote.webdriver import WebDriver

from tests.e2e.base_test import (
    BaseE2ETest,
    LoggingMixin,
    RetryMixin,
    StreamlitAssertionMixin,
    StreamlitUtilityMixin,
)
from tests.e2e.drivers import DriverConfig


class ConcreteTestClass(BaseE2ETest):
    """Concrete implementation of BaseE2ETest for testing."""

    def setup_test_data(self) -> dict[str, Any]:
        """Implement required abstract method."""
        return {
            "test_url": "http://localhost:8501",
            "expected_title": "CrackSeg Test",
            "timeout": 30.0,
        }


class TestLoggingMixin:
    """Test suite for LoggingMixin functionality."""

    def test_initialization(self) -> None:
        """Test that LoggingMixin initializes correctly."""
        mixin = LoggingMixin()
        assert hasattr(mixin, "_test_logger")
        assert isinstance(mixin._test_logger, logging.Logger)

    @patch("tests.e2e.base_test.logging.getLogger")
    def test_log_test_step(self, mock_get_logger: MagicMock) -> None:
        """Test log_test_step method."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mixin = LoggingMixin()
        mixin.log_test_step("test step", "additional details")

        mock_logger.info.assert_called_once_with(
            "TEST STEP: test step - additional details"
        )

    @patch("tests.e2e.base_test.logging.getLogger")
    def test_log_test_step_without_details(
        self, mock_get_logger: MagicMock
    ) -> None:
        """Test log_test_step method without details."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mixin = LoggingMixin()
        mixin.log_test_step("test step")

        mock_logger.info.assert_called_once_with("TEST STEP: test step")

    @patch("tests.e2e.base_test.logging.getLogger")
    def test_log_assertion_passed(self, mock_get_logger: MagicMock) -> None:
        """Test log_assertion method for passed assertions."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mixin = LoggingMixin()
        mixin.log_assertion("test assertion", True, "details")

        mock_logger.info.assert_called_once_with(
            "ASSERTION PASSED: test assertion - details"
        )

    @patch("tests.e2e.base_test.logging.getLogger")
    def test_log_assertion_failed(self, mock_get_logger: MagicMock) -> None:
        """Test log_assertion method for failed assertions."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mixin = LoggingMixin()
        mixin.log_assertion("test assertion", False, "failure details")

        mock_logger.error.assert_called_once_with(
            "ASSERTION FAILED: test assertion - failure details"
        )

    @patch("tests.e2e.base_test.logging.getLogger")
    def test_log_performance_metric(self, mock_get_logger: MagicMock) -> None:
        """Test log_performance_metric method."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mixin = LoggingMixin()
        mixin.log_performance_metric("load_time", 2.5, "seconds")

        mock_logger.info.assert_called_once_with(
            "PERFORMANCE: load_time = 2.500 seconds"
        )

    @patch("tests.e2e.base_test.logging.getLogger")
    def test_log_browser_info_success(
        self, mock_get_logger: MagicMock
    ) -> None:
        """Test log_browser_info method with successful retrieval."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_driver = MagicMock(spec=WebDriver)
        mock_driver.execute_script.return_value = "Mozilla/5.0 Chrome/91.0"
        mock_driver.get_window_size.return_value = {
            "width": 1920,
            "height": 1080,
        }

        mixin = LoggingMixin()
        mixin.log_browser_info(mock_driver)

        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call(
            "BROWSER INFO: Mozilla/5.0 Chrome/91.0"
        )
        mock_logger.info.assert_any_call("WINDOW SIZE: 1920x1080")

    @patch("tests.e2e.base_test.logging.getLogger")
    def test_log_browser_info_failure(
        self, mock_get_logger: MagicMock
    ) -> None:
        """Test log_browser_info method when browser info retrieval fails."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_driver = MagicMock(spec=WebDriver)
        mock_driver.execute_script.side_effect = WebDriverException(
            "Connection failed"
        )

        mixin = LoggingMixin()
        mixin.log_browser_info(mock_driver)

        mock_logger.warning.assert_called_once_with(
            "Could not retrieve browser info: Connection failed"
        )


class TestRetryMixin:
    """Test suite for RetryMixin functionality."""

    def test_initialization(self) -> None:
        """Test that RetryMixin initializes with correct defaults."""
        mixin = RetryMixin()
        assert mixin._default_retry_count == 3
        assert mixin._default_retry_delay == 1.0

    def test_retry_operation_success_first_attempt(self) -> None:
        """Test retry_operation when operation succeeds on first attempt."""
        mixin = RetryMixin()

        def successful_operation() -> str:
            return "success"

        result = mixin.retry_operation(successful_operation)
        assert result == "success"

    def test_retry_operation_success_after_retries(self) -> None:
        """Test retry_operation when operation succeeds after retries."""
        mixin = RetryMixin()

        call_count = 0

        def flaky_operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise WebDriverException("Temporary failure")
            return "success"

        result = mixin.retry_operation(
            flaky_operation, max_retries=3, delay=0.1, description="flaky test"
        )
        assert result == "success"
        assert call_count == 3

    def test_retry_operation_all_attempts_fail(self) -> None:
        """Test retry_operation when all attempts fail."""
        mixin = RetryMixin()

        def failing_operation() -> str:
            raise WebDriverException("Persistent failure")

        with pytest.raises(WebDriverException, match="Persistent failure"):
            mixin.retry_operation(
                failing_operation,
                max_retries=2,
                delay=0.1,
                description="failing test",
            )

    def test_retry_operation_with_custom_exceptions(self) -> None:
        """Test retry_operation with custom exception types."""
        mixin = RetryMixin()

        def operation_with_custom_exception() -> str:
            raise ValueError("Custom error")

        # Should not catch ValueError since it's not in exceptions tuple
        with pytest.raises(ValueError, match="Custom error"):
            mixin.retry_operation(
                operation_with_custom_exception,
                max_retries=2,
                exceptions=(WebDriverException,),
                description="custom exception test",
            )

    @patch("time.sleep")
    def test_retry_operation_exponential_backoff(
        self, mock_sleep: MagicMock
    ) -> None:
        """Test that retry_operation implements exponential backoff."""
        mixin = RetryMixin()

        call_count = 0

        def flaky_operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise WebDriverException("Failure")
            return "success"

        mixin.retry_operation(
            flaky_operation,
            max_retries=3,
            delay=1.0,
            description="backoff test",
        )

        # Check that sleep was called with exponentially increasing delays
        expected_delays = [1.0, 1.5]  # First retry: 1.0, second retry: 1.5
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    def test_wait_with_retry_success(self) -> None:
        """Test wait_with_retry when condition succeeds."""
        mixin = RetryMixin()

        mock_driver = MagicMock(spec=WebDriver)

        def success_condition(driver: WebDriver) -> str:
            return "element found"

        with patch("tests.e2e.base_test.WebDriverWait") as mock_wait_class:
            mock_wait = MagicMock()
            mock_wait_class.return_value = mock_wait
            mock_wait.until.return_value = "element found"

            result = mixin.wait_with_retry(
                mock_driver,
                success_condition,
                timeout=10.0,
                description="element wait",
            )

            assert result == "element found"
            mock_wait_class.assert_called_with(mock_driver, 10.0)
            mock_wait.until.assert_called_once()


class TestStreamlitAssertionMixin:
    """Test suite for StreamlitAssertionMixin functionality."""

    def test_initialization(self) -> None:
        """Test that StreamlitAssertionMixin initializes correctly."""
        mixin = StreamlitAssertionMixin()
        assert isinstance(mixin, StreamlitAssertionMixin)

    @patch("tests.e2e.base_test.WebDriverWait")
    def test_assert_streamlit_loaded_success(
        self, mock_wait_class: MagicMock
    ) -> None:
        """Test assert_streamlit_loaded when Streamlit loads successfully."""
        mixin = StreamlitAssertionMixin()
        mock_driver = MagicMock(spec=WebDriver)

        mock_wait = MagicMock()
        mock_wait_class.return_value = mock_wait

        mock_element = MagicMock()
        mock_element.is_displayed.return_value = True
        mock_wait.until.return_value = mock_element

        # Should not raise any exception
        mixin.assert_streamlit_loaded(mock_driver, timeout=30.0)

        mock_wait_class.assert_called_once_with(mock_driver, 30.0)
        mock_wait.until.assert_called_once()

    @patch("tests.e2e.base_test.WebDriverWait")
    def test_assert_streamlit_loaded_timeout(
        self, mock_wait_class: MagicMock
    ) -> None:
        """Test assert_streamlit_loaded when timeout occurs."""
        mixin = StreamlitAssertionMixin()
        mock_driver = MagicMock(spec=WebDriver)

        mock_wait = MagicMock()
        mock_wait_class.return_value = mock_wait
        mock_wait.until.side_effect = TimeoutException(
            "Timeout waiting for element"
        )

        with pytest.raises(
            AssertionError, match="Streamlit failed to load within 30 seconds"
        ):
            mixin.assert_streamlit_loaded(mock_driver, timeout=30.0)

    @patch("tests.e2e.base_test.WebDriverWait")
    def test_assert_sidebar_present_success(
        self, mock_wait_class: MagicMock
    ) -> None:
        """Test assert_sidebar_present when sidebar is found."""
        mixin = StreamlitAssertionMixin()
        mock_driver = MagicMock(spec=WebDriver)

        mock_wait = MagicMock()
        mock_wait_class.return_value = mock_wait

        mock_sidebar = MagicMock()
        mock_sidebar.is_displayed.return_value = True
        mock_wait.until.return_value = mock_sidebar

        # Should not raise any exception
        mixin.assert_sidebar_present(mock_driver, timeout=10.0)

    @patch("tests.e2e.base_test.WebDriverWait")
    def test_assert_element_text_contains_success(
        self, mock_wait_class: MagicMock
    ) -> None:
        """Test assert_element_text_contains when text is found."""
        mixin = StreamlitAssertionMixin()
        mock_driver = MagicMock(spec=WebDriver)

        mock_wait = MagicMock()
        mock_wait_class.return_value = mock_wait

        mock_element = MagicMock()
        mock_element.text = "This is test content"
        mock_wait.until.return_value = mock_element

        # Should not raise any exception
        mixin.assert_element_text_contains(
            mock_driver,
            "[data-testid='test-element']",
            "test content",
            case_sensitive=False,
        )

    def test_assert_no_streamlit_errors_no_errors(self) -> None:
        """Test assert_no_streamlit_errors when no errors are present."""
        mixin = StreamlitAssertionMixin()
        mock_driver = MagicMock(spec=WebDriver)

        # Mock that no error elements are found
        mock_driver.find_elements.return_value = []

        # Should not raise any exception
        mixin.assert_no_streamlit_errors(mock_driver)

    def test_assert_no_streamlit_errors_with_errors(self) -> None:
        """Test assert_no_streamlit_errors when errors are present."""
        mixin = StreamlitAssertionMixin()
        mock_driver = MagicMock(spec=WebDriver)

        # Mock error elements
        mock_error = MagicMock()
        mock_error.is_displayed.return_value = True
        mock_error.text = "Connection error"

        mock_driver.find_elements.return_value = [mock_error]

        with pytest.raises(
            AssertionError, match="Streamlit errors found: Connection error"
        ):
            mixin.assert_no_streamlit_errors(mock_driver)


class TestStreamlitUtilityMixin:
    """Test suite for StreamlitUtilityMixin functionality."""

    def test_initialization(self) -> None:
        """Test that StreamlitUtilityMixin initializes correctly."""
        mixin = StreamlitUtilityMixin()
        assert isinstance(mixin, StreamlitUtilityMixin)

    @patch("tests.e2e.base_test.WebDriverWait")
    def test_wait_for_streamlit_rerun_no_spinners(
        self, mock_wait_class: MagicMock
    ) -> None:
        """Test wait_for_streamlit_rerun when no spinners are present."""
        mixin = StreamlitUtilityMixin()
        mock_driver = MagicMock(spec=WebDriver)

        mock_wait = MagicMock()
        mock_wait_class.return_value = mock_wait
        mock_wait.until.side_effect = TimeoutException("No spinners found")

        # Should not raise exception when no spinners found
        mixin.wait_for_streamlit_rerun(mock_driver, timeout=10.0)

    def test_capture_streamlit_state(self) -> None:
        """Test capture_streamlit_state method."""
        mixin = StreamlitUtilityMixin()
        mock_driver = MagicMock(spec=WebDriver)

        mock_driver.current_url = "http://localhost:8501"
        mock_driver.title = "CrackSeg App"
        mock_driver.execute_script.return_value = "complete"
        mock_driver.find_elements.return_value = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        state = mixin.capture_streamlit_state(mock_driver)

        assert state["url"] == "http://localhost:8501"
        assert state["title"] == "CrackSeg App"
        assert state["ready_state"] == "complete"
        assert state["visible_elements"] == 3
        assert "timestamp" in state

    @patch("tests.e2e.base_test.WebDriverWait")
    def test_click_streamlit_button_success(
        self, mock_wait_class: MagicMock
    ) -> None:
        """Test click_streamlit_button when button is found and clicked."""
        mixin = StreamlitUtilityMixin()
        mock_driver = MagicMock(spec=WebDriver)

        mock_wait = MagicMock()
        mock_wait_class.return_value = mock_wait

        mock_button = MagicMock()
        mock_wait.until.return_value = mock_button

        # Mock wait_for_streamlit_rerun method
        with patch.object(mixin, "wait_for_streamlit_rerun") as mock_rerun:
            mixin.click_streamlit_button(
                mock_driver, "Submit", timeout=10.0, wait_for_rerun=True
            )

            mock_button.click.assert_called_once()
            mock_rerun.assert_called_once_with(mock_driver, 10.0)


class TestBaseE2ETest:
    """Test suite for BaseE2ETest abstract base class."""

    def test_concrete_implementation_initialization(self) -> None:
        """Test that concrete implementation can be instantiated."""
        test_instance = ConcreteTestClass()
        assert isinstance(test_instance, BaseE2ETest)
        assert hasattr(test_instance, "_test_data")
        assert hasattr(test_instance, "_config")

    def test_setup_test_data_implementation(self) -> None:
        """Test that setup_test_data is properly implemented."""
        test_instance = ConcreteTestClass()
        test_data = test_instance.setup_test_data()

        assert isinstance(test_data, dict)
        assert "test_url" in test_data
        assert "expected_title" in test_data
        assert "timeout" in test_data

    def test_get_test_data_caching(self) -> None:
        """Test that get_test_data caches results."""
        test_instance = ConcreteTestClass()

        # First call should initialize data
        data1 = test_instance.get_test_data()
        data2 = test_instance.get_test_data()

        # Should be the same object (cached)
        assert data1 is data2

    def test_config_management(self) -> None:
        """Test config setter and getter methods."""
        test_instance = ConcreteTestClass()

        # Initially should be None
        assert test_instance.get_config() is None

        # Set and get config
        config = DriverConfig(browser="chrome", headless=True)
        test_instance.set_config(config)

        retrieved_config = test_instance.get_config()
        assert retrieved_config is config

    @patch("time.time")
    def test_navigate_and_verify(self, mock_time: MagicMock) -> None:
        """Test navigate_and_verify method."""
        test_instance = ConcreteTestClass()
        mock_driver = MagicMock(spec=WebDriver)

        # Mock time for performance measurement
        mock_time.side_effect = [1000.0, 1002.5]  # 2.5 second load time

        # Mock the assertion methods
        with (
            patch.object(
                test_instance, "assert_page_ready_state"
            ) as mock_ready,
            patch.object(
                test_instance, "assert_streamlit_loaded"
            ) as mock_loaded,
            patch.object(
                test_instance, "log_performance_metric"
            ) as mock_metric,
            patch.object(test_instance, "log_browser_info") as mock_info,
            patch.object(test_instance, "log_test_step") as mock_step,
        ):
            test_instance.navigate_and_verify(
                mock_driver,
                "http://localhost:8501",
                verify_load=True,
                timeout=30.0,
            )

            mock_driver.get.assert_called_once_with("http://localhost:8501")
            mock_ready.assert_called_once_with(mock_driver, 30.0)
            mock_loaded.assert_called_once_with(mock_driver, 30.0)
            mock_metric.assert_called_once_with("page_load_time", 2.5)
            mock_info.assert_called_once_with(mock_driver)
            mock_step.assert_called_once_with(
                "Navigating to URL: http://localhost:8501"
            )

    def test_take_screenshot_on_failure_success(self) -> None:
        """Test take_screenshot_on_failure when screenshot succeeds."""
        test_instance = ConcreteTestClass()
        mock_driver = MagicMock(spec=WebDriver)

        with (
            patch("pathlib.Path.mkdir"),
            patch("time.time", return_value=1234567890),
            patch.object(test_instance, "log_test_step") as mock_log,
        ):
            mock_driver.save_screenshot.return_value = True

            result = test_instance.take_screenshot_on_failure(
                mock_driver, "test_failure", Path("test-artifacts")
            )

            expected_path = (
                Path("test-artifacts") / "test_failure_failure_1234567890.png"
            )
            assert result == expected_path
            mock_driver.save_screenshot.assert_called_once_with(
                str(expected_path)
            )
            mock_log.assert_called_once()

    def test_setup_and_teardown_methods(self) -> None:
        """Test setup_method and teardown_method."""
        test_instance = ConcreteTestClass()

        # Set some test data to verify reset
        test_instance._test_data = {"existing": "data"}

        mock_method = MagicMock()
        mock_method.__name__ = "test_example"

        with patch.object(test_instance, "log_test_step") as mock_log:
            test_instance.setup_method(mock_method)
            assert test_instance._test_data == {}  # Should be reset
            mock_log.assert_called_with("Starting test: test_example")

            test_instance.teardown_method(mock_method)
            mock_log.assert_called_with("Completed test: test_example")

    def test_mixin_integration(self) -> None:
        """Test that all mixins are properly integrated."""
        test_instance = ConcreteTestClass()

        # Verify that all mixin methods are available
        assert hasattr(test_instance, "log_test_step")  # LoggingMixin
        assert hasattr(test_instance, "retry_operation")  # RetryMixin
        assert hasattr(
            test_instance, "assert_streamlit_loaded"
        )  # StreamlitAssertionMixin
        assert hasattr(
            test_instance, "wait_for_streamlit_rerun"
        )  # StreamlitUtilityMixin

        # Verify method resolution order is correct
        mro_classes = [cls.__name__ for cls in ConcreteTestClass.__mro__]
        expected_order = [
            "ConcreteTestClass",
            "BaseE2ETest",
            "LoggingMixin",
            "RetryMixin",
            "StreamlitAssertionMixin",
            "StreamlitUtilityMixin",
            "ABC",
            "object",
        ]

        for expected_class in expected_order:
            assert expected_class in mro_classes

    def test_abstract_method_enforcement(self) -> None:
        """Test that abstract methods are properly enforced."""
        # Attempting to instantiate BaseE2ETest directly should fail
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class"
        ):
            BaseE2ETest()  # type: ignore[abstract]
