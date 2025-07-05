"""Unit tests for wait conditions module."""

from unittest.mock import Mock, patch

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from tests.e2e.waits.conditions import (
    CustomConditions,
    StreamlitConditions,
    element_attribute_contains,
    element_count_equals,
    element_text_matches,
    text_to_be_present_in_element_value,
)


class TestElementTextMatches:
    """Test element_text_matches condition function."""

    def test_text_matches_simple_string(self) -> None:
        """Test simple string matching."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.text = "Hello World"
        driver.find_element.return_value = element

        locator = (By.ID, "test-element")
        condition = element_text_matches(locator, "Hello")

        result = condition(driver)
        assert result == element
        driver.find_element.assert_called_once_with(*locator)

    def test_text_matches_regex_pattern(self) -> None:
        """Test regex pattern matching."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.text = "Error: File not found"
        driver.find_element.return_value = element

        locator = (By.CLASS_NAME, "error-message")
        condition = element_text_matches(locator, r"Error: .+", regex=True)

        result = condition(driver)
        assert result == element

    def test_text_no_match(self) -> None:
        """Test when text doesn't match."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.text = "Different text"
        driver.find_element.return_value = element

        locator = (By.ID, "test-element")
        condition = element_text_matches(locator, "Expected")

        result = condition(driver)
        assert result is False

    def test_element_not_found(self) -> None:
        """Test when element is not found."""
        driver = Mock(spec=WebDriver)
        driver.find_element.side_effect = NoSuchElementException()

        locator = (By.ID, "missing-element")
        condition = element_text_matches(locator, "text")

        result = condition(driver)
        assert result is False

    def test_stale_element_reference(self) -> None:
        """Test handling stale element reference."""
        driver = Mock(spec=WebDriver)
        driver.find_element.side_effect = StaleElementReferenceException()

        locator = (By.ID, "stale-element")
        condition = element_text_matches(locator, "text")

        result = condition(driver)
        assert result is False


class TestElementCountEquals:
    """Test element_count_equals condition function."""

    def test_count_matches(self) -> None:
        """Test when element count matches expected."""
        driver = Mock(spec=WebDriver)
        elements = [Mock(spec=WebElement) for _ in range(3)]
        driver.find_elements.return_value = elements

        locator = (By.CLASS_NAME, "list-item")
        condition = element_count_equals(locator, 3)

        result = condition(driver)
        assert result == elements

    def test_count_mismatch(self) -> None:
        """Test when element count doesn't match."""
        driver = Mock(spec=WebDriver)
        elements = [Mock(spec=WebElement) for _ in range(2)]
        driver.find_elements.return_value = elements

        locator = (By.CLASS_NAME, "list-item")
        condition = element_count_equals(locator, 3)

        result = condition(driver)
        assert result is False

    def test_webdriver_exception(self) -> None:
        """Test handling WebDriver exception."""
        driver = Mock(spec=WebDriver)
        driver.find_elements.side_effect = WebDriverException()

        locator = (By.CLASS_NAME, "list-item")
        condition = element_count_equals(locator, 3)

        result = condition(driver)
        assert result is False


class TestElementAttributeContains:
    """Test element_attribute_contains condition function."""

    def test_attribute_contains_value(self) -> None:
        """Test when attribute contains expected value."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.get_attribute.return_value = "btn btn-primary active"
        driver.find_element.return_value = element

        locator = (By.ID, "submit-btn")
        condition = element_attribute_contains(locator, "class", "active")

        result = condition(driver)
        assert result == element
        element.get_attribute.assert_called_once_with("class")

    def test_attribute_not_contains_value(self) -> None:
        """Test when attribute doesn't contain value."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.get_attribute.return_value = "btn btn-primary"
        driver.find_element.return_value = element

        locator = (By.ID, "submit-btn")
        condition = element_attribute_contains(locator, "class", "active")

        result = condition(driver)
        assert result is False

    def test_attribute_is_none(self) -> None:
        """Test when attribute is None."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.get_attribute.return_value = None
        driver.find_element.return_value = element

        locator = (By.ID, "submit-btn")
        condition = element_attribute_contains(locator, "data-value", "test")

        result = condition(driver)
        assert result is False


class TestTextToBePresentInElementValue:
    """Test text_to_be_present_in_element_value condition function."""

    def test_text_present_in_value(self) -> None:
        """Test when text is present in element value."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.get_attribute.return_value = "user@example.com"
        driver.find_element.return_value = element

        locator = (By.ID, "email-input")
        condition = text_to_be_present_in_element_value(locator, "example")

        result = condition(driver)
        assert result is True

    def test_text_not_present_in_value(self) -> None:
        """Test when text is not present in element value."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.get_attribute.return_value = "user@test.com"
        driver.find_element.return_value = element

        locator = (By.ID, "email-input")
        condition = text_to_be_present_in_element_value(locator, "example")

        result = condition(driver)
        assert result is False

    def test_value_is_none(self) -> None:
        """Test when element value is None."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.get_attribute.return_value = None
        driver.find_element.return_value = element

        locator = (By.ID, "email-input")
        condition = text_to_be_present_in_element_value(locator, "test")

        result = condition(driver)
        assert result is False


class TestCustomConditions:
    """Test CustomConditions factory class."""

    def test_element_to_be_stale_when_stale(self) -> None:
        """Test element_to_be_stale when element is stale."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.is_enabled.side_effect = StaleElementReferenceException()

        condition = CustomConditions.element_to_be_stale(element)
        result = condition(driver)

        assert result is True

    def test_element_to_be_stale_when_not_stale(self) -> None:
        """Test element_to_be_stale when element is not stale."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.is_enabled.return_value = True

        condition = CustomConditions.element_to_be_stale(element)
        result = condition(driver)

        assert result is False

    def test_element_attribute_to_be_updated(self) -> None:
        """Test element_attribute_to_be_updated condition."""
        driver = Mock(spec=WebDriver)
        element = Mock(spec=WebElement)
        element.get_attribute.return_value = "new-value"
        driver.find_element.return_value = element

        locator = (By.ID, "status")
        condition = CustomConditions.element_attribute_to_be_updated(
            locator, "data-status", "old-value"
        )

        result = condition(driver)
        assert result == element

    def test_page_title_matches_simple(self) -> None:
        """Test page_title_matches with simple string."""
        driver = Mock(spec=WebDriver)
        driver.title = "CrackSeg - Configuration Page"

        condition = CustomConditions.page_title_matches("Configuration")
        result = condition(driver)

        assert result is True

    def test_page_title_matches_regex(self) -> None:
        """Test page_title_matches with regex pattern."""
        driver = Mock(spec=WebDriver)
        driver.title = "CrackSeg - Training Page"

        condition = CustomConditions.page_title_matches(
            r"CrackSeg - \w+ Page", regex=True
        )
        result = condition(driver)

        assert result is True


class TestStreamlitConditions:
    """Test StreamlitConditions factory class."""

    def test_app_ready_basic(self) -> None:
        """Test app_ready condition with basic elements."""
        driver = Mock(spec=WebDriver)

        # Mock successful app structure check
        app_element = Mock(spec=WebElement)
        driver.find_element.return_value = app_element

        # Mock no spinners
        driver.find_elements.return_value = []

        condition = StreamlitConditions.app_ready(check_sidebar=False)
        result = condition(driver)

        assert result is True

    def test_app_ready_with_sidebar(self) -> None:
        """Test app_ready condition including sidebar check."""
        driver = Mock(spec=WebDriver)

        # Mock app element present and sidebar present
        app_element = Mock(spec=WebElement)
        sidebar_element = Mock(spec=WebElement)

        def mock_find_element(by: str, selector: str) -> Mock:
            if "stApp" in selector:
                return app_element
            return sidebar_element

        def mock_find_elements(by: str, selector: str) -> list[Mock]:
            if "stSpinner" in selector:
                return []  # No spinners
            elif "stSidebar" in selector:
                return [sidebar_element]  # Sidebar present
            return []

        driver.find_element = mock_find_element
        driver.find_elements = mock_find_elements

        condition = StreamlitConditions.app_ready(check_sidebar=True)
        result = condition(driver)

        assert result is True

    def test_app_ready_with_spinners(self) -> None:
        """Test app_ready when spinners are present."""
        driver = Mock(spec=WebDriver)

        app_element = Mock(spec=WebElement)
        spinner_element = Mock(spec=WebElement)

        driver.find_element.return_value = app_element
        driver.find_elements.return_value = [
            spinner_element
        ]  # Spinners present

        condition = StreamlitConditions.app_ready(check_sidebar=False)
        result = condition(driver)

        assert result is False

    def test_sidebar_loaded(self) -> None:
        """Test sidebar_loaded condition."""
        driver = Mock(spec=WebDriver)
        sidebar = Mock(spec=WebElement)
        sidebar.is_displayed.return_value = True
        driver.find_element.return_value = sidebar

        condition = StreamlitConditions.sidebar_loaded()
        result = condition(driver)

        assert result is True

    def test_no_spinners_present(self) -> None:
        """Test no_spinners_present condition."""
        driver = Mock(spec=WebDriver)
        driver.find_elements.return_value = []  # No spinners

        condition = StreamlitConditions.no_spinners_present()
        result = condition(driver)

        assert result is True

    def test_file_upload_complete_general(self) -> None:
        """Test file_upload_complete condition general case."""
        driver = Mock(spec=WebDriver)

        def mock_find_elements(by: str, selector: str) -> list[Mock]:
            if "stProgress" in selector:
                return []  # No progress indicators
            elif "stFileUploader" in selector:
                uploader = Mock()
                uploader.find_elements.return_value = (
                    []
                )  # No progress in uploaders
                return [uploader]
            return []

        driver.find_elements = mock_find_elements

        condition = StreamlitConditions.file_upload_complete()
        result = condition(driver)

        assert result is True

    def test_rerun_complete(self) -> None:
        """Test rerun_complete condition."""
        driver = Mock(spec=WebDriver)
        driver.find_elements.return_value = []  # No spinners

        condition = StreamlitConditions.rerun_complete(
            max_wait_for_spinner=0.1
        )
        result = condition(driver)

        assert result is True

    @patch("time.sleep")
    def test_session_state_contains(self, mock_sleep: Mock) -> None:
        """Test session_state_contains condition."""
        driver = Mock(spec=WebDriver)
        driver.execute_script.return_value = True

        condition = StreamlitConditions.session_state_contains("test_key")
        result = condition(driver)

        assert result is True

    def test_session_state_contains_with_value(self) -> None:
        """Test session_state_contains with specific value."""
        driver = Mock(spec=WebDriver)
        driver.execute_script.return_value = True

        condition = StreamlitConditions.session_state_contains(
            "test_key", "test_value"
        )
        result = condition(driver)

        assert result is True
