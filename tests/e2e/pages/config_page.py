"""Config page object for the CrackSeg Streamlit application.

This module provides the ConfigPage class that handles interactions with
the configuration management page, including loading configurations,
validating settings, and managing configuration files.
"""

from selenium.webdriver.common.by import By

from .base_page import BasePage
from .locators import ConfigPageLocators


class ConfigPage(BasePage):
    """Page object for the Configuration page.

    Handles configuration loading, validation, and file management
    operations on the Config page of the CrackSeg application.
    """

    @property
    def page_name(self) -> str:
        """Return the name of the page for navigation."""
        return "Config"

    @property
    def page_title_locator(self) -> tuple[str, str]:
        """Return the locator for the page title element."""
        return ConfigPageLocators.PAGE_TITLE

    @property
    def expected_url_fragment(self) -> str | None:
        """Return expected URL fragment for this page."""
        return None  # Streamlit doesn't use URL fragments for page routing

    def navigate_to_page(self, wait_for_load: bool = True) -> "ConfigPage":
        """Navigate to the config page and wait for it to be ready.

        Overrides the base navigation to add a specific wait for the
        configuration file selector, ensuring the page is fully interactive.
        """
        super().navigate_to_page(wait_for_load)
        self.wait_for_element(
            ConfigPageLocators.CONFIG_FILE_SELECTOR, visible=True
        )
        return self

    def select_config_file(self, config_name: str) -> "ConfigPage":
        """Select a configuration file from the dropdown.

        Args:
            config_name: The name of the configuration file to select.

        Returns:
            Self for method chaining.
        """
        # Click the selectbox to open the dropdown
        selector = self.wait_for_element(
            ConfigPageLocators.CONFIG_FILE_SELECTOR, visible=True
        )
        if selector:
            selector.click()
            # Wait for dropdown options to appear
            self.wait_for_element(
                ConfigPageLocators.selectbox_option(config_name), visible=True
            )
            # Click the desired option
            option = self.wait_for_element(
                ConfigPageLocators.selectbox_option(config_name), visible=True
            )
            if option:
                option.click()
                # Wait for Streamlit to process the selection
                from ..utils import wait_for_streamlit_rerun

                wait_for_streamlit_rerun(self.driver, 15)
        return self

    def click_load_config(self) -> "ConfigPage":
        """Clicks the 'Load Configuration' button."""
        self.click_element(
            ConfigPageLocators.LOAD_CONFIG_BUTTON, wait_for_rerun=True
        )
        return self

    def load_configuration_file(
        self, config_name: str | None = None
    ) -> "ConfigPage":
        """Load a configuration file from the available options.

        Args:
            config_name: Name of configuration to load. If None, loads first
                available.

        Returns:
            Self for method chaining

        Raises:
            WebDriverException: If configuration loading fails
        """
        # This method is now a higher-level workflow
        if config_name:
            self.select_config_file(config_name)

        self.click_load_config()
        self.wait_for_configuration_loaded()
        return self

    def upload_configuration_file(self, file_path: str) -> "ConfigPage":
        """Upload a configuration file using the file uploader.

        Args:
            file_path: Path to the configuration file to upload

        Returns:
            Self for method chaining

        Raises:
            WebDriverException: If file upload fails
        """
        uploader = self.wait_for_element(ConfigPageLocators.CONFIG_UPLOAD)
        if uploader:
            # Send file path to the input element
            file_input = uploader.find_element_by_css_selector(
                "input[type='file']"
            )
            file_input.send_keys(file_path)

            # Wait for upload to complete and Streamlit to process
            self.wait_for_no_spinners()

        return self

    def validate_configuration(self) -> bool:
        """Validate the currently loaded configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        if self.is_element_displayed(
            ConfigPageLocators.VALIDATE_CONFIG_BUTTON
        ):
            self.click_element(
                ConfigPageLocators.VALIDATE_CONFIG_BUTTON, wait_for_rerun=True
            )

            # Wait for validation result
            self.wait_for_no_spinners()

            # Check for success/error messages
            # This is a simplified check - actual implementation might need
            # more sophisticated result parsing
            from .locators import BaseLocators

            return self.is_element_displayed(BaseLocators.SUCCESS_MESSAGE)

        return False

    def save_configuration(self) -> bool:
        """Save the current configuration.

        Returns:
            True if save was successful, False otherwise
        """
        if self.is_element_displayed(ConfigPageLocators.SAVE_CONFIG_BUTTON):
            self.click_element(
                ConfigPageLocators.SAVE_CONFIG_BUTTON, wait_for_rerun=True
            )

            # Wait for save operation to complete
            self.wait_for_no_spinners()

            # Check for success message
            from .locators import BaseLocators

            return self.is_element_displayed(BaseLocators.SUCCESS_MESSAGE)

        return False

    def get_configuration_content(self) -> str | None:
        """Get the content of the currently displayed configuration.

        Returns:
            Configuration content as string, or None if not available
        """
        # Try to expand configuration viewer if it's collapsed
        expander = self.wait_for_element(
            ConfigPageLocators.CONFIG_VIEWER, visible=False
        )
        if expander:
            # Check if it is already expanded
            aria_expanded = expander.get_attribute("aria-expanded")
            if aria_expanded == "false":
                expander.click()
                self.wait_for_element(
                    ConfigPageLocators.CONFIG_CONTENT, visible=True
                )

        # Get configuration content
        return self.get_element_text(ConfigPageLocators.CONFIG_CONTENT)

    def is_configuration_loaded(self) -> bool:
        """Check if a configuration is currently loaded.

        Returns:
            True if configuration is loaded, False otherwise
        """
        return self.is_element_displayed(
            ConfigPageLocators.CONFIG_VIEWER
        ) or self.is_element_displayed(ConfigPageLocators.CONFIG_CONTENT)

    def expand_configuration_viewer(self) -> "ConfigPage":
        """Expand the configuration viewer if it's collapsed.

        Returns:
            Self for method chaining
        """
        if self.is_element_displayed(ConfigPageLocators.CONFIG_VIEWER):
            self.click_element(ConfigPageLocators.CONFIG_VIEWER)

        return self

    def wait_for_configuration_loaded(
        self, timeout: float | None = None
    ) -> "ConfigPage":
        """Wait for a configuration to be loaded and displayed.

        Args:
            timeout: Custom timeout for this operation

        Returns:
            Self for method chaining

        Raises:
            TimeoutException: If configuration doesn't load within timeout
        """
        timeout = timeout or self.timeout

        # Wait for either config viewer or content to be present
        self.wait_for_element(ConfigPageLocators.CONFIG_VIEWER, timeout)

        return self

    def get_available_configurations(self) -> list[str]:
        """Get list of available configuration options.

        Returns:
            List of configuration names/options
        """
        # This is a placeholder implementation
        # Actual implementation would depend on how Streamlit selectbox
        # exposes its options for Selenium interaction
        configurations = []

        try:
            selector = self.wait_for_element(
                ConfigPageLocators.CONFIG_FILE_SELECTOR
            )
            if selector:
                # Extract options from selectbox
                # Implementation depends on Streamlit's DOM structure
                options = selector.find_elements_by_tag_name("option")
                configurations = [option.text for option in options]

        except Exception:
            # Return empty list if extraction fails
            pass

        return configurations

    def clear_configuration(self) -> "ConfigPage":
        """Clear the currently loaded configuration.

        Returns:
            Self for method chaining
        """
        # This might involve clicking a clear button or selecting "None" option
        # Implementation depends on the actual UI elements available

        # Placeholder implementation
        if self.is_element_displayed(ConfigPageLocators.CONFIG_FILE_SELECTOR):
            self.click_element(ConfigPageLocators.CONFIG_FILE_SELECTOR)

        return self

    def validate_required_fields(self) -> bool:
        """Validate that all required configuration fields are present.

        Returns:
            True if all required fields are valid, False otherwise
        """
        # This is a domain-specific validation for CrackSeg configurations
        # Check for presence of key configuration elements

        config_content = self.get_configuration_content()
        if not config_content:
            return False

        # Basic validation for CrackSeg config structure
        required_sections = ["model", "training", "data"]
        return all(section in config_content for section in required_sections)

    def load_configuration_text(self, config_text: str) -> "ConfigPage":
        """Load configuration from text content.

        Args:
            config_text: YAML configuration content as string

        Returns:
            Self for method chaining
        """
        # This is a mock implementation for testing purposes
        # In a real implementation, this would interface with Streamlit's
        # text area or code editor components

        # Store the text for validation and display
        self._loaded_config_text = config_text

        # Simulate loading process
        from ..utils import wait_for_streamlit_rerun

        wait_for_streamlit_rerun(self.driver, 5)

        return self

    def is_config_loaded(self) -> bool:
        """Check if configuration text has been loaded.

        Returns:
            True if configuration text is loaded, False otherwise
        """
        # Check if we have loaded text or if the standard configuration
        # is loaded
        has_loaded_text = (
            hasattr(self, "_loaded_config_text") and self._loaded_config_text
        )
        has_standard_config = self.is_configuration_loaded()

        return bool(has_loaded_text) or bool(has_standard_config)

    def has_error_message(self) -> bool:
        """Check if there are error messages displayed.

        Returns:
            True if error messages are present, False otherwise
        """
        from .locators import BaseLocators

        # Check for various error indicators
        error_selectors = [
            BaseLocators.ERROR_MESSAGE,
            (By.CSS_SELECTOR, "[data-testid='stAlert']"),
            (By.CSS_SELECTOR, ".stException"),
            (By.XPATH, "//*[contains(@class, 'error')]"),
            (By.XPATH, "//*[contains(text(), 'Error')]"),
            (By.XPATH, "//*[contains(text(), 'error')]"),
            (By.XPATH, "//*[contains(text(), 'Invalid')]"),
            (By.XPATH, "//*[contains(text(), 'invalid')]"),
        ]

        for selector in error_selectors:
            if self.is_element_displayed(selector):
                return True

        return False
