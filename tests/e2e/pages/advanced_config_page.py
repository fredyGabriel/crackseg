"""Advanced Config page object for the CrackSeg Streamlit application.

This module provides the AdvancedConfigPage class that handles interactions
with
the advanced YAML configuration editor, including template management, live
validation, and advanced configuration editing.
"""

from .base_page import BasePage
from .locators import AdvancedConfigPageLocators


class AdvancedConfigPage(BasePage):
    """Page object for the Advanced Configuration page.

    Handles advanced YAML editing, template management, and live validation
    operations on the Advanced Config page of the CrackSeg application.
    """

    @property
    def page_name(self) -> str:
        """Return the name of the page for navigation."""
        return "Advanced Config"

    @property
    def page_title_locator(self) -> tuple[str, str]:
        """Return the locator for the page title element."""
        return AdvancedConfigPageLocators.PAGE_TITLE

    @property
    def expected_url_fragment(self) -> str | None:
        """Return expected URL fragment for this page."""
        return None  # Streamlit doesn't use URL fragments for page routing

    def get_yaml_content(self) -> str | None:
        """Get the current YAML content from the editor.

        Returns:
            YAML content as string, or None if not available
        """
        return self.get_element_text(AdvancedConfigPageLocators.YAML_EDITOR)

    def set_yaml_content(self, yaml_content: str) -> "AdvancedConfigPage":
        """Set YAML content in the editor.

        Args:
            yaml_content: YAML content to set in the editor

        Returns:
            Self for method chaining

        Raises:
            WebDriverException: If setting content fails
        """
        editor = self.wait_for_element(AdvancedConfigPageLocators.YAML_EDITOR)
        if editor:
            # Clear existing content and set new content
            editor.clear()
            editor.send_keys(yaml_content)

        return self

    def save_yaml_configuration(self) -> bool:
        """Save the current YAML configuration.

        Returns:
            True if save was successful, False otherwise
        """
        if self.is_element_displayed(
            AdvancedConfigPageLocators.SAVE_YAML_BUTTON
        ):
            self.click_element(
                AdvancedConfigPageLocators.SAVE_YAML_BUTTON,
                wait_for_rerun=True,
            )

            # Wait for save operation to complete
            self.wait_for_no_spinners()

            # Check for success message
            return self.is_element_displayed(
                AdvancedConfigPageLocators.VALIDATION_SUCCESS
            )

        return False

    def validate_yaml_configuration(self) -> bool:
        """Validate the current YAML configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        if self.is_element_displayed(
            AdvancedConfigPageLocators.VALIDATE_YAML_BUTTON
        ):
            self.click_element(
                AdvancedConfigPageLocators.VALIDATE_YAML_BUTTON,
                wait_for_rerun=True,
            )

            # Wait for validation to complete
            self.wait_for_no_spinners()

            # Check validation result
            return self.is_element_displayed(
                AdvancedConfigPageLocators.VALIDATION_SUCCESS
            )

        return False

    def has_validation_errors(self) -> bool:
        """Check if there are current validation errors.

        Returns:
            True if validation errors are present, False otherwise
        """
        return self.is_element_displayed(
            AdvancedConfigPageLocators.VALIDATION_ERROR
        )

    def get_validation_error_message(self) -> str | None:
        """Get validation error message if present.

        Returns:
            Error message text, or None if no error
        """
        return self.get_element_text(
            AdvancedConfigPageLocators.VALIDATION_ERROR
        )

    def load_template(
        self, template_name: str | None = None
    ) -> "AdvancedConfigPage":
        """Load a configuration template.

        Args:
            template_name: Name of template to load. If None, loads first
                available.

        Returns:
            Self for method chaining

        Raises:
            WebDriverException: If template loading fails
        """
        # Wait for template selector to be available
        self.wait_for_element(AdvancedConfigPageLocators.TEMPLATE_SELECTOR)

        if template_name:
            # Select specific template
            selector_element = self.wait_for_element(
                AdvancedConfigPageLocators.TEMPLATE_SELECTOR
            )
            if selector_element:
                # Implementation depends on how Streamlit selectbox works
                selector_element.click()

        # Click load template button if present
        if self.is_element_displayed(
            AdvancedConfigPageLocators.LOAD_TEMPLATE_BUTTON
        ):
            self.click_element(
                AdvancedConfigPageLocators.LOAD_TEMPLATE_BUTTON,
                wait_for_rerun=True,
            )

        return self

    def is_yaml_editor_loaded(self) -> bool:
        """Check if the YAML editor is loaded and ready for use.

        Returns:
            True if editor is loaded, False otherwise
        """
        return self.is_element_displayed(
            AdvancedConfigPageLocators.YAML_EDITOR
        )

    def clear_yaml_content(self) -> "AdvancedConfigPage":
        """Clear all content from the YAML editor.

        Returns:
            Self for method chaining
        """
        editor = self.wait_for_element(AdvancedConfigPageLocators.YAML_EDITOR)
        if editor:
            editor.clear()

        return self

    def wait_for_editor_ready(
        self, timeout: float | None = None
    ) -> "AdvancedConfigPage":
        """Wait for the YAML editor to be ready for interaction.

        Args:
            timeout: Custom timeout for this operation

        Returns:
            Self for method chaining

        Raises:
            TimeoutException: If editor doesn't become ready within timeout
        """
        timeout = timeout or self.timeout

        # Wait for YAML editor to be present
        self.wait_for_element(AdvancedConfigPageLocators.YAML_EDITOR, timeout)

        return self

    def validate_yaml_syntax(self) -> bool:
        """Perform client-side YAML syntax validation.

        Returns:
            True if YAML syntax is valid, False otherwise
        """
        yaml_content = self.get_yaml_content()
        if not yaml_content:
            return False

        try:
            # Basic check - more sophisticated validation would require YAML
            # parsing
            # This is a simple heuristic check for common YAML syntax issues
            lines = yaml_content.split("\n")

            # Check for basic YAML structure
            has_content = any(line.strip() for line in lines)
            has_valid_structure = any(
                ":" in line for line in lines if line.strip()
            )

            return has_content and has_valid_structure

        except Exception:
            return False

    def get_available_templates(self) -> list[str]:
        """Get list of available configuration templates.

        Returns:
            List of template names/options
        """
        templates = []

        try:
            selector = self.wait_for_element(
                AdvancedConfigPageLocators.TEMPLATE_SELECTOR
            )
            if selector:
                # Extract options from selectbox
                options = selector.find_elements_by_tag_name("option")
                templates = [option.text for option in options]

        except Exception:
            # Return empty list if extraction fails
            pass

        return templates

    def expand_yaml_editor(self) -> "AdvancedConfigPage":
        """Expand or maximize the YAML editor for better visibility.

        Returns:
            Self for method chaining
        """
        # This would depend on the actual UI implementation
        # Placeholder for any expand/maximize functionality

        # Scroll to editor to ensure it's visible
        if self.is_element_displayed(AdvancedConfigPageLocators.YAML_EDITOR):
            self.scroll_to_element(AdvancedConfigPageLocators.YAML_EDITOR)

        return self

    def validate_advanced_config_prerequisites(self) -> bool:
        """Validate that prerequisites for advanced config editing are met.

        Returns:
            True if advanced editing can be performed, False otherwise
        """
        # Check if YAML editor is available and ready
        return self.is_yaml_editor_loaded()

    def auto_format_yaml(self) -> "AdvancedConfigPage":
        """Auto-format the YAML content if formatting feature is available.

        Returns:
            Self for method chaining
        """
        # This would trigger any auto-formatting functionality
        # if available in the UI. Placeholder implementation.

        # Could involve keyboard shortcuts or format buttons
        editor = self.wait_for_element(AdvancedConfigPageLocators.YAML_EDITOR)
        if editor:
            # Focus the editor first
            editor.click()

        return self
