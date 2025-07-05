"""Architecture page object for the CrackSeg Streamlit application.

This module provides the ArchitecturePage class that handles interactions with
the model architecture visualization page, including model instantiation,
diagram viewing, and architecture analysis.
"""

from .base_page import BasePage
from .locators import ArchitecturePageLocators


class ArchitecturePage(BasePage):
    """Page object for the Architecture page.

    Handles model architecture visualization, instantiation, and analysis
    operations on the Architecture page of the CrackSeg application.
    """

    @property
    def page_name(self) -> str:
        """Return the name of the page for navigation."""
        return "Architecture"

    @property
    def page_title_locator(self) -> tuple[str, str]:
        """Return the locator for the page title element."""
        return ArchitecturePageLocators.PAGE_TITLE

    @property
    def expected_url_fragment(self) -> str | None:
        """Return expected URL fragment for this page."""
        return None  # Streamlit doesn't use URL fragments for page routing

    def instantiate_model(self) -> "ArchitecturePage":
        """Instantiate model from the loaded configuration.

        Returns:
            Self for method chaining

        Raises:
            WebDriverException: If model instantiation fails
        """
        if self.is_element_displayed(
            ArchitecturePageLocators.INSTANTIATE_MODEL_BUTTON
        ):
            self.click_element(
                ArchitecturePageLocators.INSTANTIATE_MODEL_BUTTON,
                wait_for_rerun=True,
            )

            # Wait for model instantiation to complete
            self.wait_for_no_spinners(
                timeout=30.0
            )  # Model instantiation can take time

        return self

    def is_model_instantiated(self) -> bool:
        """Check if a model has been successfully instantiated.

        Returns:
            True if model is instantiated, False otherwise
        """
        return self.is_element_displayed(
            ArchitecturePageLocators.MODEL_SUMMARY
        ) or self.is_element_displayed(
            ArchitecturePageLocators.ARCHITECTURE_DIAGRAM
        )

    def get_model_summary(self) -> str | None:
        """Get the model summary text.

        Returns:
            Model summary as string, or None if not available
        """
        return self.get_element_text(ArchitecturePageLocators.MODEL_SUMMARY)

    def is_architecture_diagram_displayed(self) -> bool:
        """Check if architecture diagram is currently displayed.

        Returns:
            True if diagram is visible, False otherwise
        """
        return self.is_element_displayed(
            ArchitecturePageLocators.ARCHITECTURE_DIAGRAM
        )

    def download_architecture_diagram(self) -> bool:
        """Download the architecture diagram if available.

        Returns:
            True if download was initiated successfully, False otherwise
        """
        if self.is_element_displayed(
            ArchitecturePageLocators.DOWNLOAD_DIAGRAM_BUTTON
        ):
            self.click_element(
                ArchitecturePageLocators.DOWNLOAD_DIAGRAM_BUTTON,
                wait_for_rerun=False,  # Download shouldn't trigger rerun
            )
            return True

        return False

    def expand_model_information(self) -> "ArchitecturePage":
        """Expand the model information section if it's collapsed.

        Returns:
            Self for method chaining
        """
        if self.is_element_displayed(
            ArchitecturePageLocators.MODEL_INFO_EXPANDER
        ):
            self.click_element(ArchitecturePageLocators.MODEL_INFO_EXPANDER)

        return self

    def get_parameter_count(self) -> str | None:
        """Get the model parameter count information.

        Returns:
            Parameter count as string, or None if not available
        """
        # First try to expand model info section
        self.expand_model_information()

        return self.get_element_text(ArchitecturePageLocators.PARAMETER_COUNT)

    def wait_for_model_instantiation(
        self, timeout: float | None = None
    ) -> "ArchitecturePage":
        """Wait for model instantiation to complete.

        Args:
            timeout: Custom timeout for this operation (default: 30 seconds)

        Returns:
            Self for method chaining

        Raises:
            TimeoutException: If model doesn't instantiate within timeout
        """
        timeout = timeout or 30.0  # Model instantiation can take longer

        # Wait for model summary or diagram to appear
        self.wait_for_element(ArchitecturePageLocators.MODEL_SUMMARY, timeout)

        return self

    def scroll_to_diagram(self) -> "ArchitecturePage":
        """Scroll to the architecture diagram if it exists.

        Returns:
            Self for method chaining
        """
        if self.is_element_displayed(
            ArchitecturePageLocators.ARCHITECTURE_DIAGRAM
        ):
            self.scroll_to_element(
                ArchitecturePageLocators.ARCHITECTURE_DIAGRAM
            )

        return self

    def validate_model_loaded(self) -> bool:
        """Validate that a model has been properly loaded and displayed.

        Returns:
            True if model is properly loaded, False otherwise
        """
        # Check for key indicators of successful model loading
        has_summary = self.is_element_displayed(
            ArchitecturePageLocators.MODEL_SUMMARY
        )
        has_parameter_info = self.get_parameter_count() is not None

        return has_summary and has_parameter_info

    def get_model_architecture_type(self) -> str | None:
        """Extract model architecture type from the summary or title.

        Returns:
            Architecture type (e.g., 'U-Net', 'SwinTransformer'), or None if
                not found
        """
        summary = self.get_model_summary()
        if not summary:
            return None

        # Extract common architecture patterns for crack segmentation
        architecture_patterns = [
            "U-Net",
            "UNet",
            "SwinTransformer",
            "DeepLabV3",
            "ResNet",
            "EfficientNet",
            "CNN",
        ]

        for pattern in architecture_patterns:
            if pattern.lower() in summary.lower():
                return pattern

        return None

    def wait_for_page_content_loaded(
        self, timeout: float | None = None
    ) -> "ArchitecturePage":
        """Wait for the architecture page content to be fully loaded.

        Args:
            timeout: Custom timeout for this operation

        Returns:
            Self for method chaining
        """
        timeout = timeout or self.timeout

        # Wait for page to be ready
        self.wait_for_page_ready(timeout)

        # Wait for instantiate button to be available (indicates config is
        # loaded)
        self.wait_for_element(
            ArchitecturePageLocators.INSTANTIATE_MODEL_BUTTON, timeout
        )

        return self

    def check_model_instantiation_prerequisites(self) -> bool:
        """Check if prerequisites for model instantiation are met.

        Returns:
            True if model can be instantiated, False otherwise
        """
        # Check if instantiate button is available and enabled
        button = self.wait_for_element(
            ArchitecturePageLocators.INSTANTIATE_MODEL_BUTTON
        )
        if not button:
            return False

        # Check if button is enabled (not disabled)
        try:
            return button.is_enabled()
        except Exception:
            return False

    def get_instantiation_error_message(self) -> str | None:
        """Get error message if model instantiation failed.

        Returns:
            Error message text, or None if no error
        """
        from .locators import BaseLocators

        return self.get_element_text(BaseLocators.ERROR_MESSAGE)

    def retry_model_instantiation(self, max_retries: int = 3) -> bool:
        """Retry model instantiation with automatic retry logic.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            True if instantiation succeeded, False after all retries failed
        """
        for _attempt in range(max_retries):
            try:
                self.instantiate_model()

                # Wait a bit for instantiation to complete
                self.wait_for_no_spinners(timeout=30.0)

                # Check if successful
                if self.is_model_instantiated():
                    return True

            except Exception:
                # Continue to next attempt
                pass

            # Wait before retry
            import time

            time.sleep(2.0)

        return False
