"""Results page object for the CrackSeg Streamlit application.

This module provides the ResultsPage class that handles interactions with
the results visualization page, including gallery viewing, TensorBoard
access, metrics analysis, and result exports.
"""

from .base_page import BasePage
from .locators import ResultsPageLocators


class ResultsPage(BasePage):
    """Page object for the Results page.

    Handles results visualization, analysis, and export operations
    on the Results page of the CrackSeg application.
    """

    @property
    def page_name(self) -> str:
        """Return the name of the page for navigation."""
        return "Results"

    @property
    def page_title_locator(self) -> tuple[str, str]:
        """Return the locator for the page title element."""
        return ResultsPageLocators.PAGE_TITLE

    @property
    def expected_url_fragment(self) -> str | None:
        """Return expected URL fragment for this page."""
        return None  # Streamlit doesn't use URL fragments for page routing

    def navigate_to_gallery_tab(self) -> "ResultsPage":
        """Navigate to the Results Gallery tab.

        Returns:
            Self for method chaining
        """
        self.click_element(ResultsPageLocators.GALLERY_TAB)
        return self

    def navigate_to_tensorboard_tab(self) -> "ResultsPage":
        """Navigate to the TensorBoard tab.

        Returns:
            Self for method chaining
        """
        self.click_element(ResultsPageLocators.TENSORBOARD_TAB)
        return self

    def navigate_to_metrics_tab(self) -> "ResultsPage":
        """Navigate to the Metrics Analysis tab.

        Returns:
            Self for method chaining
        """
        self.click_element(ResultsPageLocators.METRICS_TAB)
        return self

    def navigate_to_comparison_tab(self) -> "ResultsPage":
        """Navigate to the Model Comparison tab.

        Returns:
            Self for method chaining
        """
        self.click_element(ResultsPageLocators.COMPARISON_TAB)
        return self

    def is_results_gallery_displayed(self) -> bool:
        """Check if results gallery is currently displayed.

        Returns:
            True if gallery is visible, False otherwise
        """
        return self.is_element_displayed(ResultsPageLocators.RESULTS_GALLERY)

    def get_image_count(self) -> int:
        """Get the number of images in the results gallery.

        Returns:
            Number of images displayed
        """
        try:
            images = self.driver.find_elements(
                *ResultsPageLocators.RESULTS_GALLERY
            )
            return len(images)
        except Exception:
            return 0

    def export_results(self) -> bool:
        """Export results if export functionality is available.

        Returns:
            True if export was initiated successfully, False otherwise
        """
        if self.is_element_displayed(ResultsPageLocators.EXPORT_BUTTON):
            self.click_element(
                ResultsPageLocators.EXPORT_BUTTON, wait_for_rerun=False
            )
            return True

        return False

    def download_results(self) -> bool:
        """Download results if download functionality is available.

        Returns:
            True if download was initiated successfully, False otherwise
        """
        if self.is_element_displayed(
            ResultsPageLocators.DOWNLOAD_RESULTS_BUTTON
        ):
            self.click_element(
                ResultsPageLocators.DOWNLOAD_RESULTS_BUTTON,
                wait_for_rerun=False,
            )
            return True

        return False

    def wait_for_results_loaded(
        self, timeout: float | None = None
    ) -> "ResultsPage":
        """Wait for results to be loaded and displayed.

        Args:
            timeout: Custom timeout for this operation

        Returns:
            Self for method chaining
        """
        timeout = timeout or self.timeout

        # Wait for gallery or other result indicators
        self.wait_for_element(ResultsPageLocators.RESULTS_GALLERY, timeout)

        return self

    def validate_results_available(self) -> bool:
        """Validate that results are available for viewing.

        Returns:
            True if results are available, False otherwise
        """
        return (
            self.is_results_gallery_displayed() or self.get_image_count() > 0
        )

    def scroll_to_gallery(self) -> "ResultsPage":
        """Scroll to the results gallery section.

        Returns:
            Self for method chaining
        """
        if self.is_element_displayed(ResultsPageLocators.RESULTS_GALLERY):
            self.scroll_to_element(ResultsPageLocators.RESULTS_GALLERY)

        return self

    def get_active_tab(self) -> str | None:
        """Get the currently active tab name.

        Returns:
            Active tab name, or None if not determinable
        """
        # This would need to be implemented based on actual Streamlit tab
        # structure
        # Placeholder implementation
        tabs = [
            ("gallery", ResultsPageLocators.GALLERY_TAB),
            ("tensorboard", ResultsPageLocators.TENSORBOARD_TAB),
            ("metrics", ResultsPageLocators.METRICS_TAB),
            ("comparison", ResultsPageLocators.COMPARISON_TAB),
        ]

        for tab_name, tab_locator in tabs:
            element = self.wait_for_element(tab_locator)
            if element and "active" in element.get_attribute("class"):
                return tab_name

        return None

    def check_results_prerequisites(self) -> bool:
        """Check if prerequisites for viewing results are met.

        Returns:
            True if results can be viewed, False otherwise
        """
        # Results page should have some content or indicate what's needed
        return self.is_page_loaded()
