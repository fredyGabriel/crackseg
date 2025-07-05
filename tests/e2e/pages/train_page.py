"""Train page object for the CrackSeg Streamlit application.

This module provides the TrainPage class that handles interactions with
the training execution page, including starting/stopping training,
monitoring progress, and viewing training metrics.
"""

from .base_page import BasePage
from .locators import TrainPageLocators


class TrainPage(BasePage):
    """Page object for the Training page.

    Handles training execution, monitoring, and control operations
    on the Train page of the CrackSeg application.
    """

    @property
    def page_name(self) -> str:
        """Return the name of the page for navigation."""
        return "Train"

    @property
    def page_title_locator(self) -> tuple[str, str]:
        """Return the locator for the page title element."""
        return TrainPageLocators.PAGE_TITLE

    @property
    def expected_url_fragment(self) -> str | None:
        """Return expected URL fragment for this page."""
        return None  # Streamlit doesn't use URL fragments for page routing

    def start_training(self) -> "TrainPage":
        """Start model training.

        Returns:
            Self for method chaining
        """
        if self.is_element_displayed(TrainPageLocators.START_TRAINING_BUTTON):
            self.click_element(
                TrainPageLocators.START_TRAINING_BUTTON, wait_for_rerun=True
            )

        return self

    def stop_training(self) -> "TrainPage":
        """Stop currently running training.

        Returns:
            Self for method chaining
        """
        if self.is_element_displayed(TrainPageLocators.STOP_TRAINING_BUTTON):
            self.click_element(
                TrainPageLocators.STOP_TRAINING_BUTTON, wait_for_rerun=True
            )

        return self

    def resume_training(self) -> "TrainPage":
        """Resume paused training.

        Returns:
            Self for method chaining
        """
        if self.is_element_displayed(TrainPageLocators.RESUME_TRAINING_BUTTON):
            self.click_element(
                TrainPageLocators.RESUME_TRAINING_BUTTON, wait_for_rerun=True
            )

        return self

    def get_training_status(self) -> str | None:
        """Get current training status.

        Returns:
            Training status text, or None if not available
        """
        return self.get_element_text(TrainPageLocators.TRAINING_STATUS)

    def is_training_running(self) -> bool:
        """Check if training is currently running.

        Returns:
            True if training is active, False otherwise
        """
        status = self.get_training_status()
        if not status:
            return False

        # Check for common training status indicators
        running_indicators = ["running", "training", "in progress", "active"]
        return any(
            indicator in status.lower() for indicator in running_indicators
        )

    def get_current_epoch(self) -> str | None:
        """Get current training epoch information.

        Returns:
            Epoch information as string, or None if not available
        """
        return self.get_element_text(TrainPageLocators.EPOCH_INFO)

    def get_training_metrics(self) -> dict[str, str | None]:
        """Get current training metrics.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Get loss metric
        metrics["loss"] = self.get_element_text(TrainPageLocators.LOSS_METRIC)

        # Get accuracy metric
        metrics["accuracy"] = self.get_element_text(
            TrainPageLocators.ACCURACY_METRIC
        )

        return metrics

    def wait_for_training_to_start(
        self, timeout: float | None = None
    ) -> "TrainPage":
        """Wait for training to begin.

        Args:
            timeout: Custom timeout for this operation

        Returns:
            Self for method chaining
        """
        timeout = timeout or 30.0

        # Wait for training status to indicate training has started
        self.wait_for_element(TrainPageLocators.TRAINING_STATUS, timeout)

        return self

    def wait_for_training_completion(
        self, timeout: float | None = None
    ) -> "TrainPage":
        """Wait for training to complete.

        Args:
            timeout: Custom timeout for this operation (default: very long)

        Returns:
            Self for method chaining
        """
        timeout = timeout or 3600.0  # Training can take very long

        # Implementation would involve checking for completion indicators
        # This is a placeholder - actual implementation would check for
        # specific completion messages or status changes

        return self

    def get_training_progress(self) -> float | None:
        """Get training progress as percentage.

        Returns:
            Progress percentage (0-100), or None if not available
        """
        progress_element = self.wait_for_element(
            TrainPageLocators.PROGRESS_BAR
        )
        if not progress_element:
            return None

        try:
            # Extract progress value from progress bar
            # Implementation depends on how Streamlit renders progress bars
            progress_text = progress_element.get_attribute("aria-valuenow")
            if progress_text:
                return float(progress_text)
        except (ValueError, AttributeError):
            pass

        return None

    def is_training_button_available(self) -> bool:
        """Check if start training button is available and enabled.

        Returns:
            True if training can be started, False otherwise
        """
        button = self.wait_for_element(TrainPageLocators.START_TRAINING_BUTTON)
        if not button:
            return False

        try:
            return button.is_enabled()
        except Exception:
            return False

    def validate_training_prerequisites(self) -> bool:
        """Validate that prerequisites for training are met.

        Returns:
            True if training can be started, False otherwise
        """
        # Check if start training button is available
        return self.is_training_button_available()

    def get_training_error_message(self) -> str | None:
        """Get error message if training failed.

        Returns:
            Error message text, or None if no error
        """
        from .locators import BaseLocators

        return self.get_element_text(BaseLocators.ERROR_MESSAGE)
