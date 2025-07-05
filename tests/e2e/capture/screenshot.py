"""Screenshot capture utilities for E2E testing.

This module provides automated screenshot capture on test failures, manual
screenshot capture capabilities, and integration with the existing test
infrastructure through mixin patterns.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotConfig:
    """Configuration for screenshot capture functionality.

    Attributes:
        enabled: Whether screenshot capture is enabled
        capture_on_failure: Automatically capture on test failures
        capture_on_assertion: Capture on assertion failures
        artifacts_dir: Directory to store screenshots
        filename_prefix: Prefix for screenshot filenames
        timestamp_format: Format for timestamp in filenames
        quality: Image quality for screenshots (if supported)
        max_screenshots_per_test: Maximum screenshots per test
        cleanup_on_success: Remove screenshots if test passes
    """

    enabled: bool = True
    capture_on_failure: bool = True
    capture_on_assertion: bool = False
    artifacts_dir: Path = field(
        default_factory=lambda: Path("test-artifacts/screenshots")
    )
    filename_prefix: str = "screenshot"
    timestamp_format: str = "%Y%m%d_%H%M%S_%f"
    quality: int = 95
    max_screenshots_per_test: int = 10
    cleanup_on_success: bool = False


class HasScreenshotCapture(Protocol):
    """Protocol for classes that support screenshot capture."""

    def capture_screenshot(
        self,
        driver: WebDriver,
        name: str | None = None,
        context: str | None = None,
    ) -> Path | None:
        """Capture a screenshot with optional name and context."""
        ...

    def capture_failure_screenshot(
        self,
        driver: WebDriver,
        test_name: str,
        exception: Exception | None = None,
    ) -> Path | None:
        """Capture screenshot on test failure."""
        ...


class ScreenshotCapture:
    """Core screenshot capture functionality.

    Provides methods for capturing screenshots during test execution,
    with support for both manual and automatic capture scenarios.
    """

    def __init__(self, config: ScreenshotConfig | None = None) -> None:
        """Initialize screenshot capture with configuration.

        Args:
            config: Screenshot capture configuration
        """
        self.config = config or ScreenshotConfig()
        self._screenshot_count = 0
        self._test_screenshots: list[Path] = []

        # Ensure artifacts directory exists
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Screenshot capture initialized: {self.config.artifacts_dir}"
        )

    def capture_screenshot(
        self,
        driver: WebDriver,
        name: str | None = None,
        context: str | None = None,
    ) -> Path | None:
        """Capture a screenshot with optional name and context.

        Args:
            driver: WebDriver instance
            name: Optional name for the screenshot
            context: Additional context information

        Returns:
            Path to saved screenshot, or None if capture failed
        """
        if not self.config.enabled:
            logger.debug("Screenshot capture disabled")
            return None

        if self._screenshot_count >= self.config.max_screenshots_per_test:
            logger.warning(
                f"Maximum screenshots per test reached: "
                f"{self.config.max_screenshots_per_test}"
            )
            return None

        try:
            timestamp = time.strftime(self.config.timestamp_format)
            screenshot_name = self._generate_filename(name, context, timestamp)
            screenshot_path = self.config.artifacts_dir / screenshot_name

            # Capture screenshot
            success = driver.save_screenshot(str(screenshot_path))

            if success and screenshot_path.exists():
                self._screenshot_count += 1
                self._test_screenshots.append(screenshot_path)

                logger.info(f"Screenshot captured: {screenshot_path}")

                if context:
                    logger.debug(f"Screenshot context: {context}")

                return screenshot_path
            else:
                logger.error(f"Failed to save screenshot: {screenshot_path}")
                return None

        except WebDriverException as e:
            logger.error(f"WebDriver error during screenshot capture: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during screenshot capture: {e}")
            return None

    def capture_failure_screenshot(
        self,
        driver: WebDriver,
        test_name: str,
        exception: Exception | None = None,
    ) -> Path | None:
        """Capture screenshot on test failure.

        Args:
            driver: WebDriver instance
            test_name: Name of the failing test
            exception: Exception that caused the failure

        Returns:
            Path to saved screenshot, or None if capture failed
        """
        if not self.config.capture_on_failure:
            logger.debug("Failure screenshot capture disabled")
            return None

        context = f"FAILURE: {test_name}"
        if exception:
            context += f" - {type(exception).__name__}: {str(exception)[:100]}"

        screenshot_path = self.capture_screenshot(
            driver=driver,
            name=f"failure_{test_name}",
            context=context,
        )

        if screenshot_path:
            logger.error(f"Failure screenshot captured: {screenshot_path}")

        return screenshot_path

    def capture_assertion_screenshot(
        self,
        driver: WebDriver,
        assertion_description: str,
        test_name: str,
    ) -> Path | None:
        """Capture screenshot on assertion failure.

        Args:
            driver: WebDriver instance
            assertion_description: Description of the failed assertion
            test_name: Name of the test containing the assertion

        Returns:
            Path to saved screenshot, or None if capture failed
        """
        if not self.config.capture_on_assertion:
            logger.debug("Assertion screenshot capture disabled")
            return None

        context = f"ASSERTION FAILURE: {assertion_description}"

        return self.capture_screenshot(
            driver=driver,
            name=f"assertion_{test_name}",
            context=context,
        )

    def cleanup_test_screenshots(self, test_passed: bool = False) -> None:
        """Clean up screenshots from current test.

        Args:
            test_passed: Whether the test passed or failed
        """
        if test_passed and self.config.cleanup_on_success:
            logger.debug("Cleaning up screenshots from successful test")
            for screenshot_path in self._test_screenshots:
                try:
                    if screenshot_path.exists():
                        screenshot_path.unlink()
                        logger.debug(f"Removed screenshot: {screenshot_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove screenshot {screenshot_path}: {e}"
                    )

        # Reset for next test
        self._screenshot_count = 0
        self._test_screenshots.clear()

    def get_test_screenshots(self) -> list[Path]:
        """Get list of screenshots captured in current test.

        Returns:
            List of screenshot file paths
        """
        return self._test_screenshots.copy()

    def _generate_filename(
        self,
        name: str | None,
        context: str | None,
        timestamp: str,
    ) -> str:
        """Generate filename for screenshot.

        Args:
            name: Optional base name
            context: Optional context string
            timestamp: Timestamp string

        Returns:
            Generated filename
        """
        parts = [self.config.filename_prefix]

        if name:
            # Sanitize name for filename
            sanitized_name = "".join(
                c for c in name if c.isalnum() or c in "._-"
            ).rstrip()
            parts.append(sanitized_name)

        parts.append(timestamp)

        return "_".join(parts) + ".png"


class ScreenshotCaptureMixin:
    """Mixin providing screenshot capture capabilities for test classes.

    Integrates with the existing BaseE2ETest mixin pattern to provide
    screenshot capture functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize screenshot capture mixin."""
        screenshot_config = kwargs.pop("screenshot_config", None)
        super().__init__(*args, **kwargs)
        self._screenshot_capture = ScreenshotCapture(screenshot_config)

        if hasattr(self, "_test_logger"):
            self._test_logger.debug("Screenshot capture mixin initialized")  # type: ignore[attr-defined]

    def capture_screenshot(
        self,
        driver: WebDriver,
        name: str | None = None,
        context: str | None = None,
    ) -> Path | None:
        """Capture a screenshot with optional name and context.

        Args:
            driver: WebDriver instance
            name: Optional name for the screenshot
            context: Additional context information

        Returns:
            Path to saved screenshot, or None if capture failed
        """
        if hasattr(self, "log_test_step"):
            step_msg = "Capturing screenshot"
            if name:
                step_msg += f": {name}"
            self.log_test_step(step_msg)  # type: ignore[attr-defined]

        screenshot_path = self._screenshot_capture.capture_screenshot(
            driver=driver,
            name=name,
            context=context,
        )

        if screenshot_path and hasattr(self, "log_test_step"):
            self.log_test_step(  # type: ignore[attr-defined]
                f"Screenshot saved: {screenshot_path.name}"
            )

        return screenshot_path

    def capture_failure_screenshot(
        self,
        driver: WebDriver,
        test_name: str,
        exception: Exception | None = None,
    ) -> Path | None:
        """Capture screenshot on test failure.

        Args:
            driver: WebDriver instance
            test_name: Name of the failing test
            exception: Exception that caused the failure

        Returns:
            Path to saved screenshot, or None if capture failed
        """
        if hasattr(self, "_test_logger"):
            self._test_logger.error(  # type: ignore[attr-defined]
                f"Capturing failure screenshot for test: {test_name}"
            )

        return self._screenshot_capture.capture_failure_screenshot(
            driver=driver,
            test_name=test_name,
            exception=exception,
        )

    def capture_assertion_screenshot(
        self,
        driver: WebDriver,
        assertion_description: str,
        test_name: str,
    ) -> Path | None:
        """Capture screenshot on assertion failure.

        Args:
            driver: WebDriver instance
            assertion_description: Description of the failed assertion
            test_name: Name of the test containing the assertion

        Returns:
            Path to saved screenshot, or None if capture failed
        """
        return self._screenshot_capture.capture_assertion_screenshot(
            driver=driver,
            assertion_description=assertion_description,
            test_name=test_name,
        )

    def cleanup_test_screenshots(self, test_passed: bool = False) -> None:
        """Clean up screenshots from current test.

        Args:
            test_passed: Whether the test passed or failed
        """
        self._screenshot_capture.cleanup_test_screenshots(test_passed)

    def get_test_screenshots(self) -> list[Path]:
        """Get list of screenshots captured in current test.

        Returns:
            List of screenshot file paths
        """
        return self._screenshot_capture.get_test_screenshots()
