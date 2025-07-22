"""
Capture mixin for E2E testing. This module provides integration with
the comprehensive screenshot and video capture system for E2E tests.
"""

from pathlib import Path
from typing import Any

from selenium.webdriver.remote.webdriver import WebDriver

from ..capture import (
    CaptureStorage,
    ScreenshotCaptureMixin,
    ScreenshotConfig,
    StorageConfig,
    VideoConfig,
    VideoRecordingMixin,
)


class CaptureMixin(ScreenshotCaptureMixin, VideoRecordingMixin):
    """
    Mixin integrating comprehensive capture capabilities with E2E tests.
    Provides screenshot capture, video recording, and storage management
    with automatic failure handling and configurable retention policies.
    """

    _capture_storage: CaptureStorage | None = None
    _current_test_name: str | None = None
    _pytest_passed: bool = True

    def setup_capture_system(self) -> None:
        """Setup the complete capture system with storage and configuration."""
        try:
            storage_config = StorageConfig(
                base_dir=Path("test-artifacts"),
                retention_policy=self._get_retention_policy(),
                auto_cleanup=True,
                preserve_failure_artifacts=True,
            )
            self._capture_storage = CaptureStorage(storage_config)

            if hasattr(self, "log_test_step"):
                self.log_test_step("Capture system initialized")  # type: ignore[attr-defined]
        except Exception as e:
            if hasattr(self, "log_test_step"):
                self.log_test_step(f"Failed to setup capture storage: {e}")  # type: ignore[attr-defined]
            self._capture_storage = None

    def configure_capture_from_test_data(
        self, test_data: dict[str, Any]
    ) -> None:
        """
        Configure capture settings from test data. Args: test_data: Test
        configuration dictionary
        """
        capture_config = test_data.get("capture_config", {})

        # Setup screenshot configuration
        screenshot_config = capture_config.get("screenshots", {})
        if screenshot_config:
            config = ScreenshotConfig(
                enabled=screenshot_config.get("enabled", True),
                capture_on_failure=screenshot_config.get("on_failure", True),
                capture_on_assertion=screenshot_config.get(
                    "on_assertion", False
                ),
                cleanup_on_success=screenshot_config.get(
                    "cleanup_on_success", False
                ),
            )
            # Apply configuration to mixin
            if hasattr(self, "_screenshot_capture"):
                self._screenshot_capture.config = config

        # Setup video configuration
        video_config = capture_config.get("videos", {})
        if video_config:
            config = VideoConfig(
                enabled=video_config.get("enabled", True),
                record_all_tests=video_config.get("record_all", False),
                cleanup_on_success=video_config.get(
                    "cleanup_on_success", True
                ),
                format=video_config.get("format", "mp4"),
                quality=video_config.get("quality", 7),
            )
            # Apply configuration to mixin
            if hasattr(self, "_video_recording"):
                self._video_recording.config = config

    def on_test_failure(self, driver: WebDriver, exception: Exception) -> None:
        """
        Handle test failure with comprehensive capture. This method should be
        called by pytest hooks when a test fails. It automatically captures
        screenshots and stops video recording. Args: driver: WebDriver
        instance exception: Exception that caused the failure
        """
        if not self._current_test_name:
            return

        try:
            # Capture failure screenshot
            screenshot_path = self.capture_failure_screenshot(
                driver, self._current_test_name, exception
            )

            if screenshot_path and hasattr(self, "log_test_step"):
                self.log_test_step(  # type: ignore[attr-defined]
                    f"Failure evidence captured: {screenshot_path}"
                )

            # Stop video recording if active
            if hasattr(self, "_video_recording") and self.is_recording():
                video_path = self.stop_recording(save_video=True)
                if video_path and hasattr(self, "log_test_step"):
                    self.log_test_step(f"Failure video saved: {video_path}")  # type: ignore[attr-defined]

        except Exception as e:
            if hasattr(self, "log_test_step"):
                self.log_test_step(f"Failed to capture failure evidence: {e}")  # type: ignore[attr-defined]

    def capture_test_checkpoint(
        self,
        driver: WebDriver,
        checkpoint_name: str,
        context: str | None = None,
    ) -> Path | None:
        """
        Capture screenshot at a specific test checkpoint. Args: driver:
        WebDriver instance checkpoint_name: Name of the checkpoint context:
        Optional context information Returns: Path to captured screenshot or
        None if failed
        """
        if not self._current_test_name:
            return None

        full_name = f"{self._current_test_name}_{checkpoint_name}"
        return self.capture_screenshot(driver, full_name, context)

    def start_test_recording(self, driver: WebDriver) -> bool:
        """
        Start video recording for the current test. Args: driver: WebDriver
        instance Returns: True if recording started successfully
        """
        if not self._current_test_name:
            return False

        return self.start_recording(driver, self._current_test_name)

    def stop_test_recording(self, save_video: bool = True) -> Path | None:
        """
        Stop video recording for the current test. Args: save_video: Whether
        to save the recorded video Returns: Path to saved video or None if not
        saved
        """
        return self.stop_recording(save_video)

    def cleanup_capture_artifacts(self, test_passed: bool = False) -> None:
        """
        Clean up capture artifacts based on test result. Args: test_passed:
        Whether the test passed
        """
        if self._capture_storage and self._current_test_name:
            try:
                self.cleanup_test_screenshots(test_passed)
                self.cleanup_test_videos(test_passed)

                if hasattr(self, "log_test_step"):
                    self.log_test_step("Capture artifacts cleaned up")  # type: ignore[attr-defined]

            except Exception as e:
                if hasattr(self, "log_test_step"):
                    self.log_test_step(f"Capture cleanup failed: {e}")  # type: ignore[attr-defined]

    def _get_retention_policy(self) -> Any:
        """Get retention policy from test configuration."""
        from ..capture.storage import RetentionPolicy

        # Default policy - can be overridden by test data
        policy_name = getattr(self, "_test_data", {}).get(
            "retention_policy", "keep_failures"
        )
        return getattr(
            RetentionPolicy, policy_name.upper(), RetentionPolicy.KEEP_FAILURES
        )
