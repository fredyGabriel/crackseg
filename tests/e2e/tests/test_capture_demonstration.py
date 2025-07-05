"""Demonstration tests for comprehensive screenshot and video capture system.

This module demonstrates the full capabilities of the E2E capture system,
including automated failure capture, manual checkpoints, video recording,
and configurable retention policies.
"""

import logging
import time
from pathlib import Path
from typing import Any

import pytest
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..base_test import BaseE2ETest

logger = logging.getLogger(__name__)


class TestCaptureSystemDemonstration(BaseE2ETest):
    """Comprehensive demonstration of capture system functionality.

    This test class showcases the complete capture system with various
    scenarios to validate screenshot and video recording capabilities.
    """

    def setup_test_data(self) -> dict[str, Any]:
        """Setup test data with comprehensive capture configuration.

        Returns:
            Test configuration with full capture settings enabled.
        """
        return {
            "test_url": "http://localhost:8501",
            "capture_config": {
                "screenshots": {
                    "enabled": True,
                    "on_failure": True,
                    "on_assertion": True,
                    "cleanup_on_success": False,  # Keep screenshots for demo
                },
                "videos": {
                    "enabled": True,
                    "record_all": True,  # Record all tests for demonstration
                    "cleanup_on_success": False,  # Keep videos for demo
                    "format": "mp4",
                    "quality": 7,
                },
            },
            "retention_policy": "keep_all",  # Keep all artifacts for demo
        }

    def get_test_data(self) -> dict[str, Any]:
        """Get test data configuration.

        Returns:
            Test configuration dictionary
        """
        # Override default test data with custom configuration
        custom_data = self.setup_test_data()
        self._test_data.update(custom_data)
        return self._test_data

    def test_successful_workflow_with_checkpoints(
        self, webdriver: WebDriver
    ) -> None:
        """Test successful workflow with manual screenshot checkpoints.

        This test demonstrates:
        - Manual screenshot capture at checkpoints
        - Video recording of entire workflow
        - Successful test completion with artifact retention

        Args:
            webdriver: WebDriver fixture from conftest.py
        """
        test_data = self.get_test_data()

        # Start video recording for this test
        recording_started = self.start_test_recording(webdriver)
        self.log_test_step(f"Video recording started: {recording_started}")

        # Navigate to application
        self.navigate_and_verify(webdriver, test_data["test_url"])

        # Checkpoint 1: Application loaded
        screenshot_path = self.capture_test_checkpoint(
            webdriver,
            "app_loaded",
            "Application successfully loaded and Streamlit ready",
        )
        if screenshot_path:
            self.log_test_step(
                f"Checkpoint screenshot saved: {screenshot_path}"
            )

        # Simulate user interaction
        self.log_test_step("Simulating user interaction")
        time.sleep(2)  # Give time for video to capture interaction

        # Checkpoint 2: After interaction
        self.capture_test_checkpoint(
            webdriver, "after_interaction", "After simulated user interaction"
        )

        # Verify application state
        self.assert_streamlit_loaded(webdriver)
        self.assert_page_ready_state(webdriver)

        # Final checkpoint
        self.capture_test_checkpoint(
            webdriver, "test_complete", "Test completed successfully"
        )

        # Stop recording manually (will also be stopped automatically)
        video_path = self.stop_test_recording(save_video=True)
        if video_path:
            self.log_test_step(f"Test video saved: {video_path}")

    def test_assertion_failure_capture(self, webdriver: WebDriver) -> None:
        """Test automatic screenshot capture on assertion failure.

        This test demonstrates:
        - Automatic screenshot on assertion failure
        - Video recording preservation on failure
        - Failure evidence collection

        Args:
            webdriver: WebDriver fixture from conftest.py
        """
        test_data = self.get_test_data()

        # Start recording
        self.start_test_recording(webdriver)

        # Navigate to application
        self.navigate_and_verify(webdriver, test_data["test_url"])

        # Capture screenshot before failure
        self.capture_test_checkpoint(
            webdriver,
            "before_assertion_failure",
            "State before intentional assertion failure",
        )

        # Intentionally fail an assertion to demonstrate capture
        with pytest.raises(AssertionError):
            # This will trigger automatic screenshot capture
            raise AssertionError(
                "Intentional failure to demonstrate capture system"
            )

    def test_timeout_failure_capture(self, webdriver: WebDriver) -> None:
        """Test automatic capture on timeout failures.

        This test demonstrates:
        - Capture on WebDriver timeout exceptions
        - Video recording during long operations
        - Timeout failure evidence

        Args:
            webdriver: WebDriver fixture from conftest.py
        """
        test_data = self.get_test_data()

        # Start recording
        self.start_test_recording(webdriver)

        # Navigate to application
        self.navigate_and_verify(webdriver, test_data["test_url"])

        # Capture screenshot before timeout attempt
        self.capture_test_checkpoint(
            webdriver, "before_timeout", "State before intentional timeout"
        )

        # Intentionally cause a timeout to demonstrate capture
        with pytest.raises(TimeoutException):
            # Wait for a non-existent element to trigger timeout
            WebDriverWait(webdriver, 2).until(
                EC.presence_of_element_located((By.ID, "non_existent_element"))
            )

    @pytest.mark.skip(
        reason="Manual demonstration - uncomment to test failure capture"
    )
    def test_manual_failure_demonstration(self, webdriver: WebDriver) -> None:
        """Test for manual demonstration of failure capture.

        This test is skipped by default but can be enabled to manually
        demonstrate the automatic failure capture system.

        Args:
            webdriver: WebDriver fixture from conftest.py
        """
        test_data = self.get_test_data()

        # Start recording
        self.start_test_recording(webdriver)

        # Navigate to application
        self.navigate_and_verify(webdriver, test_data["test_url"])

        # Take some screenshots during workflow
        self.capture_test_checkpoint(webdriver, "step_1", "First step")
        time.sleep(1)
        self.capture_test_checkpoint(webdriver, "step_2", "Second step")
        time.sleep(1)
        self.capture_test_checkpoint(webdriver, "step_3", "Third step")

        # Cause an intentional failure
        raise Exception("Manual failure for capture demonstration")

    def test_capture_system_configuration(self, webdriver: WebDriver) -> None:
        """Test capture system configuration and metadata.

        This test demonstrates:
        - Configuration validation
        - Metadata collection
        - System status reporting

        Args:
            webdriver: WebDriver fixture from conftest.py
        """
        test_data = self.get_test_data()

        # Verify capture configuration is properly loaded
        capture_config = test_data.get("capture_config", {})
        assert capture_config["screenshots"][
            "enabled"
        ], "Screenshot capture should be enabled"
        assert capture_config["videos"][
            "enabled"
        ], "Video recording should be enabled"

        # Start recording and verify status
        recording_started = self.start_test_recording(webdriver)
        assert recording_started, "Video recording should start successfully"

        # Verify recording is active
        assert self.is_recording(), "Recording should be active"

        # Navigate and take screenshots
        self.navigate_and_verify(webdriver, test_data["test_url"])

        # Capture test evidence
        screenshot_path = self.capture_test_checkpoint(
            webdriver,
            "config_test",
            "Configuration validation test checkpoint",
        )

        assert (
            screenshot_path is not None
        ), "Screenshot should be captured successfully"
        assert screenshot_path.exists(), "Screenshot file should exist"

        # Stop recording and verify video creation
        video_path = self.stop_test_recording(save_video=True)
        assert video_path is not None, "Video should be saved successfully"

        self.log_test_step(
            "Capture system configuration validated successfully"
        )

    def test_performance_with_capture(self, webdriver: WebDriver) -> None:
        """Test performance impact of capture system.

        This test demonstrates:
        - Performance monitoring with capture enabled
        - Minimal impact on test execution time
        - Resource usage monitoring

        Args:
            webdriver: WebDriver fixture from conftest.py
        """
        test_data = self.get_test_data()

        # Start recording
        start_time = time.time()
        self.start_test_recording(webdriver)

        # Navigate with performance measurement
        performance_metrics = self.navigate_and_verify(
            webdriver, test_data["test_url"], measure_performance=True
        )

        # Take multiple screenshots to test performance impact
        for i in range(3):
            self.capture_test_checkpoint(
                webdriver,
                f"performance_test_{i}",
                f"Performance test checkpoint {i}",
            )
            time.sleep(0.5)  # Small delay between captures

        # Stop recording and measure total time
        self.stop_test_recording(save_video=True)
        total_time = time.time() - start_time

        self.log_performance_metric(
            "test_with_capture_duration", total_time, "seconds"
        )

        # Verify performance is acceptable
        assert (
            total_time < 30.0
        ), f"Test with capture took too long: {total_time:.2f}s"

        if performance_metrics:
            self.log_test_step(f"Page load performance: {performance_metrics}")

    def test_storage_and_cleanup(self, webdriver: WebDriver) -> None:
        """Test storage management and cleanup functionality.

        This test demonstrates:
        - Artifact storage organization
        - Cleanup policy application
        - Storage metadata tracking

        Args:
            webdriver: WebDriver fixture from conftest.py
        """
        test_data = self.get_test_data()

        # Start recording
        self.start_test_recording(webdriver)

        # Navigate and create test artifacts
        self.navigate_and_verify(webdriver, test_data["test_url"])

        # Create multiple artifacts for storage testing
        artifacts = []
        for i in range(3):
            screenshot_path = self.capture_test_checkpoint(
                webdriver, f"storage_test_{i}", f"Storage test artifact {i}"
            )
            if screenshot_path:
                artifacts.append(screenshot_path)

        # Verify artifacts were created
        assert len(artifacts) > 0, "Test artifacts should be created"

        for artifact in artifacts:
            assert artifact.exists(), f"Artifact should exist: {artifact}"

        # Stop recording
        video_path = self.stop_test_recording(save_video=True)
        if video_path:
            artifacts.append(video_path)

        self.log_test_step(
            f"Created {len(artifacts)} test artifacts for storage validation"
        )

        # Verify artifact organization in storage
        artifacts_dir = Path("test-artifacts")
        assert artifacts_dir.exists(), "Artifacts directory should exist"

        # Check for proper organization
        screenshots_dir = artifacts_dir / "screenshots"
        videos_dir = artifacts_dir / "videos"

        if screenshots_dir.exists():
            screenshot_count = len(list(screenshots_dir.glob("*.png")))
            self.log_test_step(
                f"Found {screenshot_count} screenshots in storage"
            )

        if videos_dir.exists():
            video_count = len(list(videos_dir.glob("*.mp4")))
            self.log_test_step(f"Found {video_count} videos in storage")
