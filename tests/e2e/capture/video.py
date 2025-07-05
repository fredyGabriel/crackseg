"""Video recording utilities for E2E testing.

This module provides video recording capabilities for test execution,
integrating with browser automation and the existing Docker video
infrastructure for comprehensive test documentation and debugging.
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video recording functionality.

    Attributes:
        enabled: Whether video recording is enabled
        record_all_tests: Record all tests (not just failures)
        artifacts_dir: Directory to store video recordings
        filename_prefix: Prefix for video filenames
        format: Video format ('mp4', 'webm', 'avi')
        fps: Frames per second for recording
        quality: Video quality (1-10, 10 is best)
        max_duration_minutes: Maximum recording duration per test
        cleanup_on_success: Remove videos if test passes
        compression_enabled: Enable video compression
    """

    enabled: bool = True
    record_all_tests: bool = False
    artifacts_dir: Path = field(
        default_factory=lambda: Path("test-artifacts/videos")
    )
    filename_prefix: str = "video"
    format: str = "mp4"
    fps: int = 10
    quality: int = 7
    max_duration_minutes: int = 30
    cleanup_on_success: bool = True
    compression_enabled: bool = True


class HasVideoRecording(Protocol):
    """Protocol for classes that support video recording."""

    def start_recording(
        self,
        driver: WebDriver,
        test_name: str,
    ) -> bool:
        """Start video recording for a test."""
        ...

    def stop_recording(
        self,
        save_video: bool = True,
    ) -> Path | None:
        """Stop video recording and optionally save."""
        ...


class VideoRecording:
    """Core video recording functionality.

    Provides methods for recording test execution through browser automation,
    with support for various video formats and quality settings.
    """

    def __init__(self, config: VideoConfig | None = None) -> None:
        """Initialize video recording with configuration.

        Args:
            config: Video recording configuration
        """
        self.config = config or VideoConfig()
        self._is_recording = False
        self._recording_process: subprocess.Popen[bytes] | None = None
        self._recording_start_time: float | None = None
        self._current_video_path: Path | None = None
        self._test_videos: list[Path] = []

        # Ensure artifacts directory exists
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Video recording initialized: {self.config.artifacts_dir}"
        )

    def start_recording(
        self,
        driver: WebDriver,
        test_name: str,
    ) -> bool:
        """Start video recording for a test.

        Args:
            driver: WebDriver instance
            test_name: Name of the test being recorded

        Returns:
            True if recording started successfully, False otherwise
        """
        if not self.config.enabled:
            logger.debug("Video recording disabled")
            return False

        if self._is_recording:
            logger.warning("Video recording already in progress")
            return False

        try:
            # Generate video filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_name = self._generate_filename(test_name, timestamp)
            self._current_video_path = self.config.artifacts_dir / video_name

            # Check if Docker video recording is available
            if self._try_docker_recording(test_name):
                logger.info(f"Started Docker video recording for: {test_name}")
                self._is_recording = True
                self._recording_start_time = time.time()
                return True

            # Fallback to browser-based recording
            if self._try_browser_recording(driver, test_name):
                logger.info(
                    f"Started browser video recording for: {test_name}"
                )
                self._is_recording = True
                self._recording_start_time = time.time()
                return True

            logger.warning("No video recording method available")
            return False

        except Exception as e:
            logger.error(f"Failed to start video recording: {e}")
            return False

    def stop_recording(
        self,
        save_video: bool = True,
    ) -> Path | None:
        """Stop video recording and optionally save.

        Args:
            save_video: Whether to save the recorded video

        Returns:
            Path to saved video, or None if not saved
        """
        if not self._is_recording:
            logger.debug("No recording in progress")
            return None

        try:
            # Stop recording process
            if self._recording_process:
                self._recording_process.terminate()
                self._recording_process.wait(timeout=10)
                self._recording_process = None

            # Calculate recording duration
            duration = 0.0
            if self._recording_start_time:
                duration = time.time() - self._recording_start_time

            logger.info(f"Video recording stopped (duration: {duration:.1f}s)")

            self._is_recording = False
            self._recording_start_time = None

            # Save or cleanup video
            if save_video and self._current_video_path:
                if self._current_video_path.exists():
                    # Apply compression if enabled
                    final_path = self._current_video_path
                    if self.config.compression_enabled:
                        final_path = self._compress_video(
                            self._current_video_path
                        )

                    self._test_videos.append(final_path)
                    logger.info(f"Video saved: {final_path}")

                    video_path = final_path
                    self._current_video_path = None
                    return video_path
                else:
                    logger.warning(
                        f"Video file not found: {self._current_video_path}"
                    )
            else:
                # Cleanup video if not saving
                if (
                    self._current_video_path
                    and self._current_video_path.exists()
                ):
                    self._current_video_path.unlink()
                    logger.debug(f"Video cleanup: {self._current_video_path}")

            self._current_video_path = None
            return None

        except Exception as e:
            logger.error(f"Error stopping video recording: {e}")
            self._is_recording = False
            self._current_video_path = None
            return None

    def is_recording(self) -> bool:
        """Check if video recording is currently active.

        Returns:
            True if recording is active, False otherwise
        """
        return self._is_recording

    def get_recording_duration(self) -> float:
        """Get current recording duration in seconds.

        Returns:
            Recording duration in seconds, or 0 if not recording
        """
        if not self._is_recording or not self._recording_start_time:
            return 0.0

        return time.time() - self._recording_start_time

    def cleanup_test_videos(self, test_passed: bool = False) -> None:
        """Clean up videos from current test.

        Args:
            test_passed: Whether the test passed or failed
        """
        if test_passed and self.config.cleanup_on_success:
            logger.debug("Cleaning up videos from successful test")
            for video_path in self._test_videos:
                try:
                    if video_path.exists():
                        video_path.unlink()
                        logger.debug(f"Removed video: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove video {video_path}: {e}")

        # Reset for next test
        self._test_videos.clear()

    def get_test_videos(self) -> list[Path]:
        """Get list of videos recorded in current test.

        Returns:
            List of video file paths
        """
        return self._test_videos.copy()

    def _try_docker_recording(self, test_name: str) -> bool:
        """Try to start Docker-based video recording.

        Args:
            test_name: Name of the test

        Returns:
            True if Docker recording started, False otherwise
        """
        try:
            # Check if Docker video recording infrastructure is available
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=video-recorder", "--quiet"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                logger.debug("Docker video recording container found")
                # Docker video recording handled by container infrastructure
                # We just need to signal that recording should start
                return True

            logger.debug("Docker video recording not available")
            return False

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.debug(f"Docker check failed: {e}")
            return False

    def _try_browser_recording(
        self, driver: WebDriver, test_name: str
    ) -> bool:
        """Try to start browser-based screen recording.

        Args:
            driver: WebDriver instance
            test_name: Name of the test

        Returns:
            True if browser recording started, False otherwise
        """
        try:
            # Check if browser supports screen recording
            if not self._browser_supports_recording(driver):
                logger.debug("Browser does not support recording")
                return False

            # Try to start recording using JavaScript MediaRecorder API
            recording_script = """
            return new Promise((resolve) => {
                navigator.mediaDevices.getDisplayMedia({
                    video: {
                        mediaSource: 'screen',
                        frameRate: arguments[0]
                    },
                    audio: false
                }).then(stream => {
                    window.testRecorder = new MediaRecorder(stream);
                    window.testRecordedChunks = [];

                    window.testRecorder.ondataavailable = function(event) {
                        if (event.data.size > 0) {
                            window.testRecordedChunks.push(event.data);
                        }
                    };

                    window.testRecorder.start();
                    resolve(true);
                }).catch(error => {
                    console.log('Recording error:', error);
                    resolve(false);
                });
            });
            """

            result = driver.execute_async_script(
                recording_script, self.config.fps
            )

            if result:
                logger.debug("Browser recording started successfully")
                return True
            else:
                logger.debug("Failed to start browser recording")
                return False

        except WebDriverException as e:
            logger.debug(f"Browser recording error: {e}")
            return False

    def _browser_supports_recording(self, driver: WebDriver) -> bool:
        """Check if browser supports screen recording.

        Args:
            driver: WebDriver instance

        Returns:
            True if browser supports recording, False otherwise
        """
        try:
            support_check = """
            return (
                typeof navigator.mediaDevices !== 'undefined' &&
                typeof navigator.mediaDevices.getDisplayMedia !==
                    'undefined' &&
                typeof MediaRecorder !== 'undefined'
            );
            """

            return driver.execute_script(support_check)

        except WebDriverException:
            return False

    def _compress_video(self, video_path: Path) -> Path:
        """Compress video file for storage efficiency.

        Args:
            video_path: Path to original video file

        Returns:
            Path to compressed video file
        """
        if not video_path.exists():
            return video_path

        try:
            compressed_path = video_path.with_name(
                f"{video_path.stem}_compressed{video_path.suffix}"
            )

            # Use ffmpeg for compression if available
            compress_cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-crf",
                str(23 + (10 - self.config.quality)),
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                str(compressed_path),
            ]

            result = subprocess.run(
                compress_cmd,
                capture_output=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0 and compressed_path.exists():
                # Remove original if compression successful
                video_path.unlink()
                logger.debug(f"Video compressed: {compressed_path}")
                return compressed_path
            else:
                logger.warning("Video compression failed, keeping original")
                return video_path

        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            FileNotFoundError,
        ):
            logger.debug("ffmpeg not available for compression")
            return video_path

    def _generate_filename(self, test_name: str, timestamp: str) -> str:
        """Generate filename for video recording.

        Args:
            test_name: Name of the test
            timestamp: Timestamp string

        Returns:
            Generated filename
        """
        # Sanitize test name for filename
        sanitized_name = "".join(
            c for c in test_name if c.isalnum() or c in "._-"
        ).rstrip()

        filename = (
            f"{self.config.filename_prefix}_{sanitized_name}_{timestamp}"
            f".{self.config.format}"
        )
        return filename


class VideoRecordingMixin:
    """Mixin providing video recording capabilities for test classes.

    Integrates with the existing BaseE2ETest mixin pattern to provide
    video recording functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize video recording mixin."""
        super().__init__(*args, **kwargs)
        self._video_recording = VideoRecording()

        if hasattr(self, "_test_logger"):
            self._test_logger.debug("Video recording mixin initialized")  # type: ignore[attr-defined]

    def start_recording(
        self,
        driver: WebDriver,
        test_name: str,
    ) -> bool:
        """Start video recording for a test.

        Args:
            driver: WebDriver instance
            test_name: Name of the test being recorded

        Returns:
            True if recording started successfully, False otherwise
        """
        if hasattr(self, "log_test_step"):
            self.log_test_step(f"Starting video recording: {test_name}")  # type: ignore[attr-defined]

        success = self._video_recording.start_recording(driver, test_name)

        if success and hasattr(self, "log_test_step"):
            self.log_test_step("Video recording started successfully")  # type: ignore[attr-defined]
        elif hasattr(self, "_test_logger"):
            self._test_logger.warning("Failed to start video recording")  # type: ignore[attr-defined]

        return success

    def stop_recording(
        self,
        save_video: bool = True,
    ) -> Path | None:
        """Stop video recording and optionally save.

        Args:
            save_video: Whether to save the recorded video

        Returns:
            Path to saved video, or None if not saved
        """
        if hasattr(self, "log_test_step"):
            self.log_test_step("Stopping video recording")  # type: ignore[attr-defined]

        video_path = self._video_recording.stop_recording(save_video)

        if video_path and hasattr(self, "log_test_step"):
            self.log_test_step(f"Video saved: {video_path.name}")  # type: ignore[attr-defined]

        return video_path

    def is_recording(self) -> bool:
        """Check if video recording is currently active.

        Returns:
            True if recording is active, False otherwise
        """
        return self._video_recording.is_recording()

    def get_recording_duration(self) -> float:
        """Get current recording duration in seconds.

        Returns:
            Recording duration in seconds, or 0 if not recording
        """
        return self._video_recording.get_recording_duration()

    def cleanup_test_videos(self, test_passed: bool = False) -> None:
        """Clean up videos from current test.

        Args:
            test_passed: Whether the test passed or failed
        """
        self._video_recording.cleanup_test_videos(test_passed)

    def get_test_videos(self) -> list[Path]:
        """Get list of videos recorded in current test.

        Returns:
            List of video file paths
        """
        return self._video_recording.get_test_videos()
