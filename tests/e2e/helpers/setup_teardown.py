"""Advanced setup and teardown helpers for E2E test environment management.

This module provides comprehensive utilities for managing test environments,
including cleanup procedures, state restoration, and artifact management.
These helpers ensure clean test isolation and reliable teardown procedures.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, TypedDict

from selenium.webdriver.remote.webdriver import WebDriver

from ..utils import (
    cleanup_test_files,
    ensure_artifacts_dir,
    get_current_timestamp,
    save_test_artifacts,
)

logger = logging.getLogger(__name__)


class TestEnvironmentState(TypedDict):
    """State information for test environment."""

    timestamp: float
    artifacts_dir: Path
    cleanup_paths: list[Path]
    running_processes: list[int]
    environment_vars: dict[str, str | None]
    temp_files: list[Path]


class TestEnvironmentManager:
    """Comprehensive test environment manager with cleanup and restoration."""

    def __init__(
        self, test_name: str, artifacts_base_dir: str | Path = "test-artifacts"
    ) -> None:
        """Initialize test environment manager.

        Args:
            test_name: Name of the test for environment isolation
            artifacts_base_dir: Base directory for test artifacts
        """
        self.test_name = test_name
        self.artifacts_base_dir = Path(artifacts_base_dir)
        self.state: TestEnvironmentState = {
            "timestamp": get_current_timestamp(),
            "artifacts_dir": self.artifacts_base_dir / test_name,
            "cleanup_paths": [],
            "running_processes": [],
            "environment_vars": {},
            "temp_files": [],
        }
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the environment manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.test_name}")

    def create_clean_environment(self) -> Path:
        """Create a clean, isolated test environment.

        Returns:
            Path to the test artifacts directory

        Raises:
            RuntimeError: If environment setup fails
        """
        try:
            # Create isolated artifacts directory
            artifacts_dir = ensure_artifacts_dir(
                str(self.state["artifacts_dir"])
            )
            self.state["artifacts_dir"] = artifacts_dir

            # Clean any existing artifacts from previous runs
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for different artifact types
            (artifacts_dir / "screenshots").mkdir(exist_ok=True)
            (artifacts_dir / "logs").mkdir(exist_ok=True)
            (artifacts_dir / "configs").mkdir(exist_ok=True)
            (artifacts_dir / "reports").mkdir(exist_ok=True)

            self.logger.info(
                f"Clean test environment created: {artifacts_dir}"
            )
            return artifacts_dir

        except Exception as e:
            self.logger.error(f"Failed to create clean environment: {e}")
            raise RuntimeError(f"Environment setup failed: {e}") from e

    def register_cleanup_path(self, path: Path | str) -> None:
        """Register a path for cleanup during teardown.

        Args:
            path: Path to be cleaned up
        """
        cleanup_path = Path(path)
        if cleanup_path not in self.state["cleanup_paths"]:
            self.state["cleanup_paths"].append(cleanup_path)
            self.logger.debug(f"Registered cleanup path: {cleanup_path}")

    def register_process(self, pid: int) -> None:
        """Register a process for cleanup during teardown.

        Args:
            pid: Process ID to be terminated
        """
        if pid not in self.state["running_processes"]:
            self.state["running_processes"].append(pid)
            self.logger.debug(f"Registered process for cleanup: {pid}")

    def register_temp_file(self, file_path: Path | str) -> None:
        """Register a temporary file for cleanup.

        Args:
            file_path: Path to temporary file
        """
        temp_file = Path(file_path)
        if temp_file not in self.state["temp_files"]:
            self.state["temp_files"].append(temp_file)
            self.logger.debug(f"Registered temp file: {temp_file}")

    def cleanup_environment(self, preserve_artifacts: bool = True) -> bool:
        """Perform comprehensive environment cleanup.

        Args:
            preserve_artifacts: Whether to preserve test artifacts

        Returns:
            True if cleanup was successful, False otherwise
        """
        cleanup_success = True

        try:
            # Stop any running Streamlit processes
            if self._cleanup_streamlit_processes():
                self.logger.info("Streamlit processes cleaned up")
            else:
                cleanup_success = False

            # Clean up registered processes
            if self._cleanup_processes():
                self.logger.info("Registered processes cleaned up")
            else:
                cleanup_success = False

            # Clean up temporary files
            if self._cleanup_temp_files():
                self.logger.info("Temporary files cleaned up")
            else:
                cleanup_success = False

            # Clean up registered paths
            if self._cleanup_paths():
                self.logger.info("Cleanup paths processed")
            else:
                cleanup_success = False

            # Clean up general test files
            if self.state["temp_files"]:
                cleanup_test_files(
                    [str(path) for path in self.state["temp_files"]]
                )
                self.logger.info("General test files cleaned up")

            # Handle artifacts
            if not preserve_artifacts and self.state["artifacts_dir"].exists():
                try:
                    shutil.rmtree(self.state["artifacts_dir"])
                    self.logger.info(
                        f"Test artifacts removed: "
                        f"{self.state['artifacts_dir']}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to remove artifacts: {e}")
                    cleanup_success = False

            if cleanup_success:
                self.logger.info(
                    f"Environment cleanup completed for {self.test_name}"
                )
            else:
                self.logger.warning(
                    f"Environment cleanup had issues for {self.test_name}"
                )

            return cleanup_success

        except Exception as e:
            self.logger.error(f"Environment cleanup failed: {e}")
            return False

    def _cleanup_streamlit_processes(self) -> bool:
        """Clean up Streamlit application processes."""
        try:
            # Note: Cannot check app status without a driver
            # Proceed directly to process cleanup

            # Force kill any remaining streamlit processes
            self.logger.debug("Attempting to cleanup Streamlit processes")

            # Force kill any remaining streamlit processes
            try:
                subprocess.run(
                    ["pkill", "-f", "streamlit"],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # pkill might not be available on Windows, try taskkill
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/IM", "streamlit.exe"],
                        check=False,
                        capture_output=True,
                        timeout=10,
                    )
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass  # Neither command available, continue

            return True

        except Exception as e:
            self.logger.error(f"Failed to cleanup Streamlit processes: {e}")
            return False

    def _cleanup_processes(self) -> bool:
        """Clean up registered processes."""
        success = True
        for pid in self.state["running_processes"]:
            try:
                # First try graceful termination
                subprocess.run(["kill", str(pid)], check=False, timeout=5)
                time.sleep(1)

                # Then force kill if needed
                subprocess.run(
                    ["kill", "-9", str(pid)], check=False, timeout=5
                )

            except Exception as e:
                self.logger.error(f"Failed to cleanup process {pid}: {e}")
                success = False

        return success

    def _cleanup_temp_files(self) -> bool:
        """Clean up temporary files."""
        success = True
        for temp_file in self.state["temp_files"]:
            try:
                if temp_file.exists():
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                    self.logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                self.logger.error(
                    f"Failed to cleanup temp file {temp_file}: {e}"
                )
                success = False

        return success

    def _cleanup_paths(self) -> bool:
        """Clean up registered paths."""
        success = True
        for cleanup_path in self.state["cleanup_paths"]:
            try:
                if cleanup_path.exists():
                    if cleanup_path.is_file():
                        cleanup_path.unlink()
                    elif cleanup_path.is_dir():
                        shutil.rmtree(cleanup_path)
                    self.logger.debug(f"Cleaned up path: {cleanup_path}")
            except Exception as e:
                self.logger.error(
                    f"Failed to cleanup path {cleanup_path}: {e}"
                )
                success = False

        return success

    def save_test_state(
        self, additional_info: dict[str, Any] | None = None
    ) -> dict[str, Path] | None:
        """Save current test state for debugging or restoration.

        Args:
            additional_info: Additional information to save with state

        Returns:
            Dictionary mapping artifact names to saved file paths, or None if
            save failed
        """
        try:
            state_info = {
                "test_name": self.test_name,
                "state": self.state,
                "additional_info": additional_info or {},
                "timestamp": get_current_timestamp(),
            }

            return save_test_artifacts(
                state_info, str(self.state["artifacts_dir"]), self.test_name
            )

        except Exception as e:
            self.logger.error(f"Failed to save test state: {e}")
            return None


def create_clean_test_environment(test_name: str) -> TestEnvironmentManager:
    """Create a clean test environment with automatic cleanup registration.

    Args:
        test_name: Name of the test for environment isolation

    Returns:
        Configured TestEnvironmentManager instance
    """
    manager = TestEnvironmentManager(test_name)
    manager.create_clean_environment()
    return manager


def cleanup_test_environment(
    manager: TestEnvironmentManager, preserve_artifacts: bool = True
) -> bool:
    """Clean up test environment using the provided manager.

    Args:
        manager: TestEnvironmentManager instance
        preserve_artifacts: Whether to preserve test artifacts

    Returns:
        True if cleanup was successful
    """
    return manager.cleanup_environment(preserve_artifacts)


def setup_crackseg_test_state(
    driver: WebDriver, config_file: str | None = None, timeout: float = 30.0
) -> bool:
    """Setup CrackSeg application in a clean test state.

    Args:
        driver: WebDriver instance
        config_file: Optional configuration file to load
        timeout: Timeout for setup operations

    Returns:
        True if setup was successful
    """
    try:
        from ..utils.streamlit import wait_for_streamlit_ready

        # Navigate to home page and wait for ready state
        driver.refresh()
        time.sleep(2)
        return wait_for_streamlit_ready(driver, int(timeout))

    except Exception as e:
        logger.error(f"Failed to setup CrackSeg test state: {e}")
        return False


def restore_default_state(driver: WebDriver) -> bool:
    """Restore CrackSeg application to default state.

    Args:
        driver: WebDriver instance

    Returns:
        True if restoration was successful
    """
    try:
        # Navigate to home page
        driver.refresh()
        time.sleep(2)

        # Wait for Streamlit to be ready
        from ..utils.streamlit import wait_for_streamlit_ready

        wait_for_streamlit_ready(driver, 10)

        logger.info("Default state restored successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to restore default state: {e}")
        return False


def manage_test_artifacts(
    test_name: str,
    driver: WebDriver | None = None,
    preserve_screenshots: bool = True,
    preserve_logs: bool = True,
) -> Path | None:
    """Manage test artifacts with configurable preservation options.

    Args:
        test_name: Name of the test
        driver: Optional WebDriver for screenshot capture
        preserve_screenshots: Whether to save screenshots
        preserve_logs: Whether to save logs

    Returns:
        Path to artifacts directory, or None if management failed
    """
    try:
        manager = TestEnvironmentManager(test_name)
        artifacts_dir = manager.state["artifacts_dir"]

        if preserve_screenshots and driver:
            screenshot_path = (
                artifacts_dir / "screenshots" / f"{test_name}_final.png"
            )
            try:
                driver.save_screenshot(str(screenshot_path))
                logger.info(f"Screenshot saved: {screenshot_path}")
            except Exception as e:
                logger.warning(f"Failed to save screenshot: {e}")

        if preserve_logs:
            # Save any available logs
            log_dir = artifacts_dir / "logs"
            log_dir.mkdir(exist_ok=True)

            # This could be extended to collect application logs
            logger.info(f"Log directory prepared: {log_dir}")

        return artifacts_dir

    except Exception as e:
        logger.error(f"Failed to manage test artifacts: {e}")
        return None
