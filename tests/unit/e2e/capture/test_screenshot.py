"""Unit tests for screenshot capture functionality."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from selenium.common.exceptions import WebDriverException

# Import from the actual module
from tests.e2e.capture.screenshot import (
    ScreenshotCapture,
    ScreenshotCaptureMixin,
    ScreenshotConfig,
)


class TestScreenshotConfig:
    """Test suite for ScreenshotConfig."""

    def test_default_configuration(self) -> None:
        """Test default screenshot configuration values."""
        config = ScreenshotConfig()

        assert config.enabled is True
        assert config.capture_on_failure is True
        assert config.capture_on_assertion is False
        assert config.filename_prefix == "screenshot"
        assert config.quality == 95
        assert config.max_screenshots_per_test == 10
        assert config.cleanup_on_success is False

    def test_custom_configuration(self) -> None:
        """Test custom screenshot configuration."""
        custom_dir = Path("/tmp/custom-screenshots")
        config = ScreenshotConfig(
            enabled=False,
            artifacts_dir=custom_dir,
            filename_prefix="test",
            quality=75,
            capture_on_assertion=True,
        )

        assert config.enabled is False
        assert config.artifacts_dir == custom_dir
        assert config.filename_prefix == "test"
        assert config.quality == 75
        assert config.capture_on_assertion is True


class TestScreenshotCapture:
    """Test suite for ScreenshotCapture functionality."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        import tempfile

        tmpdir = tempfile.mkdtemp()
        return Path(tmpdir)

    @pytest.fixture
    def screenshot_config(self, temp_dir: Path) -> ScreenshotConfig:
        """Create test screenshot configuration."""
        return ScreenshotConfig(
            artifacts_dir=temp_dir,
            cleanup_on_success=False,  # Disable cleanup for testing
        )

    @pytest.fixture
    def mock_driver(self) -> MagicMock:
        """Create mock WebDriver."""
        driver = MagicMock()
        driver.save_screenshot.return_value = True
        return driver

    def test_initialization(self, screenshot_config: ScreenshotConfig) -> None:
        """Test ScreenshotCapture initialization."""
        capture = ScreenshotCapture(screenshot_config)

        assert capture.config == screenshot_config
        assert screenshot_config.artifacts_dir.exists()

    def test_capture_screenshot_creates_file(
        self, screenshot_config: ScreenshotConfig, mock_driver: MagicMock
    ) -> None:
        """Test that capture_screenshot creates a file."""
        capture = ScreenshotCapture(screenshot_config)

        # Mock file creation
        def mock_save_screenshot(path: str) -> bool:
            Path(path).touch()  # Create the file
            return True

        mock_driver.save_screenshot.side_effect = mock_save_screenshot

        with patch("time.strftime", return_value="20240115_143052_123456"):
            result = capture.capture_screenshot(
                mock_driver, name="test", context="test context"
            )

        assert result is not None
        assert result.name == "screenshot_test_20240115_143052_123456.png"
        assert result.exists()  # Verify file was created
        mock_driver.save_screenshot.assert_called_once()

    def test_capture_screenshot_disabled(
        self, temp_dir: Path, mock_driver: MagicMock
    ) -> None:
        """Test screenshot capture when disabled."""
        config = ScreenshotConfig(artifacts_dir=temp_dir, enabled=False)
        capture = ScreenshotCapture(config)

        result = capture.capture_screenshot(mock_driver)

        assert result is None
        mock_driver.save_screenshot.assert_not_called()

    def test_capture_failure_screenshot(
        self, screenshot_config: ScreenshotConfig, mock_driver: MagicMock
    ) -> None:
        """Test failure screenshot capture."""
        capture = ScreenshotCapture(screenshot_config)

        # Mock file creation
        def mock_save_screenshot(path: str) -> bool:
            Path(path).touch()  # Create the file
            return True

        mock_driver.save_screenshot.side_effect = mock_save_screenshot

        with patch("time.strftime", return_value="20240115_143052_123456"):
            result = capture.capture_failure_screenshot(
                mock_driver, "test_login", ValueError("Login failed")
            )

        assert result is not None
        assert "failure_test_login" in result.name
        assert result.exists()  # Verify file was created
        mock_driver.save_screenshot.assert_called_once()

    def test_generate_filename_includes_context(
        self, screenshot_config: ScreenshotConfig
    ) -> None:
        """Test filename generation includes context."""
        capture = ScreenshotCapture(screenshot_config)

        filename = capture._generate_filename(
            "test", "login_page", "20240115_143052_123456"
        )

        assert filename == "screenshot_test_20240115_143052_123456.png"

    def test_sanitize_test_name(
        self, screenshot_config: ScreenshotConfig
    ) -> None:
        """Test test name sanitization."""
        capture = ScreenshotCapture(screenshot_config)

        filename = capture._generate_filename(
            "test/with:invalid<chars>", None, "20240115_143052_123456"
        )

        assert "/" not in filename
        assert ":" not in filename
        assert "<" not in filename
        assert ">" not in filename

    def test_error_handling(self, screenshot_config: ScreenshotConfig) -> None:
        """Test error handling during screenshot capture."""
        capture = ScreenshotCapture(screenshot_config)
        mock_driver = MagicMock()
        mock_driver.save_screenshot.side_effect = WebDriverException(
            "Driver error"
        )

        result = capture.capture_screenshot(mock_driver)

        assert result is None


class TestScreenshotCaptureMixin:
    """Test suite for ScreenshotCaptureMixin."""

    def test_mixin_initialization(self) -> None:
        """Test mixin initialization."""

        class BaseTestClass:
            """Base class for mixin testing."""

            def __init__(self, **kwargs: Any):
                # Consume any unknown kwargs
                pass

        class TestClass(ScreenshotCaptureMixin, BaseTestClass):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ScreenshotConfig(
                artifacts_dir=Path(tmpdir), cleanup_on_success=False
            )
            instance = TestClass(screenshot_config=config)

        assert hasattr(instance, "_screenshot_capture")
        assert instance._screenshot_capture.config == config

    def test_mixin_methods(self) -> None:
        """Test mixin exposes screenshot methods."""

        class BaseTestClass:
            """Base class for mixin testing."""

            def __init__(self, **kwargs: Any):
                # Consume any unknown kwargs
                pass

        class TestClass(ScreenshotCaptureMixin, BaseTestClass):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ScreenshotConfig(
                artifacts_dir=Path(tmpdir), cleanup_on_success=False
            )
            instance = TestClass(screenshot_config=config)

        # Verify methods exist and are callable
        assert hasattr(instance, "capture_screenshot")
        assert hasattr(instance, "capture_failure_screenshot")
        assert hasattr(instance, "capture_assertion_screenshot")
        assert hasattr(instance, "cleanup_test_screenshots")
        assert hasattr(instance, "get_test_screenshots")

        # Test methods are callable
        assert callable(instance.capture_screenshot)
        assert callable(instance.capture_failure_screenshot)
        assert callable(instance.capture_assertion_screenshot)
