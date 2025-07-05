"""Unit tests for browser configuration manager.

Tests browser configuration management, matrix generation, and parallel
execution capabilities.
"""

from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from selenium.webdriver.remote.webdriver import WebDriver

from tests.e2e.config.browser_capabilities import (
    ChromeCapabilities,
)
from tests.e2e.config.browser_config_manager import (
    BrowserConfigManager,
    BrowserMatrix,
    ParallelExecutionConfig,
)
from tests.e2e.drivers.config import DriverConfig


class TestParallelExecutionConfig:
    """Test suite for parallel execution configuration."""

    def test_default_configuration(self) -> None:
        """Test default parallel execution configuration."""
        config = ParallelExecutionConfig()

        assert config.max_workers == 3
        assert config.execution_mode == "parallel"
        assert config.timeout_per_browser == 300.0
        assert config.browser_pool_size == 5
        assert config.cleanup_on_failure is True

    def test_configuration_validation_valid(self) -> None:
        """Test validation with valid configuration."""
        config = ParallelExecutionConfig(
            max_workers=5, timeout_per_browser=600.0, browser_pool_size=10
        )

        # Should not raise any exceptions
        config.validate()

    def test_configuration_validation_invalid_max_workers(self) -> None:
        """Test validation with invalid max_workers."""
        config = ParallelExecutionConfig(max_workers=0)

        with pytest.raises(ValueError, match="max_workers must be positive"):
            config.validate()

    def test_configuration_validation_invalid_timeout(self) -> None:
        """Test validation with invalid timeout."""
        config = ParallelExecutionConfig(timeout_per_browser=-1.0)

        with pytest.raises(
            ValueError, match="timeout_per_browser must be positive"
        ):
            config.validate()

    def test_configuration_validation_invalid_pool_size(self) -> None:
        """Test validation with invalid pool size."""
        config = ParallelExecutionConfig(browser_pool_size=0)

        with pytest.raises(
            ValueError, match="browser_pool_size must be positive"
        ):
            config.validate()


class TestBrowserMatrix:
    """Test suite for browser matrix configuration."""

    def test_default_configuration(self) -> None:
        """Test default browser matrix configuration."""
        matrix = BrowserMatrix()

        assert "chrome" in matrix.browsers
        assert "firefox" in matrix.browsers
        assert "edge" in matrix.browsers
        assert matrix.headless_modes == [True, False]
        assert (1920, 1080) in matrix.window_sizes
        assert (1366, 768) in matrix.window_sizes

    def test_generate_configurations_desktop_only(self) -> None:
        """Test configuration generation for desktop browsers only."""
        matrix = BrowserMatrix(
            browsers=["chrome", "firefox"],
            headless_modes=[True],
            window_sizes=[(1920, 1080)],
            include_mobile=False,
        )

        configurations = matrix.generate_configurations()

        # Should generate 2 browsers × 1 headless mode × 1 window size
        # = 2 configs
        assert len(configurations) == 2

        for config in configurations:
            assert config["browser"] in ["chrome", "firefox"]
            assert config["headless"] is True
            assert config["window_width"] == 1920
            assert config["window_height"] == 1080
            assert config["mobile"] is False

    def test_generate_configurations_with_mobile(self) -> None:
        """Test configuration generation including mobile devices."""
        matrix = BrowserMatrix(
            browsers=["chrome"],
            headless_modes=[True],
            window_sizes=[(1920, 1080)],
            mobile_devices=["iPhone 12", "iPad"],
            include_mobile=True,
        )

        configurations = matrix.generate_configurations()

        # Should generate: 1 desktop config + 2 mobile configs = 3 total
        assert len(configurations) == 3

        desktop_configs = [c for c in configurations if not c["mobile"]]
        mobile_configs = [c for c in configurations if c["mobile"]]

        assert len(desktop_configs) == 1
        assert len(mobile_configs) == 2

        for mobile_config in mobile_configs:
            assert mobile_config["browser"] == "chrome"
            assert mobile_config["headless"] is True
            assert mobile_config["device_name"] in ["iPhone 12", "iPad"]

    def test_generate_configurations_exclude_browsers(self) -> None:
        """Test configuration generation with browser exclusion."""
        matrix = BrowserMatrix(
            browsers=["chrome", "firefox", "edge"],
            exclude_browsers=["firefox"],
            headless_modes=[True],
            window_sizes=[(1920, 1080)],
        )

        configurations = matrix.generate_configurations()

        # Should only generate configs for chrome and edge
        browsers_used = {config["browser"] for config in configurations}
        assert browsers_used == {"chrome", "edge"}
        assert "firefox" not in browsers_used

    def test_get_configuration_count(self) -> None:
        """Test configuration count calculation."""
        matrix = BrowserMatrix(
            browsers=["chrome", "firefox"],
            headless_modes=[True, False],
            window_sizes=[(1920, 1080), (1366, 768)],
        )

        # 2 browsers × 2 headless modes × 2 window sizes = 8 configurations
        assert matrix.get_configuration_count() == 8


class TestBrowserConfigManager:
    """Test suite for browser configuration manager."""

    def test_initialization_default(self) -> None:
        """Test manager initialization with default settings."""
        manager = BrowserConfigManager()

        assert manager.base_driver_config is not None
        assert manager.parallel_config is not None
        assert isinstance(manager._browser_capabilities, dict)
        assert "chrome" in manager._browser_capabilities
        assert "firefox" in manager._browser_capabilities
        assert "edge" in manager._browser_capabilities

    def test_initialization_custom_config(self) -> None:
        """Test manager initialization with custom configurations."""
        driver_config = DriverConfig(headless=False, browser="firefox")
        parallel_config = ParallelExecutionConfig(max_workers=5)

        manager = BrowserConfigManager(
            base_driver_config=driver_config, parallel_config=parallel_config
        )

        assert manager.base_driver_config.browser == "firefox"
        assert manager.base_driver_config.headless is False
        assert manager.parallel_config.max_workers == 5

    def test_set_browser_capabilities(self) -> None:
        """Test setting custom browser capabilities."""
        manager = BrowserConfigManager()

        custom_chrome = ChromeCapabilities(
            window_width=1280, window_height=720, disable_gpu=False
        )

        manager.set_browser_capabilities("chrome", custom_chrome)

        retrieved = manager.get_browser_capabilities("chrome")
        assert retrieved.window_width == 1280
        assert retrieved.window_height == 720
        assert retrieved.disable_gpu is False

    def test_get_browser_capabilities_unsupported(self) -> None:
        """Test getting capabilities for unsupported browser."""
        manager = BrowserConfigManager()

        with pytest.raises(ValueError, match="Browser safari not supported"):
            manager.get_browser_capabilities("safari")

    @patch("tests.e2e.config.browser_config_manager.DriverFactory")
    def test_create_driver_for_browser(
        self, mock_driver_factory: MagicMock
    ) -> None:
        """Test driver creation for specific browser."""
        # Setup mock
        mock_driver = MagicMock(spec=WebDriver)
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_driver.return_value = mock_driver
        mock_driver_factory.return_value = mock_factory_instance

        manager = BrowserConfigManager()

        # Create driver
        driver = manager.create_driver_for_browser("chrome")

        assert driver == mock_driver
        mock_driver_factory.assert_called_once()
        mock_factory_instance.create_driver.assert_called_once()

    @patch("tests.e2e.config.browser_config_manager.DriverFactory")
    def test_create_driver_with_capabilities_override(
        self, mock_driver_factory: MagicMock
    ) -> None:
        """Test driver creation with custom capabilities override."""
        # Setup mock
        mock_driver = MagicMock(spec=WebDriver)
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_driver.return_value = mock_driver
        mock_driver_factory.return_value = mock_factory_instance

        manager = BrowserConfigManager()

        # Custom capabilities
        custom_capabilities = ChromeCapabilities(
            window_width=800, window_height=600, headless_mode="disabled"
        )

        # Create driver with override
        driver = manager.create_driver_for_browser(
            "chrome", custom_capabilities
        )

        assert driver == mock_driver
        mock_driver_factory.assert_called_once()

        # Verify the driver config passed to factory has overridden values
        factory_call_args = mock_driver_factory.call_args[0]
        driver_config = factory_call_args[0]
        assert driver_config.window_size == (800, 600)
        assert driver_config.headless is False  # "disabled" -> False

    def test_config_to_capabilities_desktop(self) -> None:
        """Test conversion of matrix entry to desktop capabilities."""
        manager = BrowserConfigManager()

        config_entry = {
            "browser": "chrome",
            "headless": True,
            "window_width": 1366,
            "window_height": 768,
            "mobile": False,
        }

        capabilities = manager._config_to_capabilities(config_entry)

        assert isinstance(capabilities, ChromeCapabilities)
        assert capabilities.headless_mode == "new"
        assert capabilities.window_width == 1366
        assert capabilities.window_height == 768

    def test_config_to_capabilities_mobile(self) -> None:
        """Test conversion of matrix entry to mobile capabilities."""
        manager = BrowserConfigManager()

        config_entry = {
            "browser": "chrome",
            "headless": True,
            "mobile": True,
            "device_name": "iPhone 12",
        }

        capabilities = manager._config_to_capabilities(config_entry)

        from tests.e2e.config.browser_capabilities import (
            MobileBrowserCapabilities,
        )

        assert isinstance(capabilities, MobileBrowserCapabilities)
        assert capabilities.device_name == "iPhone 12"
        assert capabilities.headless_mode == "new"

    def test_config_to_capabilities_unsupported_browser(self) -> None:
        """Test conversion with unsupported browser."""
        manager = BrowserConfigManager()

        config_entry = {
            "browser": "safari",
            "headless": True,
            "window_width": 1920,
            "window_height": 1080,
            "mobile": False,
        }

        with pytest.raises(ValueError, match="Unsupported browser: safari"):
            manager._config_to_capabilities(config_entry)

    @patch("tests.e2e.config.browser_config_manager.DriverFactory")
    def test_create_drivers_from_matrix_sequential(
        self, mock_driver_factory: MagicMock
    ) -> None:
        """Test sequential driver creation from browser matrix."""
        # Setup mock
        mock_driver = MagicMock(spec=WebDriver)
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_driver.return_value = mock_driver
        mock_driver_factory.return_value = mock_factory_instance

        manager = BrowserConfigManager()
        manager.parallel_config.execution_mode = "sequential"

        matrix = BrowserMatrix(
            browsers=["chrome"],
            headless_modes=[True],
            window_sizes=[(1920, 1080)],
        )

        drivers = manager.create_drivers_from_matrix(matrix)

        assert len(drivers) == 1
        config_id = list(drivers.keys())[0]
        assert config_id.startswith("config_0_chrome")
        assert drivers[config_id] == mock_driver

    def test_cleanup_drivers_success(self) -> None:
        """Test successful driver cleanup."""
        # Create mock drivers
        mock_drivers = {
            "chrome_1920x1080": Mock(spec=WebDriver),
            "firefox_1366x768": Mock(spec=WebDriver),
        }

        # Cast to the expected type for the manager
        typed_drivers = cast(dict[str, WebDriver], mock_drivers)

        manager = BrowserConfigManager()
        manager.cleanup_drivers(typed_drivers)

        # Verify quit was called on each driver
        for driver in mock_drivers.values():
            driver.quit.assert_called_once()

    def test_cleanup_drivers_with_exceptions(self) -> None:
        """Test driver cleanup with exceptions."""
        # Create mock drivers where one fails to quit
        mock_driver1 = Mock(spec=WebDriver)
        mock_driver2 = Mock(spec=WebDriver)
        mock_driver2.quit.side_effect = Exception("Driver quit failed")

        mock_drivers = {
            "chrome_1920x1080": mock_driver1,
            "firefox_1366x768": mock_driver2,
        }

        # Cast to the expected type
        typed_drivers = cast(dict[str, WebDriver], mock_drivers)

        manager = BrowserConfigManager()

        # Should not raise exception even if individual driver cleanup fails
        manager.cleanup_drivers(typed_drivers)

        # Verify quit was called on both drivers
        mock_driver1.quit.assert_called_once()
        mock_driver2.quit.assert_called_once()

    @patch("tests.e2e.config.browser_config_manager.DriverFactory")
    def test_execute_cross_browser_test_sequential(
        self, mock_driver_factory: MagicMock
    ) -> None:
        """Test cross-browser test execution in sequential mode."""
        # Setup mock
        mock_driver = MagicMock(spec=WebDriver)
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_driver.return_value = mock_driver
        mock_driver_factory.return_value = mock_factory_instance

        manager = BrowserConfigManager()
        manager.parallel_config.execution_mode = "sequential"

        matrix = BrowserMatrix(
            browsers=["chrome"],
            headless_modes=[True],
            window_sizes=[(1920, 1080)],
        )

        # Test function
        def test_function(driver: WebDriver, config_id: str) -> dict[str, str]:
            return {"driver_title": driver.title, "config": config_id}

        mock_driver.title = "Test Page"

        results = manager.execute_cross_browser_test(matrix, test_function)

        assert len(results) == 1
        config_id = list(results.keys())[0]
        result = results[config_id]
        assert result["driver_title"] == "Test Page"
        assert result["config"] == config_id

        # Verify cleanup was called
        mock_driver.quit.assert_called_once()
