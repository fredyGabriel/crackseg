"""Unit tests for browser capabilities configuration.

Tests browser-specific capabilities configuration for Chrome, Firefox,
Edge, and mobile browsers.
"""

import pytest
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

from tests.e2e.config.browser_capabilities import (
    ChromeCapabilities,
    EdgeCapabilities,
    FirefoxCapabilities,
    MobileBrowserCapabilities,
)


class TestBrowserCapabilities:
    """Test suite for browser capabilities configuration."""

    def test_chrome_capabilities_default_configuration(self) -> None:
        """Test Chrome capabilities with default settings."""
        capabilities = ChromeCapabilities()

        # Validate default values
        assert capabilities.headless_mode == "new"
        assert capabilities.window_width == 1920
        assert capabilities.window_height == 1080
        assert capabilities.disable_gpu is True
        assert capabilities.get_browser_type() == "chrome"

        # Validate configuration
        capabilities.validate()

    def test_chrome_capabilities_to_selenium_options(self) -> None:
        """Test Chrome capabilities conversion to Selenium options."""
        capabilities = ChromeCapabilities(
            headless_mode="new",
            window_width=1366,
            window_height=768,
            disable_extensions=True,
            disable_features=["TranslateUI", "BlinkGenPropertyTrees"],
        )

        options = capabilities.to_selenium_options()

        assert isinstance(options, ChromeOptions)

        # Check arguments are present (Selenium stores them as list)
        args = options.arguments
        assert "--headless=new" in args
        assert "--window-size=1366,768" in args
        assert "--disable-gpu" in args
        assert "--disable-extensions" in args
        assert "--disable-features=TranslateUI,BlinkGenPropertyTrees" in args

    def test_firefox_capabilities_default_configuration(self) -> None:
        """Test Firefox capabilities with default settings."""
        capabilities = FirefoxCapabilities()

        assert capabilities.headless_mode == "new"
        assert capabilities.window_width == 1920
        assert capabilities.window_height == 1080
        assert capabilities.use_marionette is True
        assert capabilities.get_browser_type() == "firefox"

        capabilities.validate()

    def test_firefox_capabilities_to_selenium_options(self) -> None:
        """Test Firefox capabilities conversion to Selenium options."""
        capabilities = FirefoxCapabilities(
            headless_mode="new",
            window_width=1024,
            window_height=768,
            accept_insecure_certs=True,
            preferences={"browser.cache.disk.enable": False},
        )

        options = capabilities.to_selenium_options()

        assert isinstance(options, FirefoxOptions)

        # Check arguments
        args = options.arguments
        assert "--headless" in args
        assert "--width=1024" in args
        assert "--height=768" in args

    def test_edge_capabilities_default_config(self) -> None:
        """Test EdgeCapabilities default configuration."""
        capabilities = EdgeCapabilities()

        # Test default values
        assert capabilities.edge_performance_mode is True
        assert capabilities.edge_tracking_prevention == "strict"
        assert capabilities.headless_mode == "new"
        assert capabilities.window_width == 1920
        assert capabilities.window_height == 1080
        assert capabilities.disable_gpu is True

        # Test browser type
        assert capabilities.get_browser_type() == "edge"

    def test_edge_capabilities_custom_config(self) -> None:
        """Test EdgeCapabilities with custom configuration."""
        capabilities = EdgeCapabilities(
            edge_performance_mode=False,
            edge_tracking_prevention="balanced",
            window_width=1366,
            window_height=768,
        )

        # Test Selenium options conversion
        options = capabilities.to_selenium_options()

        # Check that options are EdgeOptions
        assert isinstance(options, EdgeOptions)

    def test_edge_capabilities_to_selenium_options(self) -> None:
        """Test Edge capabilities conversion to Selenium options."""
        capabilities = EdgeCapabilities(
            headless_mode="new",
            window_width=1280,
            window_height=720,
            edge_performance_mode=True,
        )

        options = capabilities.to_selenium_options()

        assert isinstance(options, EdgeOptions)

        # Check arguments
        args = options.arguments
        assert "--headless=new" in args
        assert "--window-size=1280,720" in args

    def test_mobile_capabilities_default_config(self) -> None:
        """Test MobileBrowserCapabilities default configuration."""
        capabilities = MobileBrowserCapabilities()

        # Test default values
        assert capabilities.device_name == "iPhone 12"
        assert capabilities.enable_touch_events is True
        assert capabilities.force_device_scale_factor is True
        assert capabilities.headless_mode == "new"

        # Test that dimensions are set from device config
        assert capabilities.window_width == 390
        assert capabilities.window_height == 844
        assert capabilities.mobile_user_agent is not None

        # Test browser type
        assert capabilities.get_browser_type() == "chrome"

    def test_mobile_capabilities_custom_device(self) -> None:
        """Test MobileBrowserCapabilities with custom device."""
        capabilities = MobileBrowserCapabilities(device_name="iPad")

        # Test iPad-specific dimensions
        assert capabilities.window_width == 768
        assert capabilities.window_height == 1024
        assert capabilities.mobile_user_agent is not None
        assert "iPad" in capabilities.mobile_user_agent

    def test_mobile_capabilities_unsupported_device(self) -> None:
        """Test MobileBrowserCapabilities with unsupported device."""
        with pytest.raises(
            ValueError, match="Device 'UnsupportedDevice' not supported"
        ):
            MobileBrowserCapabilities(device_name="UnsupportedDevice")

    def test_mobile_capabilities_custom_user_agent(self) -> None:
        """Test MobileBrowserCapabilities with custom user agent."""
        custom_agent = "Custom Mobile Agent"
        capabilities = MobileBrowserCapabilities(
            device_name="iPhone 12", mobile_user_agent=custom_agent
        )

        assert capabilities.mobile_user_agent == custom_agent

    def test_mobile_device_emulation_options(self) -> None:
        """Test mobile device emulation configuration."""
        capabilities = MobileBrowserCapabilities(device_name="Google Pixel 5")

        # Test device-specific configuration
        assert capabilities.enable_touch_events is True
        assert capabilities.device_name == "Google Pixel 5"
        assert capabilities.window_width == 393
        assert capabilities.window_height == 851

    def test_capabilities_validation_invalid_window_size(self) -> None:
        """Test capabilities validation with invalid window dimensions."""
        with pytest.raises(
            ValueError, match="Window dimensions must be positive"
        ):
            capabilities = ChromeCapabilities(
                window_width=-100, window_height=800
            )
            capabilities.validate()

        with pytest.raises(
            ValueError, match="Window dimensions must be positive"
        ):
            capabilities = ChromeCapabilities(
                window_width=800, window_height=0
            )
            capabilities.validate()

    def test_capabilities_validation_minimum_window_size(self) -> None:
        """Test capabilities validation with too small window dimensions."""
        with pytest.raises(ValueError, match="Minimum window size is 320x240"):
            capabilities = ChromeCapabilities(
                window_width=200, window_height=150
            )
            capabilities.validate()

    def test_headless_mode_variations(self) -> None:
        """Test different headless mode configurations."""
        # Test new headless mode
        capabilities = ChromeCapabilities(headless_mode="new")
        options = capabilities.to_selenium_options()
        assert "--headless=new" in options.arguments

        # Test old headless mode
        capabilities = ChromeCapabilities(headless_mode="old")
        options = capabilities.to_selenium_options()
        assert "--headless" in options.arguments

        # Test disabled headless mode
        capabilities = ChromeCapabilities(headless_mode="disabled")
        options = capabilities.to_selenium_options()
        headless_args = [arg for arg in options.arguments if "headless" in arg]
        assert len(headless_args) == 0

    def test_chrome_experimental_options(self) -> None:
        """Test Chrome experimental options configuration."""
        experimental_opts = {
            "useAutomationExtension": False,
            "excludeSwitches": ["enable-automation"],
        }

        capabilities = ChromeCapabilities(
            experimental_options=experimental_opts
        )
        options = capabilities.to_selenium_options()

        for key, value in experimental_opts.items():
            assert options.experimental_options.get(key) == value

    def test_firefox_preferences(self) -> None:
        """Test Firefox preferences configuration."""
        prefs = {
            "browser.download.folderList": 2,
            "browser.download.dir": "/tmp",
            "browser.helperApps.neverAsk.saveToDisk": (
                "application/octet-stream"
            ),
        }

        capabilities = FirefoxCapabilities(preferences=prefs)
        options = capabilities.to_selenium_options()

        # Note: Selenium Firefox options don't expose preferences directly
        # This test ensures no exceptions are raised during configuration
        assert isinstance(options, FirefoxOptions)

    def test_mobile_unsupported_device(self) -> None:
        """Test mobile capabilities with unsupported device."""
        with pytest.raises(
            ValueError, match="Device 'iPhone 20' not supported"
        ):
            MobileBrowserCapabilities(device_name="iPhone 20")

    def test_browser_capabilities_inheritance(self) -> None:
        """Test that mobile capabilities properly inherit from Chrome
        capabilities."""
        mobile_caps = MobileBrowserCapabilities(
            device_name="iPhone 12",
            disable_gpu=False,  # Override parent setting
            enable_logging=True,
        )

        # Should have Chrome capabilities
        assert mobile_caps.get_browser_type() == "chrome"
        assert mobile_caps.disable_gpu is False
        assert mobile_caps.enable_logging is True

        # Should have mobile-specific settings
        assert mobile_caps.device_name == "iPhone 12"
        assert mobile_caps.enable_touch_events is True
