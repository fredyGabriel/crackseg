"""Browser-specific capabilities configuration for cross-browser testing.

This module provides advanced browser capabilities configuration for
Chrome, Firefox, Edge, and mobile browsers with support for headless modes,
performance optimization, and browser-specific features.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

from ..drivers.config import BrowserType

logger = logging.getLogger(__name__)

# Type aliases for better readability
MobileDevice = Literal[
    "iPhone 12", "iPhone 13", "iPad", "Pixel 5", "Galaxy S21"
]
HeadlessMode = Literal["old", "new", "disabled"]

# Device specifications for mobile testing
MOBILE_DEVICES = {
    "iPhone 12": {
        "width": 390,
        "height": 844,
        "pixel_ratio": 3.0,
        "user_agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 "
            "Mobile/15E148 Safari/604.1"
        ),
    },
    "iPhone 13": {
        "width": 390,
        "height": 844,
        "pixel_ratio": 3.0,
        "user_agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 "
            "Mobile/15E148 Safari/604.1"
        ),
    },
    "iPad": {
        "width": 768,
        "height": 1024,
        "pixel_ratio": 2.0,
        "user_agent": (
            "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 "
            "Mobile/15E148 Safari/604.1"
        ),
    },
    "Google Pixel 5": {
        "width": 393,
        "height": 851,
        "pixel_ratio": 2.75,
        "user_agent": (
            "Mozilla/5.0 (Linux; Android 11; Pixel 5) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.91 Mobile Safari/537.36"
        ),
    },
    "Samsung Galaxy S21": {
        "width": 360,
        "height": 800,
        "pixel_ratio": 3.0,
        "user_agent": (
            "Mozilla/5.0 (Linux; Android 11; SM-G991B) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.91 Mobile Safari/537.36"
        ),
    },
}


@dataclass
class BrowserCapabilities(ABC):
    """Abstract base class for browser-specific capabilities configuration."""

    # Common configuration
    headless_mode: HeadlessMode = "new"
    window_width: int = 1920
    window_height: int = 1080

    # Performance settings
    disable_gpu: bool = True
    disable_dev_shm_usage: bool = True
    no_sandbox: bool = True
    disable_extensions: bool = True

    # Security and privacy
    disable_web_security: bool = False
    ignore_certificate_errors: bool = False

    # Debugging and logging
    enable_logging: bool = False
    log_level: Literal["INFO", "WARNING", "SEVERE"] = "INFO"
    enable_performance_logging: bool = False

    # Advanced options
    experimental_options: dict[str, Any] = field(default_factory=dict)
    additional_arguments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        self.validate()

    @abstractmethod
    def to_selenium_options(
        self,
    ) -> ChromeOptions | FirefoxOptions | EdgeOptions:
        """Convert capabilities to Selenium options object.

        Returns:
            Browser-specific Selenium options object
        """
        pass

    @abstractmethod
    def get_browser_type(self) -> BrowserType:
        """Get the browser type for this configuration.

        Returns:
            Browser type identifier
        """
        pass

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError(
                f"Window dimensions must be positive: "
                f"{self.window_width}x{self.window_height}"
            )

        if self.window_width < 320 or self.window_height < 240:
            raise ValueError(
                f"Minimum window size is 320x240, got "
                f"{self.window_width}x{self.window_height}"
            )


@dataclass
class ChromeCapabilities(BrowserCapabilities):
    """Chrome-specific capabilities configuration."""

    # Chrome-specific options
    user_data_dir: str | None = None
    disable_background_timer_throttling: bool = True
    disable_renderer_backgrounding: bool = True
    disable_backgrounding_occluded_windows: bool = True
    disable_features: list[str] = field(
        default_factory=lambda: ["TranslateUI"]
    )
    enable_features: list[str] = field(default_factory=list)

    # Mobile emulation
    mobile_emulation: dict[str, Any] = field(default_factory=dict)

    def to_selenium_options(self) -> ChromeOptions:
        """Convert to Chrome-specific Selenium options."""
        options = ChromeOptions()

        # Headless configuration
        if self.headless_mode == "new":
            options.add_argument("--headless=new")
        elif self.headless_mode == "old":
            options.add_argument("--headless")

        # Window size
        options.add_argument(
            f"--window-size={self.window_width},{self.window_height}"
        )

        # Performance and stability options
        if self.disable_gpu:
            options.add_argument("--disable-gpu")
        if self.disable_dev_shm_usage:
            options.add_argument("--disable-dev-shm-usage")
        if self.no_sandbox:
            options.add_argument("--no-sandbox")
        if self.disable_extensions:
            options.add_argument("--disable-extensions")

        # Chrome-specific performance options
        if self.disable_background_timer_throttling:
            options.add_argument("--disable-background-timer-throttling")
        if self.disable_renderer_backgrounding:
            options.add_argument("--disable-renderer-backgrounding")
        if self.disable_backgrounding_occluded_windows:
            options.add_argument("--disable-backgrounding-occluded-windows")

        # Security options
        if self.disable_web_security:
            options.add_argument("--disable-web-security")
        if self.ignore_certificate_errors:
            options.add_argument("--ignore-certificate-errors")

        # Features
        if self.disable_features:
            features = ",".join(self.disable_features)
            options.add_argument(f"--disable-features={features}")
        if self.enable_features:
            features = ",".join(self.enable_features)
            options.add_argument(f"--enable-features={features}")

        # User data directory
        if self.user_data_dir:
            options.add_argument(f"--user-data-dir={self.user_data_dir}")

        # Mobile emulation
        if self.mobile_emulation:
            options.add_experimental_option(
                "mobileEmulation", self.mobile_emulation
            )

        # Experimental options
        for key, value in self.experimental_options.items():
            options.add_experimental_option(key, value)

        # Logging
        if self.enable_logging:
            options.add_argument("--enable-logging")
            options.add_argument(f"--log-level={self.log_level.lower()}")

        if self.enable_performance_logging:
            options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        return options

    def get_browser_type(self) -> BrowserType:
        """Get browser type for Chrome."""
        return "chrome"


@dataclass
class FirefoxCapabilities(BrowserCapabilities):
    """Firefox-specific capabilities configuration."""

    # Firefox-specific options
    profile_path: str | None = None
    accept_insecure_certs: bool = False
    use_marionette: bool = True

    # Firefox preferences
    preferences: dict[str, Any] = field(default_factory=dict)

    def to_selenium_options(self) -> FirefoxOptions:
        """Convert to Firefox-specific Selenium options."""
        options = FirefoxOptions()

        # Headless configuration
        if self.headless_mode in ["new", "old"]:
            options.add_argument("--headless")

        # Window size
        options.add_argument(f"--width={self.window_width}")
        options.add_argument(f"--height={self.window_height}")

        # Security options
        if self.accept_insecure_certs:
            options.set_capability("acceptInsecureCerts", True)

        if self.ignore_certificate_errors:
            options.set_preference(
                "security.tls.insecure_fallback_hosts", True
            )

        # Marionette
        options.set_capability("marionette", self.use_marionette)

        # Performance preferences
        if self.disable_gpu:
            options.set_preference("layers.acceleration.disabled", True)

        # Set common performance preferences
        performance_prefs = {
            "browser.cache.disk.enable": False,
            "browser.cache.memory.enable": False,
            "browser.cache.offline.enable": False,
            "network.cookie.cookieBehavior": 0,
        }

        for pref, value in performance_prefs.items():
            options.set_preference(pref, value)

        # User preferences
        for pref, value in self.preferences.items():
            options.set_preference(pref, value)

        # Profile
        if self.profile_path:
            # Note: Profile path handling would require FirefoxProfile
            # This is a placeholder for future implementation
            pass

        return options

    def get_browser_type(self) -> BrowserType:
        """Get browser type for Firefox."""
        return "firefox"


@dataclass
class EdgeCapabilities(BrowserCapabilities):
    """Edge browser specific capabilities and options."""

    edge_performance_mode: bool = True
    edge_tracking_prevention: str = "strict"

    def to_selenium_options(self) -> EdgeOptions:
        """Get Edge-specific options.

        Returns:
            Configured EdgeOptions instance
        """
        options = EdgeOptions()

        # Basic Edge options
        if self.headless_mode == "new":
            options.add_argument("--headless=new")
        elif self.headless_mode == "old":
            options.add_argument("--headless")

        if self.disable_gpu:
            options.add_argument("--disable-gpu")

        if self.no_sandbox:
            options.add_argument("--no-sandbox")

        # Window size
        if self.window_width and self.window_height:
            options.add_argument(
                f"--window-size={self.window_width},{self.window_height}"
            )

        # Edge-specific performance optimizations
        if self.edge_performance_mode:
            options.add_argument("--enable-features=VaapiVideoDecoder")
            options.add_argument("--disable-features=VizDisplayCompositor")

        # Privacy and security
        options.add_argument(
            f"--tracking-prevention={self.edge_tracking_prevention}"
        )

        # Additional Edge arguments
        for arg in self.additional_arguments:
            options.add_argument(arg)

        # Experimental options
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option(
            "excludeSwitches", ["enable-automation"]
        )

        return options

    def get_browser_type(self) -> BrowserType:
        """Get the browser type identifier.

        Returns:
            Browser type as literal string
        """
        return "edge"


@dataclass
class MobileBrowserCapabilities(BrowserCapabilities):
    """Mobile device browser capabilities for responsive testing."""

    device_name: str = "iPhone 12"
    mobile_user_agent: str | None = None
    enable_touch_events: bool = True
    force_device_scale_factor: bool = True

    def __post_init__(self) -> None:
        """Initialize mobile device settings after dataclass creation."""
        super().__post_init__()

        if self.device_name not in MOBILE_DEVICES:
            msg = (
                f"Device '{self.device_name}' not supported. "
                f"Available: {list(MOBILE_DEVICES.keys())}"
            )
            raise ValueError(msg)

        device_config = MOBILE_DEVICES[self.device_name]

        # Set window dimensions from device config with proper type conversion
        self.window_width = int(device_config["width"])
        self.window_height = int(device_config["height"])

        # Set user agent if not explicitly provided
        if self.mobile_user_agent is None:
            device_user_agent = device_config.get("user_agent")
            if isinstance(device_user_agent, str):
                self.mobile_user_agent = device_user_agent
            else:
                # Fallback user agent if device config doesn't have one
                self.mobile_user_agent = (
                    "Mozilla/5.0 (Linux; Android 11; SM-G991B) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/90.0.4430.91 Mobile Safari/537.36"
                )

    def to_selenium_options(self) -> ChromeOptions:
        """Get mobile Chrome options with device emulation.

        Returns:
            Configured ChromeOptions with mobile emulation
        """
        options = ChromeOptions()

        # Basic mobile Chrome options
        if self.headless_mode == "new":
            options.add_argument("--headless=new")
        elif self.headless_mode == "old":
            options.add_argument("--headless")

        if self.disable_gpu:
            options.add_argument("--disable-gpu")

        if self.no_sandbox:
            options.add_argument("--no-sandbox")

        # Mobile-specific arguments
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")

        if self.enable_touch_events:
            options.add_argument("--touch-events=enabled")

        # Device emulation configuration
        device_config = MOBILE_DEVICES[self.device_name]
        mobile_emulation = {
            "deviceMetrics": {
                "width": device_config["width"],
                "height": device_config["height"],
                "pixelRatio": device_config["pixel_ratio"],
            },
            "userAgent": self.mobile_user_agent,
        }

        options.add_experimental_option("mobileEmulation", mobile_emulation)

        # Force device scale factor if enabled
        if self.force_device_scale_factor:
            pixel_ratio = device_config["pixel_ratio"]
            options.add_argument(f"--force-device-scale-factor={pixel_ratio}")

        # Additional arguments
        for arg in self.additional_arguments:
            options.add_argument(arg)

        return options

    def get_browser_type(self) -> BrowserType:
        """Get the browser type identifier.

        Returns:
            Browser type as literal string
        """
        return "chrome"  # Mobile capabilities use Chrome with device emulation
