"""Configuration management for WebDriver setup and management.

This module provides configuration classes for driver creation, browser
options, and integration with the existing Docker infrastructure.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .exceptions import DriverConfigurationError

# Type aliases for better readability
BrowserType = Literal["chrome", "firefox", "edge", "safari"]
DriverMethod = Literal["docker", "local", "webdriver-manager", "auto"]
WindowSize = tuple[int, int]


@dataclass(frozen=True)
class DriverConfig:
    """Configuration for WebDriver creation and management.

    This configuration supports both Docker Grid and local WebDriver setups,
    with automatic fallback capabilities and integration with existing
    Docker infrastructure.
    """

    # Browser Configuration
    browser: BrowserType = "chrome"
    browser_version: str = "latest"

    # Driver Management
    driver_method: DriverMethod = "auto"
    implicit_wait: float = 10.0
    page_load_timeout: float = 30.0
    script_timeout: float = 30.0

    # Display Configuration
    headless: bool = True
    window_size: WindowSize = (1920, 1080)

    # Docker Grid Configuration
    selenium_hub_host: str = "localhost"
    selenium_hub_port: int = 4444
    grid_timeout: float = 30.0

    # Local Driver Configuration
    enable_webdriver_manager: bool = True
    driver_cache_valid_range: int = 7  # days

    # Browser Options
    disable_gpu: bool = True
    disable_dev_shm_usage: bool = True
    no_sandbox: bool = True
    disable_extensions: bool = True

    # Debug and Logging
    enable_logging: bool = False
    log_level: str = "INFO"
    enable_performance_logging: bool = False

    # Artifact Management
    screenshot_on_failure: bool = True
    video_recording: bool = False
    artifacts_dir: Path = field(default_factory=lambda: Path("test-results"))

    # Advanced Options
    experimental_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_browser()
        self._validate_timeouts()
        self._validate_window_size()
        self._validate_docker_config()
        self._validate_paths()

    def _validate_browser(self) -> None:
        """Validate browser configuration."""
        supported_browsers: list[BrowserType] = [
            "chrome",
            "firefox",
            "edge",
            "safari",
        ]
        if self.browser not in supported_browsers:
            raise DriverConfigurationError(
                config_field="browser",
                config_value=self.browser,
                expected=f"One of: {', '.join(supported_browsers)}",
                details={"supported_browsers": ", ".join(supported_browsers)},
            )

    def _validate_timeouts(self) -> None:
        """Validate timeout configurations."""
        timeouts = {
            "implicit_wait": self.implicit_wait,
            "page_load_timeout": self.page_load_timeout,
            "script_timeout": self.script_timeout,
            "grid_timeout": self.grid_timeout,
        }

        for timeout_name, timeout_value in timeouts.items():
            if timeout_value <= 0:
                raise DriverConfigurationError(
                    config_field=timeout_name,
                    config_value=str(timeout_value),
                    expected="Positive number greater than 0",
                )

    def _validate_window_size(self) -> None:
        """Validate window size configuration."""
        width, height = self.window_size

        if width <= 0 or height <= 0:
            raise DriverConfigurationError(
                config_field="window_size",
                config_value=f"{width}x{height}",
                expected="Both width and height must be positive integers",
            )

        # Reasonable minimum size
        if width < 320 or height < 240:
            raise DriverConfigurationError(
                config_field="window_size",
                config_value=f"{width}x{height}",
                expected="Minimum size: 320x240",
            )

    def _validate_docker_config(self) -> None:
        """Validate Docker configuration."""
        if not (1024 <= self.selenium_hub_port <= 65535):
            raise DriverConfigurationError(
                config_field="selenium_hub_port",
                config_value=str(self.selenium_hub_port),
                expected="Port number between 1024 and 65535",
            )

    def _validate_paths(self) -> None:
        """Validate path configurations."""
        # Ensure artifacts directory exists or can be created
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise DriverConfigurationError(
                config_field="artifacts_dir",
                config_value=str(self.artifacts_dir),
                expected="Writable directory path",
                details={"error": str(e)},
            ) from e

    @property
    def selenium_grid_url(self) -> str:
        """Get Selenium Grid WebDriver URL."""
        return (
            f"http://{self.selenium_hub_host}:{self.selenium_hub_port}/wd/hub"
        )

    @property
    def is_docker_environment(self) -> bool:
        """Check if running in Docker environment."""
        return (
            os.getenv("DOCKER_CONTAINER") == "true"
            or os.path.exists("/.dockerenv")
            or self.selenium_hub_host in ["selenium-hub", "hub"]
        )

    @property
    def chrome_options_dict(self) -> dict[str, Any]:
        """Get Chrome-specific options as dictionary."""
        options = []

        if self.headless:
            options.append("--headless=new")
        if self.disable_gpu:
            options.append("--disable-gpu")
        if self.disable_dev_shm_usage:
            options.append("--disable-dev-shm-usage")
        if self.no_sandbox:
            options.append("--no-sandbox")
        if self.disable_extensions:
            options.append("--disable-extensions")

        # Add window size
        options.append(
            f"--window-size={self.window_size[0]},{self.window_size[1]}"
        )

        # Performance and stability options
        options.extend(
            [
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-features=TranslateUI",
                "--disable-iframes-sandbox-policy",
            ]
        )

        config = {
            "args": options,
            "prefs": {
                "profile.default_content_setting_values.notifications": 2,
                "profile.default_content_settings.popups": 0,
            },
        }

        # Add experimental options
        if self.experimental_options:
            config.update(self.experimental_options)

        return config

    @property
    def firefox_options_dict(self) -> dict[str, Any]:
        """Get Firefox-specific options as dictionary."""
        options = []

        if self.headless:
            options.append("--headless")

        # Add window size
        options.extend(
            [
                f"--width={self.window_size[0]}",
                f"--height={self.window_size[1]}",
            ]
        )

        # Performance options
        options.extend(
            [
                "--disable-extensions",
                "--disable-plugins",
            ]
        )

        config = {
            "args": options,
            "prefs": {
                "dom.webnotifications.enabled": False,
                "media.volume_scale": "0.0",
                "media.autoplay.default": 5,  # Block autoplay
            },
        }

        return config

    @property
    def edge_options_dict(self) -> dict[str, Any]:
        """Get Edge-specific options as dictionary."""
        # Edge uses similar options to Chrome
        options = []

        if self.headless:
            options.append("--headless=new")
        if self.disable_gpu:
            options.append("--disable-gpu")
        if self.disable_dev_shm_usage:
            options.append("--disable-dev-shm-usage")
        if self.no_sandbox:
            options.append("--no-sandbox")
        if self.disable_extensions:
            options.append("--disable-extensions")

        options.append(
            f"--window-size={self.window_size[0]},{self.window_size[1]}"
        )

        config = {
            "args": options,
            "prefs": {
                "profile.default_content_setting_values.notifications": 2,
            },
        }

        return config

    @classmethod
    def from_environment(cls, **overrides: Any) -> "DriverConfig":
        """Create configuration from environment variables.

        Reads configuration from environment variables with TEST_ prefix,
        falling back to defaults. Supports integration with existing
        Docker environment configuration.

        Args:
            **overrides: Additional configuration overrides

        Returns:
            DriverConfig instance with environment-based configuration

        Example:
            >>> config = DriverConfig.from_environment(browser="firefox")
            >>> config.selenium_hub_host
            'selenium-hub'  # from SELENIUM_HUB_HOST env var
        """
        # Browser configuration
        browser = os.getenv("TEST_BROWSER", "chrome").lower()
        browser_version = os.getenv("TEST_BROWSER_VERSION", "latest")

        # Driver method
        driver_method = os.getenv("TEST_DRIVER_METHOD", "auto")

        # Timeouts
        implicit_wait = float(os.getenv("TEST_IMPLICIT_WAIT", "10.0"))
        page_load_timeout = float(os.getenv("TEST_PAGE_LOAD_TIMEOUT", "30.0"))
        grid_timeout = float(os.getenv("TEST_GRID_TIMEOUT", "30.0"))

        # Display
        headless = os.getenv("TEST_HEADLESS", "true").lower() == "true"
        window_size_str = os.getenv("TEST_WINDOW_SIZE", "1920,1080")
        window_size_parts = list(map(int, window_size_str.split(",")))
        if len(window_size_parts) != 2:
            window_size_parts = [1920, 1080]  # fallback to default
        window_size: WindowSize = (window_size_parts[0], window_size_parts[1])

        # Docker Grid (using existing environment variables from Task 13)
        selenium_hub_host = os.getenv("SELENIUM_HUB_HOST", "localhost")
        selenium_hub_port = int(os.getenv("SELENIUM_HUB_PORT", "4444"))

        # Artifacts
        artifacts_dir = Path(os.getenv("TEST_ARTIFACTS_PATH", "test-results"))

        # Debug
        enable_logging = os.getenv("TEST_DEBUG", "false").lower() == "true"
        video_recording = (
            os.getenv("VIDEO_RECORDING_ENABLED", "false").lower() == "true"
        )

        # Create base configuration with proper types
        base_config = cls(
            browser=browser,  # type: ignore[arg-type]
            browser_version=browser_version,
            driver_method=driver_method,  # type: ignore[arg-type]
            implicit_wait=implicit_wait,
            page_load_timeout=page_load_timeout,
            grid_timeout=grid_timeout,
            headless=headless,
            window_size=window_size,
            selenium_hub_host=selenium_hub_host,
            selenium_hub_port=selenium_hub_port,
            artifacts_dir=artifacts_dir,
            enable_logging=enable_logging,
            video_recording=video_recording,
        )

        # Apply overrides by creating a new instance if needed
        if overrides:
            # Convert current config to dict and update with overrides
            config_dict = base_config.to_dict()
            config_dict.update(overrides)

            # Handle special type conversions for overrides
            if "window_size" in overrides and isinstance(
                overrides["window_size"], str
            ):
                parts = list(map(int, overrides["window_size"].split(",")))
                if len(parts) == 2:
                    config_dict["window_size"] = (parts[0], parts[1])
            if "artifacts_dir" in overrides and isinstance(
                overrides["artifacts_dir"], str
            ):
                config_dict["artifacts_dir"] = Path(overrides["artifacts_dir"])

            # Create new instance with updated configuration
            return cls(
                browser=config_dict["browser"],
                browser_version=config_dict["browser_version"],
                driver_method=config_dict["driver_method"],
                implicit_wait=config_dict["implicit_wait"],
                page_load_timeout=config_dict["page_load_timeout"],
                script_timeout=config_dict.get("script_timeout", 30.0),
                headless=config_dict["headless"],
                window_size=config_dict["window_size"],
                selenium_hub_host=config_dict["selenium_hub_host"],
                selenium_hub_port=config_dict["selenium_hub_port"],
                grid_timeout=config_dict["grid_timeout"],
                enable_webdriver_manager=config_dict.get(
                    "enable_webdriver_manager", True
                ),
                artifacts_dir=(
                    config_dict["artifacts_dir"]
                    if isinstance(config_dict["artifacts_dir"], Path)
                    else Path(config_dict["artifacts_dir"])
                ),
                enable_logging=config_dict["enable_logging"],
                video_recording=config_dict["video_recording"],
            )

        return base_config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "browser": self.browser,
            "browser_version": self.browser_version,
            "driver_method": self.driver_method,
            "implicit_wait": self.implicit_wait,
            "page_load_timeout": self.page_load_timeout,
            "script_timeout": self.script_timeout,
            "headless": self.headless,
            "window_size": self.window_size,
            "selenium_hub_host": self.selenium_hub_host,
            "selenium_hub_port": self.selenium_hub_port,
            "grid_timeout": self.grid_timeout,
            "enable_webdriver_manager": self.enable_webdriver_manager,
            "artifacts_dir": str(self.artifacts_dir),
            "enable_logging": self.enable_logging,
            "video_recording": self.video_recording,
        }
