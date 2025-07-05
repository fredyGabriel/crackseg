"""Browser configuration manager for cross-browser testing with parallel
execution.

This module provides centralized browser configuration management with support
for browser matrices, parallel execution, and seamless integration with the
existing DriverFactory infrastructure.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

from selenium.webdriver.remote.webdriver import WebDriver

from ..drivers.config import BrowserType, DriverConfig
from ..drivers.driver_factory import DriverFactory
from ..drivers.exceptions import DriverCreationError
from .browser_capabilities import (
    BrowserCapabilities,
    ChromeCapabilities,
    EdgeCapabilities,
    FirefoxCapabilities,
    MobileBrowserCapabilities,
)

# Type aliases
ExecutionMode = Literal["sequential", "parallel", "distributed"]
BrowserMatrixEntry = dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel browser execution."""

    max_workers: int = 3
    execution_mode: ExecutionMode = "parallel"
    timeout_per_browser: float = 300.0  # 5 minutes per browser

    # Resource management
    browser_pool_size: int = 5
    cleanup_on_failure: bool = True

    # Parallel execution settings
    distribute_by_browser: bool = True
    distribute_by_test: bool = False

    def validate(self) -> None:
        """Validate parallel execution configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.max_workers <= 0:
            raise ValueError(
                f"max_workers must be positive, got {self.max_workers}"
            )

        if self.timeout_per_browser <= 0:
            raise ValueError(
                f"timeout_per_browser must be positive, got "
                f"{self.timeout_per_browser}"
            )

        if self.browser_pool_size <= 0:
            raise ValueError(
                f"browser_pool_size must be positive, got "
                f"{self.browser_pool_size}"
            )


@dataclass
class BrowserMatrix:
    """Browser testing matrix configuration."""

    browsers: list[BrowserType] = field(
        default_factory=lambda: ["chrome", "firefox", "edge"]
    )
    headless_modes: list[bool] = field(default_factory=lambda: [True, False])
    window_sizes: list[tuple[int, int]] = field(
        default_factory=lambda: [(1920, 1080), (1366, 768)]
    )
    mobile_devices: list[str] = field(default_factory=list)

    # Matrix filtering
    include_mobile: bool = False
    exclude_browsers: list[BrowserType] = field(default_factory=list)

    def generate_configurations(self) -> list[BrowserMatrixEntry]:
        """Generate all browser configuration combinations.

        Returns:
            List of browser configuration dictionaries
        """
        configurations = []

        # Filter browsers
        active_browsers = [
            browser
            for browser in self.browsers
            if browser not in self.exclude_browsers
        ]

        # Generate desktop browser configurations
        for browser in active_browsers:
            for headless in self.headless_modes:
                for width, height in self.window_sizes:
                    config = {
                        "browser": browser,
                        "headless": headless,
                        "window_width": width,
                        "window_height": height,
                        "mobile": False,
                    }
                    configurations.append(config)

        # Generate mobile configurations if enabled
        if self.include_mobile and self.mobile_devices:
            for device in self.mobile_devices:
                config = {
                    "browser": "chrome",  # Mobile emulation uses Chrome
                    "headless": True,  # Mobile emulation typically headless
                    "mobile": True,
                    "device_name": device,
                }
                configurations.append(config)

        return configurations

    def get_configuration_count(self) -> int:
        """Get total number of configurations in matrix.

        Returns:
            Total configuration count
        """
        return len(self.generate_configurations())


class BrowserConfigManager:
    """Manager for cross-browser configuration and execution.

    Provides centralized management for browser configurations, parallel
    execution support, and integration with existing DriverFactory
    infrastructure.
    """

    def __init__(
        self,
        base_driver_config: DriverConfig | None = None,
        parallel_config: ParallelExecutionConfig | None = None,
    ) -> None:
        """Initialize browser configuration manager.

        Args:
            base_driver_config: Base driver configuration for all browsers
            parallel_config: Parallel execution configuration
        """
        self.base_driver_config = base_driver_config or DriverConfig()
        self.parallel_config = parallel_config or ParallelExecutionConfig()
        self._driver_factories: dict[str, DriverFactory] = {}
        self._browser_capabilities: dict[BrowserType, BrowserCapabilities] = {}

        # Validate configurations
        self.parallel_config.validate()

        # Initialize default browser capabilities
        self._initialize_default_capabilities()

    def _initialize_default_capabilities(self) -> None:
        """Initialize default browser capabilities."""
        self._browser_capabilities = {
            "chrome": ChromeCapabilities(),
            "firefox": FirefoxCapabilities(),
            "edge": EdgeCapabilities(),
        }

    def set_browser_capabilities(
        self, browser: BrowserType, capabilities: BrowserCapabilities
    ) -> None:
        """Set custom capabilities for a specific browser.

        Args:
            browser: Browser type
            capabilities: Browser-specific capabilities
        """
        capabilities.validate()
        self._browser_capabilities[browser] = capabilities

        # Clear cached factory for this browser
        if browser in self._driver_factories:
            del self._driver_factories[browser]

    def get_browser_capabilities(
        self, browser: BrowserType
    ) -> BrowserCapabilities:
        """Get capabilities for a specific browser.

        Args:
            browser: Browser type

        Returns:
            Browser capabilities

        Raises:
            ValueError: If browser is not supported
        """
        if browser not in self._browser_capabilities:
            raise ValueError(
                f"Browser {browser} not supported. "
                f"Supported browsers: "
                f"{list(self._browser_capabilities.keys())}"
            )

        return self._browser_capabilities[browser]

    def create_driver_for_browser(
        self,
        browser: BrowserType,
        capabilities_override: BrowserCapabilities | None = None,
    ) -> WebDriver:
        """Create WebDriver for specific browser with custom capabilities.

        Args:
            browser: Browser type
            capabilities_override: Override default browser capabilities

        Returns:
            Configured WebDriver instance

        Raises:
            DriverCreationError: If driver creation fails
        """
        # Use override capabilities or default
        capabilities = capabilities_override or self.get_browser_capabilities(
            browser
        )

        # Create browser-specific driver config
        browser_config = self._create_browser_driver_config(
            browser, capabilities
        )

        # Get or create driver factory
        factory_key = f"{browser}_{id(capabilities)}"
        if factory_key not in self._driver_factories:
            self._driver_factories[factory_key] = DriverFactory(browser_config)

        factory = self._driver_factories[factory_key]

        try:
            driver = factory.create_driver()
            logger.info(f"Successfully created {browser} driver")
            return driver
        except Exception as e:
            logger.error(f"Failed to create {browser} driver: {e}")
            raise DriverCreationError(
                browser=browser, method="auto", original_error=e
            ) from e

    def _create_browser_driver_config(
        self, browser: BrowserType, capabilities: BrowserCapabilities
    ) -> DriverConfig:
        """Create DriverConfig from browser capabilities.

        Args:
            browser: Browser type
            capabilities: Browser capabilities

        Returns:
            Driver configuration
        """
        # Start with base configuration
        config_dict = self.base_driver_config.to_dict()

        # Override with browser-specific settings
        config_dict.update(
            {
                "browser": browser,
                "headless": capabilities.headless_mode != "disabled",
                "window_size": (
                    capabilities.window_width,
                    capabilities.window_height,
                ),
                "disable_gpu": capabilities.disable_gpu,
                "disable_dev_shm_usage": capabilities.disable_dev_shm_usage,
                "no_sandbox": capabilities.no_sandbox,
                "disable_extensions": capabilities.disable_extensions,
                "enable_logging": capabilities.enable_logging,
            }
        )

        return DriverConfig(**config_dict)

    def create_drivers_from_matrix(
        self, matrix: BrowserMatrix
    ) -> dict[str, WebDriver]:
        """Create drivers for all configurations in browser matrix.

        Args:
            matrix: Browser matrix configuration

        Returns:
            Dictionary mapping configuration IDs to WebDriver instances
        """
        configurations = matrix.generate_configurations()

        if self.parallel_config.execution_mode == "parallel":
            return self._create_drivers_parallel(configurations)
        else:
            return self._create_drivers_sequential(configurations)

    def _create_drivers_sequential(
        self, configurations: list[BrowserMatrixEntry]
    ) -> dict[str, WebDriver]:
        """Create drivers sequentially.

        Args:
            configurations: List of browser configurations

        Returns:
            Dictionary mapping configuration IDs to WebDriver instances
        """
        drivers = {}

        for i, config in enumerate(configurations):
            config_id = f"config_{i}_{config['browser']}"

            try:
                capabilities = self._config_to_capabilities(config)
                driver = self.create_driver_for_browser(
                    config["browser"], capabilities
                )
                drivers[config_id] = driver

            except Exception as e:
                logger.error(f"Failed to create driver for {config_id}: {e}")
                if not self.parallel_config.cleanup_on_failure:
                    raise

        return drivers

    def _create_drivers_parallel(
        self, configurations: list[BrowserMatrixEntry]
    ) -> dict[str, WebDriver]:
        """Create drivers in parallel.

        Args:
            configurations: List of browser configurations

        Returns:
            Dictionary mapping configuration IDs to WebDriver instances
        """
        drivers = {}

        with ThreadPoolExecutor(
            max_workers=self.parallel_config.max_workers
        ) as executor:
            # Submit all driver creation tasks
            future_to_config = {}

            for i, config in enumerate(configurations):
                config_id = f"config_{i}_{config['browser']}"

                future = executor.submit(
                    self._create_driver_from_config, config_id, config
                )
                future_to_config[future] = config_id

            # Collect results
            for future in as_completed(
                future_to_config,
                timeout=self.parallel_config.timeout_per_browser,
            ):
                config_id = future_to_config[future]

                try:
                    driver = future.result()
                    drivers[config_id] = driver
                except Exception as e:
                    logger.error(
                        f"Failed to create driver for {config_id}: {e}"
                    )
                    if not self.parallel_config.cleanup_on_failure:
                        raise

        return drivers

    def _create_driver_from_config(
        self, config_id: str, config: BrowserMatrixEntry
    ) -> WebDriver:
        """Create driver from configuration entry.

        Args:
            config_id: Configuration identifier
            config: Browser configuration entry

        Returns:
            WebDriver instance
        """
        capabilities = self._config_to_capabilities(config)
        return self.create_driver_for_browser(config["browser"], capabilities)

    def _config_to_capabilities(
        self, config: BrowserMatrixEntry
    ) -> BrowserCapabilities:
        """Convert configuration entry to browser capabilities.

        Args:
            config: Browser configuration entry

        Returns:
            Browser capabilities object
        """
        browser = config["browser"]

        if config.get("mobile", False):
            # Mobile configuration
            return MobileBrowserCapabilities(
                device_name=config["device_name"],
                headless_mode=(
                    "new" if config.get("headless", True) else "disabled"
                ),
            )
        else:
            # Desktop configuration
            base_capabilities = self.get_browser_capabilities(browser)

            # Create modified capabilities
            if browser == "chrome":
                capabilities = ChromeCapabilities(
                    headless_mode=(
                        "new" if config.get("headless", True) else "disabled"
                    ),
                    window_width=config.get(
                        "window_width", base_capabilities.window_width
                    ),
                    window_height=config.get(
                        "window_height", base_capabilities.window_height
                    ),
                    disable_gpu=base_capabilities.disable_gpu,
                    disable_dev_shm_usage=base_capabilities.disable_dev_shm_usage,
                    no_sandbox=base_capabilities.no_sandbox,
                    disable_extensions=base_capabilities.disable_extensions,
                )
            elif browser == "firefox":
                capabilities = FirefoxCapabilities(
                    headless_mode=(
                        "new" if config.get("headless", True) else "disabled"
                    ),
                    window_width=config.get(
                        "window_width", base_capabilities.window_width
                    ),
                    window_height=config.get(
                        "window_height", base_capabilities.window_height
                    ),
                    accept_insecure_certs=getattr(
                        base_capabilities, "accept_insecure_certs", False
                    ),
                )
            elif browser == "edge":
                capabilities = EdgeCapabilities(
                    headless_mode=(
                        "new" if config.get("headless", True) else "disabled"
                    ),
                    window_width=config.get(
                        "window_width", base_capabilities.window_width
                    ),
                    window_height=config.get(
                        "window_height", base_capabilities.window_height
                    ),
                    edge_performance_mode=getattr(
                        base_capabilities, "edge_performance_mode", True
                    ),
                    edge_tracking_prevention=getattr(
                        base_capabilities, "edge_tracking_prevention", "strict"
                    ),
                )
            else:
                raise ValueError(f"Unsupported browser: {browser}")

            return capabilities

    def cleanup_drivers(self, drivers: dict[str, WebDriver]) -> None:
        """Clean up WebDriver instances.

        Args:
            drivers: Dictionary of WebDriver instances to cleanup
        """
        for config_id, driver in drivers.items():
            try:
                driver.quit()
                logger.debug(f"Cleaned up driver for {config_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup driver {config_id}: {e}")

    def execute_cross_browser_test(
        self,
        matrix: BrowserMatrix,
        test_function: Callable[[WebDriver, str], Any],
    ) -> dict[str, Any]:
        """Execute test function across browser matrix.

        Args:
            matrix: Browser matrix configuration
            test_function: Function to execute with each driver

        Returns:
            Dictionary mapping configuration IDs to test results
        """
        drivers = self.create_drivers_from_matrix(matrix)
        results = {}

        try:
            if self.parallel_config.execution_mode == "parallel":
                results = self._execute_test_parallel(drivers, test_function)
            else:
                results = self._execute_test_sequential(drivers, test_function)
        finally:
            self.cleanup_drivers(drivers)

        return results

    def _execute_test_sequential(
        self,
        drivers: dict[str, WebDriver],
        test_function: Callable[[WebDriver, str], Any],
    ) -> dict[str, Any]:
        """Execute test function sequentially across drivers."""
        results = {}

        for config_id, driver in drivers.items():
            try:
                result = test_function(driver, config_id)
                results[config_id] = result
            except Exception as e:
                logger.error(f"Test failed for {config_id}: {e}")
                results[config_id] = {"error": str(e)}

        return results

    def _execute_test_parallel(
        self,
        drivers: dict[str, WebDriver],
        test_function: Callable[[WebDriver, str], Any],
    ) -> dict[str, Any]:
        """Execute test function in parallel across drivers."""
        results = {}

        with ThreadPoolExecutor(
            max_workers=self.parallel_config.max_workers
        ) as executor:
            future_to_config = {
                executor.submit(test_function, driver, config_id): config_id
                for config_id, driver in drivers.items()
            }

            for future in as_completed(
                future_to_config,
                timeout=self.parallel_config.timeout_per_browser,
            ):
                config_id = future_to_config[future]

                try:
                    result = future.result()
                    results[config_id] = result
                except Exception as e:
                    logger.error(f"Test failed for {config_id}: {e}")
                    results[config_id] = {"error": str(e)}

        return results
