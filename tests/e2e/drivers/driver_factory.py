"""
WebDriver factory for hybrid driver creation with cross-platform
support. This module implements the factory pattern for WebDriver
creation, supporting both Docker Grid and local WebDriver setups with
browser-specific optimizations.
"""

import logging
from typing import Protocol

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.remote.webdriver import WebDriver

from .config import BrowserType, DriverConfig
from .exceptions import DriverCreationError, DriverNotSupportedError


# Protocol for WebDriver manager to enable dependency injection
class WebDriverManagerProtocol(Protocol):
    """Protocol for WebDriver manager implementations."""

    def install(self) -> str:
        """Install driver and return path to executable."""
        ...


logger = logging.getLogger(__name__)


class DriverFactory:
    """
    Factory for creating WebDriver instances with browser-specific
    configurations. This factory supports both Docker Grid and local
    WebDriver creation, automatic method selection, and comprehensive
    error handling with fallback capabilities.
    """

    def __init__(self, config: DriverConfig) -> None:
        """
        Initialize driver factory with configuration. Args: config: Driver
        configuration containing browser settings, timeouts, and
        infrastructure details
        """
        self.config = config
        self._webdriver_managers: dict[str, WebDriverManagerProtocol] = {}

    def create_driver(self, method: str = "auto") -> WebDriver:
        """Create WebDriver instance using specified method.

        Args:
            method: Driver creation method ('docker', 'local',
            'webdriver-manager', 'auto')

        Returns:
            Configured WebDriver instance ready for testing

        Raises:
            DriverCreationError: When driver creation fails
            DriverNotSupportedError: When browser is not supported

        Example:
            >>> factory = DriverFactory(config)
            >>> driver = factory.create_driver("docker")
            >>> driver.get("http://localhost:8501")
        """
        if method == "auto":
            method = self._determine_best_method()

        logger.info(
            f"Creating {self.config.browser} driver using {method} method"
        )

        try:
            if method == "docker":
                return self._create_docker_driver()
            elif method == "local":
                return self._create_local_driver()
            elif method == "webdriver-manager":
                return self._create_webdriver_manager_driver()
            else:
                raise DriverCreationError(
                    browser=self.config.browser,
                    method=method,
                    details={"error": f"Unknown method: {method}"},
                )

        except Exception as e:
            if isinstance(e, DriverCreationError):
                raise
            raise DriverCreationError(
                browser=self.config.browser, method=method, original_error=e
            ) from e

    def _determine_best_method(self) -> str:
        """
        Determine the best driver creation method based on environment.
        Returns: Best method name ('docker', 'local', or 'webdriver-manager')
        """
        # Force specific method if configured
        if self.config.driver_method != "auto":
            return self.config.driver_method

        # Prefer Docker if infrastructure is available
        if self._is_docker_grid_available():
            logger.debug("Docker Grid available, using docker method")
            return "docker"

        # Fallback to WebDriverManager for automatic driver management
        if self.config.enable_webdriver_manager:
            logger.debug("Using WebDriverManager for automatic driver setup")
            return "webdriver-manager"

        # Last resort: local driver (requires manual setup)
        logger.debug("Using local driver method")
        return "local"

    def _create_docker_driver(self) -> WebDriver:
        """
        Create WebDriver using Docker Grid infrastructure. Returns: Remote
        WebDriver connected to Selenium Grid Raises: DriverCreationError: When
        Docker Grid connection fails
        """
        logger.debug(f"Creating Docker Grid driver for {self.config.browser}")

        # Get browser options (modern Selenium approach)
        options = self._get_browser_options()

        # Create remote driver using modern syntax
        try:
            driver = webdriver.Remote(
                command_executor=self.config.selenium_grid_url,
                options=options,
            )

            # Configure timeouts
            self._configure_driver_timeouts(driver)

            logger.info(
                f"Successfully created Docker Grid {self.config.browser} "
                f"driver"
            )
            return driver

        except Exception as e:
            raise DriverCreationError(
                browser=self.config.browser,
                method="docker",
                original_error=e,
                details={
                    "grid_url": self.config.selenium_grid_url,
                    "browser": self.config.browser,
                },
            ) from e

    def _create_local_driver(self) -> WebDriver:
        """
        Create WebDriver using local browser installation. Returns: Local
        WebDriver instance Raises: DriverCreationError: When local driver
        creation fails
        """
        logger.debug(f"Creating local driver for {self.config.browser}")

        try:
            if self.config.browser == "chrome":
                return self._create_local_chrome_driver()
            elif self.config.browser == "firefox":
                return self._create_local_firefox_driver()
            elif self.config.browser == "edge":
                return self._create_local_edge_driver()
            else:
                raise DriverNotSupportedError(
                    browser=self.config.browser,
                    supported_browsers=["chrome", "firefox", "edge"],
                    details={"method": "local"},
                )

        except Exception as e:
            if isinstance(e, DriverCreationError | DriverNotSupportedError):
                raise
            raise DriverCreationError(
                browser=self.config.browser, method="local", original_error=e
            ) from e

    def _create_webdriver_manager_driver(self) -> WebDriver:
        """
        Create WebDriver using WebDriverManager for automatic driver
        downloads. Returns: Local WebDriver with automatically managed driver
        binary Raises: DriverCreationError: When WebDriverManager fails
        """
        logger.debug(
            f"Creating WebDriverManager driver for {self.config.browser}"
        )

        try:
            # Import WebDriverManager components lazily
            if self.config.browser == "chrome":
                return self._create_webdriver_manager_chrome()
            elif self.config.browser == "firefox":
                return self._create_webdriver_manager_firefox()
            elif self.config.browser == "edge":
                return self._create_webdriver_manager_edge()
            else:
                raise DriverNotSupportedError(
                    browser=self.config.browser,
                    supported_browsers=["chrome", "firefox", "edge"],
                    details={"method": "webdriver-manager"},
                )

        except ImportError as e:
            raise DriverCreationError(
                browser=self.config.browser,
                method="webdriver-manager",
                original_error=e,
                details={
                    "error": "WebDriverManager not installed. "
                    "Run: pip install webdriver-manager"
                },
            ) from e
        except Exception as e:
            if isinstance(e, DriverCreationError | DriverNotSupportedError):
                raise
            raise DriverCreationError(
                browser=self.config.browser,
                method="webdriver-manager",
                original_error=e,
            ) from e

    def _create_local_chrome_driver(self) -> WebDriver:
        """Create local Chrome WebDriver instance."""
        options = ChromeOptions()
        for arg in self.config.chrome_options_dict["args"]:
            options.add_argument(arg)

        # Add preferences
        for key, value in self.config.chrome_options_dict["prefs"].items():
            options.add_experimental_option("prefs", {key: value})

        # Enable logging if configured
        if self.config.enable_logging:
            options.add_experimental_option("useAutomationExtension", False)
            options.add_experimental_option(
                "excludeSwitches", ["enable-automation"]
            )

        driver = webdriver.Chrome(options=options)
        self._configure_driver_timeouts(driver)
        return driver

    def _create_local_firefox_driver(self) -> WebDriver:
        """Create local Firefox WebDriver instance."""
        options = FirefoxOptions()
        for arg in self.config.firefox_options_dict["args"]:
            options.add_argument(arg)

        # Add preferences
        for key, value in self.config.firefox_options_dict["prefs"].items():
            options.set_preference(key, value)

        driver = webdriver.Firefox(options=options)
        self._configure_driver_timeouts(driver)
        return driver

    def _create_local_edge_driver(self) -> WebDriver:
        """Create local Edge WebDriver instance."""
        options = EdgeOptions()
        for arg in self.config.edge_options_dict["args"]:
            options.add_argument(arg)

        # Add preferences
        for key, value in self.config.edge_options_dict["prefs"].items():
            options.add_experimental_option("prefs", {key: value})

        driver = webdriver.Edge(options=options)
        self._configure_driver_timeouts(driver)
        return driver

    def _create_webdriver_manager_chrome(self) -> WebDriver:
        """Create Chrome WebDriver using WebDriverManager."""
        try:
            from webdriver_manager.chrome import (  # type: ignore[import-untyped]
                ChromeDriverManager,
            )
        except ImportError:
            raise ImportError(
                "webdriver-manager is required for E2E testing. "
                "Install with: conda install webdriver-manager"
            ) from None

        service = ChromeService(
            ChromeDriverManager(
                cache_valid_range=self.config.driver_cache_valid_range
            ).install()
        )
        options = ChromeOptions()

        for arg in self.config.chrome_options_dict["args"]:
            options.add_argument(arg)

        driver = webdriver.Chrome(service=service, options=options)
        self._configure_driver_timeouts(driver)
        return driver

    def _create_webdriver_manager_firefox(self) -> WebDriver:
        """Create Firefox WebDriver using WebDriverManager."""
        try:
            from webdriver_manager.firefox import (  # type: ignore[import-untyped]
                GeckoDriverManager,
            )
        except ImportError:
            raise ImportError(
                "webdriver-manager is required for E2E testing. "
                "Install with: conda install webdriver-manager"
            ) from None

        service = FirefoxService(
            GeckoDriverManager(
                cache_valid_range=self.config.driver_cache_valid_range
            ).install()
        )
        options = FirefoxOptions()

        for arg in self.config.firefox_options_dict["args"]:
            options.add_argument(arg)

        driver = webdriver.Firefox(service=service, options=options)
        self._configure_driver_timeouts(driver)
        return driver

    def _create_webdriver_manager_edge(self) -> WebDriver:
        """Create Edge WebDriver using WebDriverManager."""
        try:
            from webdriver_manager.microsoft import (  # type: ignore[import-untyped]
                EdgeChromiumDriverManager,
            )
        except ImportError:
            raise ImportError(
                "webdriver-manager is required for E2E testing. "
                "Install with: conda install webdriver-manager"
            ) from None

        service = EdgeService(
            EdgeChromiumDriverManager(
                cache_valid_range=self.config.driver_cache_valid_range
            ).install()
        )
        options = EdgeOptions()

        for arg in self.config.edge_options_dict["args"]:
            options.add_argument(arg)

        driver = webdriver.Edge(service=service, options=options)
        self._configure_driver_timeouts(driver)
        return driver

    def _get_browser_options(
        self,
    ) -> ChromeOptions | FirefoxOptions | EdgeOptions:
        """
        Get browser-specific options object. Returns: Browser options object
        configured for the current browser
        """
        if self.config.browser == "chrome":
            options = ChromeOptions()
            for arg in self.config.chrome_options_dict["args"]:
                options.add_argument(arg)
            return options
        elif self.config.browser == "firefox":
            options = FirefoxOptions()
            for arg in self.config.firefox_options_dict["args"]:
                options.add_argument(arg)
            return options
        elif self.config.browser == "edge":
            options = EdgeOptions()
            for arg in self.config.edge_options_dict["args"]:
                options.add_argument(arg)
            return options
        else:
            raise DriverNotSupportedError(
                browser=self.config.browser,
                supported_browsers=["chrome", "firefox", "edge"],
            )

    def _configure_driver_timeouts(self, driver: WebDriver) -> None:
        """
        Configure standard timeouts for WebDriver instance. Args: driver:
        WebDriver instance to configure
        """
        driver.implicitly_wait(self.config.implicit_wait)
        driver.set_page_load_timeout(self.config.page_load_timeout)
        driver.set_script_timeout(self.config.script_timeout)

        # Set window size
        driver.set_window_size(*self.config.window_size)

        logger.debug(
            f"Configured driver timeouts: "
            f"implicit={self.config.implicit_wait}, "
            f"page_load={self.config.page_load_timeout}, "
            f"script={self.config.script_timeout}"
        )

    def _is_docker_grid_available(self) -> bool:
        """
        Check if Docker Grid infrastructure is available. Returns: True if
        Selenium Grid is accessible, False otherwise
        """
        try:
            import requests

            # Check if Selenium Hub is responding
            response = requests.get(
                f"{self.config.selenium_grid_url}/status",
                timeout=self.config.grid_timeout,
            )

            if response.status_code == 200:
                status_data = response.json()
                is_ready = status_data.get("value", {}).get("ready", False)

                if is_ready:
                    logger.debug("Docker Grid is available and ready")
                    return True
                else:
                    logger.debug("Docker Grid is available but not ready")
                    return False
            else:
                logger.debug(
                    f"Docker Grid responded with status {response.status_code}"
                )
                return False

        except ImportError:
            logger.warning(
                "requests library not available, cannot check Docker Grid "
                "status"
            )
            return False
        except Exception as e:
            logger.debug(f"Docker Grid check failed: {e}")
            return False

    def supports_browser(self, browser: BrowserType) -> bool:
        """
        Check if browser is supported by this factory. Args: browser: Browser
        name to check Returns: True if browser is supported, False otherwise
        """
        supported_browsers: list[BrowserType] = ["chrome", "firefox", "edge"]
        return browser in supported_browsers

    def get_supported_browsers(self) -> list[BrowserType]:
        """
        Get list of supported browsers. Returns: List of supported browser
        names
        """
        return ["chrome", "firefox", "edge"]
