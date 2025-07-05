"""Hybrid WebDriver manager for CrackSeg E2E testing.

This module provides the main orchestration layer for WebDriver management,
implementing the hybrid approach with Docker Grid prioritization and
WebDriverManager fallback for robust testing across environments.
"""

import atexit
import importlib.util
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from selenium.webdriver.remote.webdriver import WebDriver

from .config import DriverConfig
from .driver_factory import DriverFactory
from .exceptions import (
    DriverCleanupError,
    DriverCreationError,
)

logger = logging.getLogger(__name__)


class HybridDriverManager:
    """Hybrid WebDriver manager with Docker Grid priority and smart fallback.

    This manager provides a comprehensive WebDriver management solution that:
    - Prioritizes Docker Grid infrastructure when available
    - Falls back to WebDriverManager for local development
    - Implements robust error handling and retry logic
    - Manages driver lifecycle and cleanup automatically
    - Integrates with existing CrackSeg Docker infrastructure

    Example:
        >>> manager = HybridDriverManager()
        >>> with manager.get_driver("chrome") as driver:
        ...     driver.get("http://localhost:8501")
        ...     # Driver automatically cleaned up
    """

    def __init__(self, config: DriverConfig | None = None) -> None:
        """Initialize hybrid driver manager.

        Args:
            config: Driver configuration. If None, creates from environment
        """
        self.config = config or DriverConfig.from_environment()
        self.factory = DriverFactory(self.config)
        self._active_drivers: list[WebDriver] = []
        self._cleanup_registered = False

        # Set up logging
        self._setup_logging()

        # Register cleanup on exit
        self._register_cleanup()

        logger.info(
            f"Initialized HybridDriverManager with {self.config.browser} "
            "browser"
        )
        logger.debug(f"Configuration: {self.config.to_dict()}")

    def create_driver(
        self,
        browser: str | None = None,
        method: str | None = None,
        retry_count: int = 3,
        retry_delay: float = 2.0,
    ) -> WebDriver:
        """Create WebDriver instance with hybrid method support and retry
        logic.

        Args:
            browser: Browser type ('chrome', 'firefox', 'edge', 'safari')
            method: Creation method ('docker', 'local', 'webdriver-manager',
            'auto')
            retry_count: Number of retry attempts
            retry_delay: Delay between retry attempts in seconds

        Returns:
            Configured WebDriver instance

        Raises:
            DriverCreationError: When all creation attempts fail
        """
        browser = browser or self.config.browser
        method = method or "auto"

        logger.info(
            f"Creating {browser} driver with method={method}, "
            f"retries={retry_count}"
        )

        # Update config for this browser if different
        # Ensure browser is a valid BrowserType
        valid_browsers = ["chrome", "firefox", "edge", "safari"]
        if browser not in valid_browsers:
            raise DriverCreationError(
                browser=browser,
                method=method,
                details={
                    "error": f"Unsupported browser: {browser}. Must be one of "
                    f"{valid_browsers}"
                },
            )

        if browser != self.config.browser:
            config = DriverConfig(
                browser=browser,  # type: ignore  # We've validated browser type above
                **{
                    k: v
                    for k, v in self.config.to_dict().items()
                    if k != "browser"
                },
            )
            factory = DriverFactory(config)
        else:
            factory = self.factory

        last_error = None
        methods_tried = []
        current_method = "auto"  # Initialize to prevent unbound variable

        for attempt in range(retry_count + 1):
            if attempt > 0:
                logger.debug(
                    f"Retry attempt {attempt}/{retry_count} after "
                    f"{retry_delay}s delay"
                )
                time.sleep(retry_delay)

            try:
                # Determine method for this attempt
                current_method = self._get_method_for_attempt(
                    method, attempt, methods_tried
                )
                methods_tried.append(current_method)

                logger.debug(
                    f"Attempting driver creation with method: {current_method}"
                )
                driver = factory.create_driver(current_method)

                # Track active driver for cleanup
                self._active_drivers.append(driver)

                logger.info(
                    f"Successfully created {browser} driver using "
                    f"{current_method} method"
                )
                return driver

            except DriverCreationError as e:
                last_error = e
                logger.warning(
                    f"Driver creation failed (attempt {attempt + 1}): {e}"
                )

                # If Docker method failed, check if infrastructure is the issue
                if (
                    current_method == "docker"
                    and self._is_docker_infrastructure_error(e)
                ):
                    logger.info(
                        "Docker infrastructure appears to be unavailable, "
                        "trying fallback methods"
                    )

            except Exception as e:
                last_error = DriverCreationError(
                    browser=browser, method=current_method, original_error=e
                )
                logger.error(f"Unexpected error during driver creation: {e}")

        # All attempts failed
        methods_str = ", ".join(methods_tried)
        error_msg = f"Failed to create {browser} driver after "
        f"{retry_count + 1} attempts using methods: {methods_str}"

        if last_error:
            raise DriverCreationError(
                browser=browser,
                method=methods_str,
                original_error=last_error,
                details={
                    "attempts": str(retry_count + 1),
                    "methods_tried": methods_str,
                    "last_error": str(last_error),
                },
            )
        else:
            raise DriverCreationError(
                browser=browser,
                method=methods_str,
                details={"error": error_msg},
            )

    @contextmanager
    def get_driver(
        self,
        browser: str | None = None,
        method: str | None = None,
        **kwargs: Any,
    ) -> Generator[WebDriver, None, None]:
        """Context manager for WebDriver with automatic cleanup.

        Args:
            browser: Browser type to create
            method: Driver creation method
            **kwargs: Additional arguments for create_driver

        Yields:
            WebDriver instance that will be automatically cleaned up

        Example:
            >>> manager = HybridDriverManager()
            >>> with manager.get_driver("chrome") as driver:
            ...     driver.get("http://localhost:8501")
            ...     assert "CrackSeg" in driver.title
            # Driver automatically quit here
        """
        driver = None
        try:
            driver = self.create_driver(
                browser=browser, method=method, **kwargs
            )
            logger.debug(
                f"Context manager created driver: {driver.session_id}"
            )
            yield driver
        finally:
            if driver:
                self.cleanup_driver(driver)

    def cleanup_driver(self, driver: WebDriver) -> None:
        """Clean up a single WebDriver instance.

        Args:
            driver: WebDriver instance to clean up

        Raises:
            DriverCleanupError: When cleanup operations fail
        """
        if driver not in self._active_drivers:
            logger.warning("Attempted to cleanup driver that is not tracked")
            return

        try:
            session_id = getattr(driver, "session_id", "unknown")
            logger.debug(f"Cleaning up driver with session: {session_id}")

            # Attempt graceful quit
            driver.quit()

            # Remove from tracking
            self._active_drivers.remove(driver)

            logger.info(f"Successfully cleaned up driver: {session_id}")

        except Exception as e:
            logger.error(f"Error during driver cleanup: {e}")

            # Still remove from tracking to prevent memory leaks
            if driver in self._active_drivers:
                self._active_drivers.remove(driver)

            raise DriverCleanupError(
                cleanup_operation="driver_quit",
                driver_info={
                    "session_id": getattr(driver, "session_id", "unknown")
                },
                details={"error": str(e)},
            ) from e

    def cleanup_all_drivers(self) -> None:
        """Clean up all active WebDriver instances.

        This method attempts to gracefully quit all tracked drivers,
        logging errors but not raising exceptions for individual failures.
        """
        if not self._active_drivers:
            logger.debug("No active drivers to clean up")
            return

        logger.info(f"Cleaning up {len(self._active_drivers)} active drivers")

        # Create a copy of the list since cleanup_driver modifies it
        drivers_to_cleanup = self._active_drivers.copy()

        for driver in drivers_to_cleanup:
            try:
                self.cleanup_driver(driver)
            except DriverCleanupError as e:
                logger.error(f"Failed to cleanup driver: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during driver cleanup: {e}")

        # Final check for any remaining drivers
        if self._active_drivers:
            logger.warning(
                f"{len(self._active_drivers)} drivers could not be cleaned up "
                "properly"
            )
            self._active_drivers.clear()

    def get_active_driver_count(self) -> int:
        """Get the number of currently active drivers.

        Returns:
            Number of active WebDriver instances
        """
        return len(self._active_drivers)

    def is_docker_grid_available(self) -> bool:
        """Check if Docker Grid infrastructure is available.

        Returns:
            True if Selenium Grid is accessible and ready
        """
        return self.factory._is_docker_grid_available()

    def validate_configuration(self) -> dict[str, Any]:
        """Validate current configuration and environment.

        Returns:
            Dictionary with validation results and recommendations

        Example:
            >>> manager = HybridDriverManager()
            >>> status = manager.validate_configuration()
            >>> print(f"Docker available: {status['docker_available']}")
        """
        # Initialize with explicit types to help type checker
        validation_results: dict[str, Any] = {
            "configuration_valid": True,
            "docker_available": False,
            "webdriver_manager_available": False,
            "local_drivers_available": False,
            "supported_browsers": self.factory.get_supported_browsers(),
            "recommendations": [],  # list[str]
            "warnings": [],  # list[str]
        }

        try:
            # Test Docker Grid availability
            validation_results["docker_available"] = (
                self.is_docker_grid_available()
            )
            if validation_results["docker_available"]:
                recommendations = validation_results["recommendations"]
                recommendations.append(
                    "Docker Grid is available - optimal for testing"
                )
            else:
                warnings = validation_results["warnings"]
                warnings.append(
                    "Docker Grid is not available - will use fallback methods"
                )

            # Test WebDriverManager availability
            try:
                # Check if webdriver-manager is available without importing it
                webdriver_manager_spec = importlib.util.find_spec(
                    "webdriver_manager"
                )

                if webdriver_manager_spec is not None:
                    validation_results["webdriver_manager_available"] = True
                    recommendations = validation_results["recommendations"]
                    recommendations.append(
                        "WebDriverManager is available for automatic driver "
                        "management"
                    )
                else:
                    warnings = validation_results["warnings"]
                    warnings.append(
                        "WebDriverManager not installed - run: pip install "
                        "webdriver-manager"
                    )
            except ImportError:
                warnings = validation_results["warnings"]
                warnings.append(
                    "WebDriverManager not installed - run: pip install "
                    "webdriver-manager"
                )

            # Test local driver availability (basic check)
            try:
                # Check if selenium webdriver module is available
                selenium_spec = importlib.util.find_spec("selenium.webdriver")

                if selenium_spec is not None:
                    validation_results["local_drivers_available"] = True
                    recommendations = validation_results["recommendations"]
                    recommendations.append(
                        "Local drivers available for fallback"
                    )
                else:
                    warnings = validation_results["warnings"]
                    warnings.append(
                        "Selenium WebDriver not properly installed"
                    )
            except ImportError:
                warnings = validation_results["warnings"]
                warnings.append("Selenium WebDriver not available")

        except Exception as e:
            validation_results["configuration_valid"] = False
            warnings = validation_results["warnings"]
            warnings.append(f"Validation error: {e}")

        return validation_results

    def _get_method_for_attempt(
        self, method: str, attempt: int, methods_tried: list[str]
    ) -> str:
        """Determine the best method for a specific retry attempt.

        Args:
            method: Originally requested method
            attempt: Current attempt number (0-based)
            methods_tried: List of methods already attempted

        Returns:
            Method name to try for this attempt
        """
        if method != "auto":
            return method

        # For auto method, implement fallback strategy
        fallback_order = ["docker", "webdriver-manager", "local"]

        for fallback_method in fallback_order:
            if fallback_method not in methods_tried:
                return fallback_method

        # If all methods tried, start over (for retry attempts)
        return fallback_order[attempt % len(fallback_order)]

    def _is_docker_infrastructure_error(
        self, error: DriverCreationError
    ) -> bool:
        """Check if error indicates Docker infrastructure issues.

        Args:
            error: Driver creation error to analyze

        Returns:
            True if error appears to be Docker infrastructure related
        """
        error_indicators = [
            "connection refused",
            "connection timeout",
            "network is unreachable",
            "selenium grid",
            "hub",
            "remote",
            "grid/api",
        ]

        error_message = str(error).lower()
        return any(
            indicator in error_message for indicator in error_indicators
        )

    def _setup_logging(self) -> None:
        """Configure logging for the driver manager."""
        if self.config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper()),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

    def _register_cleanup(self) -> None:
        """Register cleanup function to run on program exit."""
        if not self._cleanup_registered:
            atexit.register(self.cleanup_all_drivers)
            self._cleanup_registered = True
            logger.debug("Registered exit cleanup handler")

    def __enter__(self) -> "HybridDriverManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup_all_drivers()

    def __del__(self) -> None:
        """Destructor to ensure cleanup on garbage collection."""
        try:
            self.cleanup_all_drivers()
        except Exception:
            # Ignore errors during garbage collection
            pass


# Convenience functions for common use cases


def create_driver(
    browser: str = "chrome",
    method: str = "auto",
    config: DriverConfig | None = None,
    **kwargs: Any,
) -> WebDriver:
    """Convenience function to create a WebDriver instance.

    Args:
        browser: Browser type to create
        method: Driver creation method
        config: Optional configuration. Uses environment defaults if None
        **kwargs: Additional configuration overrides

    Returns:
        Configured WebDriver instance

    Example:
        >>> driver = create_driver("firefox", headless=False)
        >>> driver.get("http://localhost:8501")
    """
    if config is None:
        config = DriverConfig.from_environment(browser=browser, **kwargs)

    manager = HybridDriverManager(config)
    return manager.create_driver(browser=browser, method=method)


@contextmanager
def driver_session(
    browser: str = "chrome",
    method: str = "auto",
    config: DriverConfig | None = None,
    **kwargs: Any,
) -> Generator[WebDriver, None, None]:
    """Context manager for WebDriver session with automatic cleanup.

    Args:
        browser: Browser type to create
        method: Driver creation method
        config: Optional configuration
        **kwargs: Additional configuration overrides

    Yields:
        WebDriver instance with automatic cleanup

    Example:
        >>> with driver_session("chrome", headless=False) as driver:
        ...     driver.get("http://localhost:8501")
        ...     assert "CrackSeg" in driver.title
    """
    if config is None:
        config = DriverConfig.from_environment(browser=browser, **kwargs)

    with HybridDriverManager(config) as manager:
        with manager.get_driver(browser=browser, method=method) as driver:
            yield driver
