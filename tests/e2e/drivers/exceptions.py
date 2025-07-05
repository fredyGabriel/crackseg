"""Custom exceptions for WebDriver management system.

This module defines the exception hierarchy for driver creation, configuration,
and lifecycle management errors in the CrackSeg E2E testing framework.
"""


class DriverError(Exception):
    """Base exception for all driver-related errors."""

    def __init__(
        self, message: str, details: dict[str, str] | None = None
    ) -> None:
        """Initialize driver error with message and optional details.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message with details."""
        base_msg = self.message
        if self.details:
            details_str = ", ".join(
                f"{k}={v}" for k, v in self.details.items()
            )
            return f"{base_msg} ({details_str})"
        return base_msg


class DriverCreationError(DriverError):
    """Exception raised when WebDriver instance cannot be created.

    This exception is raised when driver initialization fails due to:
    - Missing or incompatible driver binaries
    - Browser installation issues
    - Network connectivity problems (Docker Grid)
    - Invalid configuration parameters
    """

    def __init__(
        self,
        browser: str,
        method: str,
        original_error: Exception | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        """Initialize driver creation error.

        Args:
            browser: Browser name that failed to initialize
            method: Driver creation method used
            (docker/local/webdriver-manager)
            original_error: Original exception that caused the failure
            details: Additional context about the failure
        """
        self.browser = browser
        self.method = method
        self.original_error = original_error

        combined_details = details or {}
        combined_details.update(
            {
                "browser": browser,
                "method": method,
            }
        )

        if original_error:
            combined_details["original_error"] = str(original_error)

        message = f"Failed to create {browser} WebDriver using {method} method"
        super().__init__(message, combined_details)


class DriverNotSupportedError(DriverError):
    """Exception raised when requested browser is not supported.

    This exception is raised when:
    - Browser name is not recognized
    - Browser version is not compatible
    - Platform doesn't support the browser
    - Docker image is not available for the browser
    """

    def __init__(
        self,
        browser: str,
        supported_browsers: list[str] | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        """Initialize driver not supported error.

        Args:
            browser: Unsupported browser name
            supported_browsers: List of supported browser names
            details: Additional context about why browser is not supported
        """
        self.browser = browser
        self.supported_browsers = supported_browsers or []

        combined_details = details or {}
        combined_details["browser"] = browser

        if self.supported_browsers:
            combined_details["supported_browsers"] = ", ".join(
                self.supported_browsers
            )

        message = f"Browser '{browser}' is not supported"
        if self.supported_browsers:
            message += (
                f". Supported browsers: {', '.join(self.supported_browsers)}"
            )

        super().__init__(message, combined_details)


class DriverConfigurationError(DriverError):
    """Exception raised when driver configuration is invalid.

    This exception is raised when:
    - Required configuration parameters are missing
    - Configuration values are invalid or incompatible
    - Environment setup is incorrect
    - Docker infrastructure is not available when required
    """

    def __init__(
        self,
        config_field: str,
        config_value: str | None = None,
        expected: str | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        """Initialize driver configuration error.

        Args:
            config_field: Name of the invalid configuration field
            config_value: Invalid configuration value
            expected: Expected value or format description
            details: Additional context about the configuration error
        """
        self.config_field = config_field
        self.config_value = config_value
        self.expected = expected

        combined_details = details or {}
        combined_details["config_field"] = config_field

        if config_value is not None:
            combined_details["config_value"] = str(config_value)

        if expected:
            combined_details["expected"] = expected

        message = f"Invalid configuration for '{config_field}'"
        if expected:
            message += f". Expected: {expected}"
        if config_value is not None:
            message += f". Got: {config_value}"

        super().__init__(message, combined_details)


class DockerInfrastructureError(DriverError):
    """Exception raised when Docker infrastructure is not available or failing.

    This exception is raised when:
    - Docker daemon is not running
    - Selenium Grid is not accessible
    - Browser nodes are not available
    - Network connectivity issues within Docker
    """

    def __init__(
        self,
        service: str,
        endpoint: str | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        """Initialize Docker infrastructure error.

        Args:
            service: Name of the failing Docker service
            endpoint: Service endpoint that failed
            details: Additional context about the infrastructure failure
        """
        self.service = service
        self.endpoint = endpoint

        combined_details = details or {}
        combined_details["service"] = service

        if endpoint:
            combined_details["endpoint"] = endpoint

        message = f"Docker infrastructure error: {service} is not available"
        if endpoint:
            message += f" at {endpoint}"

        super().__init__(message, combined_details)


class DriverCleanupError(DriverError):
    """Exception raised when driver cleanup operations fail.

    This exception is raised when:
    - WebDriver session cannot be terminated properly
    - Temporary files cannot be cleaned up
    - Browser processes remain after driver quit
    - Docker containers fail to stop
    """

    def __init__(
        self,
        cleanup_operation: str,
        driver_info: dict[str, str] | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        """Initialize driver cleanup error.

        Args:
            cleanup_operation: Type of cleanup operation that failed
            driver_info: Information about the driver being cleaned up
            details: Additional context about the cleanup failure
        """
        self.cleanup_operation = cleanup_operation
        self.driver_info = driver_info or {}

        combined_details = details or {}
        combined_details["cleanup_operation"] = cleanup_operation
        combined_details.update(self.driver_info)

        message = f"Driver cleanup failed: {cleanup_operation}"
        super().__init__(message, combined_details)
