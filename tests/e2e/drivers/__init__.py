"""
WebDriver management system for CrackSeg E2E testing. This module
provides a hybrid driver management system that supports both Docker
Grid and local WebDriver setups with automatic fallback capabilities.
Key Features: - Docker-first approach leveraging existing Selenium
Grid infrastructure - WebDriverManager fallback for local development
- Automatic version compatibility checks - Driver lifecycle management
and cleanup - Cross-browser support (Chrome, Firefox, Edge) -
Integration with existing Docker orchestration
"""

from .config import BrowserType, DriverConfig, DriverMethod, WindowSize
from .driver_factory import DriverFactory
from .driver_manager import HybridDriverManager, create_driver, driver_session
from .exceptions import (
    DockerInfrastructureError,
    DriverCleanupError,
    DriverConfigurationError,
    DriverCreationError,
    DriverError,
    DriverNotSupportedError,
)

__all__ = [
    # Configuration
    "DriverConfig",
    "BrowserType",
    "DriverMethod",
    "WindowSize",
    # Main Components
    "DriverFactory",
    "HybridDriverManager",
    # Convenience Functions
    "create_driver",
    "driver_session",
    # Exceptions
    "DriverError",
    "DriverCreationError",
    "DriverNotSupportedError",
    "DriverConfigurationError",
    "DockerInfrastructureError",
    "DriverCleanupError",
]

__version__ = "1.0.0"
