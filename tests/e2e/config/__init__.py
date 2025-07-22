"""
Cross-browser configuration management for E2E testing. This package
provides configuration management for cross-browser testing with
support for parallel execution, browser-specific capabilities, and
mobile browser emulation.
"""

from .browser_capabilities import (
    BrowserCapabilities,
    ChromeCapabilities,
    EdgeCapabilities,
    FirefoxCapabilities,
    MobileBrowserCapabilities,
)
from .browser_config_manager import (
    BrowserConfigManager,
    BrowserMatrix,
    ParallelExecutionConfig,
)

__all__ = [
    "BrowserCapabilities",
    "ChromeCapabilities",
    "FirefoxCapabilities",
    "EdgeCapabilities",
    "MobileBrowserCapabilities",
    "BrowserConfigManager",
    "BrowserMatrix",
    "ParallelExecutionConfig",
]
