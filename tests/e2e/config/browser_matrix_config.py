"""Browser matrix configuration for cross-browser testing framework.

This module provides comprehensive configuration management for cross-browser
testing using pytest parametrization and parallel execution. Extends the
existing parallel execution framework with browser-specific capabilities and
validation.
"""

import logging
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..drivers import BrowserType, DriverConfig

logger = logging.getLogger(__name__)


class BrowserCapability(Enum):
    """Browser capabilities for feature testing."""

    JAVASCRIPT = "javascript"
    CSS_GRID = "css_grid"
    WEBGL = "webgl"
    FILE_UPLOAD = "file_upload"
    LOCAL_STORAGE = "local_storage"
    SESSION_STORAGE = "session_storage"
    GEOLOCATION = "geolocation"
    NOTIFICATIONS = "notifications"
    FULLSCREEN = "fullscreen"
    RESPONSIVE_DESIGN = "responsive_design"


class BrowserMatrixStrategy(Enum):
    """Strategies for browser matrix execution."""

    SEQUENTIAL = "sequential"  # Run browsers one by one
    PARALLEL = "parallel"  # Run all browsers in parallel
    STAGED = "staged"  # Run in groups with fallback
    SMOKE_TEST = "smoke_test"  # Quick compatibility check
    FULL_SUITE = "full_suite"  # Comprehensive browser testing


@dataclass
class BrowserProfile:
    """Configuration profile for a specific browser."""

    browser: BrowserType
    version: str = "latest"
    enabled: bool = True
    priority: int = 1  # 1=high, 2=medium, 3=low
    headless: bool = True
    window_size: tuple[int, int] = (1920, 1080)

    # Browser-specific capabilities
    required_capabilities: list[BrowserCapability] = field(
        default_factory=list
    )
    experimental_features: list[str] = field(default_factory=list)

    # Performance and resource settings
    max_instances: int = 2
    timeout_multiplier: float = 1.0
    memory_limit_mb: int = 1024

    # Platform compatibility
    supported_platforms: list[str] = field(
        default_factory=lambda: ["Windows", "Linux", "Darwin"]
    )

    def __post_init__(self) -> None:
        """Validate browser profile after initialization."""
        self._validate_platform_compatibility()
        self._set_browser_defaults()

    def _validate_platform_compatibility(self) -> None:
        """Check if browser is supported on current platform."""
        current_platform = platform.system()

        if current_platform not in self.supported_platforms:
            logger.warning(
                f"Browser {self.browser} may not be supported on "
                f"{current_platform}. "
                f"Supported platforms: {', '.join(self.supported_platforms)}"
            )

        # Safari is only available on macOS
        if self.browser == "safari" and current_platform != "Darwin":
            logger.warning(
                f"Safari browser requested but running on {current_platform}. "
                "Safari is only available on macOS."
            )
            self.enabled = False

    def _set_browser_defaults(self) -> None:
        """Set browser-specific default capabilities."""
        if self.browser == "chrome":
            if not self.required_capabilities:
                self.required_capabilities = [
                    BrowserCapability.JAVASCRIPT,
                    BrowserCapability.CSS_GRID,
                    BrowserCapability.FILE_UPLOAD,
                    BrowserCapability.LOCAL_STORAGE,
                ]
        elif self.browser == "firefox":
            if not self.required_capabilities:
                self.required_capabilities = [
                    BrowserCapability.JAVASCRIPT,
                    BrowserCapability.CSS_GRID,
                    BrowserCapability.FILE_UPLOAD,
                    BrowserCapability.SESSION_STORAGE,
                ]
        elif self.browser == "edge":
            if not self.required_capabilities:
                self.required_capabilities = [
                    BrowserCapability.JAVASCRIPT,
                    BrowserCapability.CSS_GRID,
                    BrowserCapability.WEBGL,
                    BrowserCapability.LOCAL_STORAGE,
                ]

    def to_driver_config(self, base_config: DriverConfig) -> DriverConfig:
        """Convert browser profile to DriverConfig.

        Args:
            base_config: Base configuration to extend

        Returns:
            DriverConfig: Browser-specific driver configuration
        """
        return DriverConfig(
            browser=self.browser,
            browser_version=self.version,
            headless=self.headless,
            window_size=self.window_size,
            page_load_timeout=base_config.page_load_timeout
            * self.timeout_multiplier,
            script_timeout=base_config.script_timeout
            * self.timeout_multiplier,
            **{
                k: v
                for k, v in base_config.to_dict().items()
                if k
                not in {
                    "browser",
                    "browser_version",
                    "headless",
                    "window_size",
                    "page_load_timeout",
                    "script_timeout",
                }
            },
        )


@dataclass
class BrowserMatrixConfig:
    """Configuration for cross-browser testing matrix."""

    # Browser profiles
    browsers: dict[BrowserType, BrowserProfile] = field(default_factory=dict)

    # Execution strategy
    strategy: BrowserMatrixStrategy = BrowserMatrixStrategy.PARALLEL

    # Test selection
    run_smoke_tests_only: bool = False
    compatibility_tests_enabled: bool = True
    performance_comparison_enabled: bool = False

    # Resource management
    max_parallel_browsers: int = 3
    browser_startup_delay: float = 2.0
    cleanup_between_browsers: bool = True

    # Failure handling
    fail_fast_on_browser_error: bool = False
    continue_on_capability_failure: bool = True
    skip_unsupported_browsers: bool = True

    # Reporting
    generate_browser_comparison_report: bool = True
    capture_browser_specific_screenshots: bool = True
    enable_browser_performance_metrics: bool = False

    def __post_init__(self) -> None:
        """Initialize default browser profiles if none provided."""
        if not self.browsers:
            self._create_default_browser_profiles()
        self._validate_configuration()

    def _create_default_browser_profiles(self) -> None:
        """Create default browser profiles for supported browsers."""
        # Chrome profile (highest priority)
        self.browsers["chrome"] = BrowserProfile(
            browser="chrome",
            priority=1,
            enabled=True,
            max_instances=3,
        )

        # Firefox profile (medium priority)
        self.browsers["firefox"] = BrowserProfile(
            browser="firefox",
            priority=2,
            enabled=True,
            max_instances=2,
            timeout_multiplier=1.2,  # Firefox typically slower
        )

        # Edge profile (medium priority)
        self.browsers["edge"] = BrowserProfile(
            browser="edge",
            priority=2,
            enabled=True,
            max_instances=2,
        )

        # Safari profile (low priority, macOS only)
        self.browsers["safari"] = BrowserProfile(
            browser="safari",
            priority=3,
            enabled=platform.system() == "Darwin",
            max_instances=1,
            timeout_multiplier=1.5,
            supported_platforms=["Darwin"],
        )

    def _validate_configuration(self) -> None:
        """Validate browser matrix configuration."""
        enabled_browsers = [p for p in self.browsers.values() if p.enabled]

        if not enabled_browsers:
            raise ValueError(
                "At least one browser must be enabled for testing"
            )

        if self.max_parallel_browsers < 1:
            raise ValueError("max_parallel_browsers must be at least 1")

        total_max_instances = sum(p.max_instances for p in enabled_browsers)
        if total_max_instances > 10:
            logger.warning(
                f"Total max browser instances ({total_max_instances}) may "
                "cause resource exhaustion. Consider reducing "
                "max_instances per browser."
            )

    def get_enabled_browsers(self) -> list[BrowserProfile]:
        """Get list of enabled browser profiles sorted by priority.

        Returns:
            List of enabled browser profiles
        """
        enabled = [p for p in self.browsers.values() if p.enabled]
        return sorted(enabled, key=lambda p: p.priority)

    def get_pytest_parameters(self) -> list[BrowserType]:
        """Get browser names for pytest parametrization.

        Returns:
            List of browser names for pytest.mark.parametrize
        """
        return [p.browser for p in self.get_enabled_browsers()]

    def get_browser_profile(
        self, browser: BrowserType
    ) -> BrowserProfile | None:
        """Get browser profile by browser type.

        Args:
            browser: Browser type to lookup

        Returns:
            Browser profile or None if not found
        """
        return self.browsers.get(browser)

    def should_run_browser(self, browser: BrowserType) -> bool:
        """Check if browser should be included in test run.

        Args:
            browser: Browser type to check

        Returns:
            True if browser should be tested
        """
        profile = self.get_browser_profile(browser)
        return profile is not None and profile.enabled

    def get_execution_groups(self) -> list[list[BrowserType]]:
        """Get browser execution groups based on strategy.

        Returns:
            List of browser groups for execution
        """
        enabled_browsers: list[BrowserType] = [
            p.browser for p in self.get_enabled_browsers()
        ]

        if self.strategy == BrowserMatrixStrategy.SEQUENTIAL:
            return [[browser] for browser in enabled_browsers]
        elif self.strategy == BrowserMatrixStrategy.PARALLEL:
            return [enabled_browsers]
        elif self.strategy == BrowserMatrixStrategy.STAGED:
            # Group by priority
            high_priority: list[BrowserType] = [
                p.browser
                for p in self.get_enabled_browsers()
                if p.priority == 1
            ]
            medium_priority: list[BrowserType] = [
                p.browser
                for p in self.get_enabled_browsers()
                if p.priority == 2
            ]
            low_priority: list[BrowserType] = [
                p.browser
                for p in self.get_enabled_browsers()
                if p.priority == 3
            ]

            groups: list[list[BrowserType]] = []
            if high_priority:
                groups.append(high_priority)
            if medium_priority:
                groups.append(medium_priority)
            if low_priority:
                groups.append(low_priority)
            return groups
        elif self.strategy == BrowserMatrixStrategy.SMOKE_TEST:
            # Only run highest priority browser for quick check
            highest_priority = min(
                p.priority for p in self.get_enabled_browsers()
            )
            smoke_browsers: list[BrowserType] = [
                p.browser
                for p in self.get_enabled_browsers()
                if p.priority == highest_priority
            ]
            return [smoke_browsers[:1]]  # Take only first browser
        else:  # FULL_SUITE
            return [enabled_browsers]


def create_browser_matrix_config(
    strategy: BrowserMatrixStrategy = BrowserMatrixStrategy.PARALLEL,
    enable_safari: bool | None = None,
    max_parallel: int = 3,
    **kwargs: Any,
) -> BrowserMatrixConfig:
    """Create browser matrix configuration with common presets.

    Args:
        strategy: Execution strategy for browser matrix
        enable_safari: Enable Safari browser (auto-detected if None)
        max_parallel: Maximum parallel browsers
        **kwargs: Additional configuration options

    Returns:
        Configured BrowserMatrixConfig instance
    """
    config = BrowserMatrixConfig(
        strategy=strategy,
        max_parallel_browsers=max_parallel,
        **kwargs,
    )

    # Override Safari setting if specified
    if enable_safari is not None and "safari" in config.browsers:
        config.browsers["safari"].enabled = enable_safari

    return config


def get_browser_matrix_pytest_args(config: BrowserMatrixConfig) -> list[str]:
    """Generate pytest arguments for browser matrix testing.

    Args:
        config: Browser matrix configuration

    Returns:
        List of pytest command line arguments
    """
    args = []

    # Add browser parametrization
    enabled_browsers = config.get_pytest_parameters()
    if len(enabled_browsers) > 1:
        browser_params = ",".join(enabled_browsers)
        args.extend(["--browser-matrix", browser_params])

    # Add parallel execution if enabled
    if config.strategy == BrowserMatrixStrategy.PARALLEL:
        args.extend(["-n", str(config.max_parallel_browsers)])
        args.extend(["--dist", "loadgroup"])

    # Add markers for browser testing
    args.extend(["-m", "cross_browser or browser_matrix"])

    # Add failure handling
    if config.fail_fast_on_browser_error:
        args.append("-x")

    # Add verbose output for browser debugging
    args.extend(["-v", "--tb=short"])

    return args


# Predefined configurations
DEVELOPMENT_CONFIG = create_browser_matrix_config(
    strategy=BrowserMatrixStrategy.SMOKE_TEST,
    max_parallel=1,
    run_smoke_tests_only=True,
)

CI_CONFIG = create_browser_matrix_config(
    strategy=BrowserMatrixStrategy.PARALLEL,
    max_parallel=2,
    enable_safari=False,  # Safari not available in most CI environments
    fail_fast_on_browser_error=True,
)

COMPREHENSIVE_CONFIG = create_browser_matrix_config(
    strategy=BrowserMatrixStrategy.FULL_SUITE,
    max_parallel=3,
    compatibility_tests_enabled=True,
    performance_comparison_enabled=True,
    generate_browser_comparison_report=True,
)

STAGING_CONFIG = create_browser_matrix_config(
    strategy=BrowserMatrixStrategy.STAGED,
    max_parallel=2,
    cleanup_between_browsers=True,
    capture_browser_specific_screenshots=True,
)
