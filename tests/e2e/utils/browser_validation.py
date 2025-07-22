"""
Browser capability validation and compatibility testing utilities.
This module provides functionality to validate browser capabilities
and test compatibility features specific to the CrackSeg application.
Integrates with the browser matrix configuration to ensure consistent
testing across browsers.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only imports for when selenium is available
    from selenium.common.exceptions import (
        JavascriptException,  # type: ignore[import-untyped]
    )
    from selenium.webdriver.remote.webdriver import (
        WebDriver,  # type: ignore[import-untyped]
    )
    from selenium.webdriver.support.ui import (
        WebDriverWait,  # type: ignore[import-untyped]
    )
else:
    # Runtime fallbacks when selenium is not available
    try:
        from selenium.common.exceptions import JavascriptException
        from selenium.webdriver.remote.webdriver import WebDriver
        from selenium.webdriver.support.ui import WebDriverWait

        SELENIUM_AVAILABLE = True
    except ImportError:
        # Mock classes for when selenium is not available
        class WebDriver:  # type: ignore[no-redef]
            pass

        class WebDriverWait:  # type: ignore[no-redef]
            def __init__(self, driver: Any, timeout: float) -> None:
                pass

        class JavascriptException(Exception):  # type: ignore[no-redef]
            pass

        SELENIUM_AVAILABLE = False

from ..config.browser_matrix_config import BrowserCapability, BrowserProfile

logger = logging.getLogger(__name__)


@dataclass
class CapabilityTestResult:
    """Result of a browser capability test."""

    capability: BrowserCapability
    supported: bool
    details: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    test_duration: float = 0.0


@dataclass
class BrowserCompatibilityReport:
    """Comprehensive browser compatibility report."""

    browser: str
    browser_version: str
    platform: str
    test_results: list[CapabilityTestResult] = field(default_factory=list)
    overall_compatibility: bool = True
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    test_timestamp: str = ""

    def add_result(self, result: CapabilityTestResult) -> None:
        """Add a capability test result to the report."""
        self.test_results.append(result)
        if not result.supported:
            self.overall_compatibility = False
            if result.capability in [
                BrowserCapability.JAVASCRIPT,
                BrowserCapability.FILE_UPLOAD,
                BrowserCapability.LOCAL_STORAGE,
            ]:
                self.critical_failures.append(result.capability.value)
            else:
                self.warnings.append(result.capability.value)

    def get_compatibility_score(self) -> float:
        """
        Calculate compatibility score as percentage of supported features.
        """
        if not self.test_results:
            return 0.0
        supported_count = sum(1 for r in self.test_results if r.supported)
        return (supported_count / len(self.test_results)) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "browser": self.browser,
            "browser_version": self.browser_version,
            "platform": self.platform,
            "overall_compatibility": self.overall_compatibility,
            "compatibility_score": self.get_compatibility_score(),
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "test_timestamp": self.test_timestamp,
            "test_results": [
                {
                    "capability": result.capability.value,
                    "supported": result.supported,
                    "details": result.details,
                    "error_message": result.error_message,
                    "test_duration": result.test_duration,
                }
                for result in self.test_results
            ],
        }


class BrowserCapabilityValidator:
    """
    Validates browser capabilities for CrackSeg application compatibility.
    """

    def __init__(self, driver: WebDriver, timeout: float = 10.0) -> None:
        """
        Initialize capability validator.

        Args:
            driver: Selenium WebDriver instance
            timeout: Timeout for capability tests in seconds
        """
        self.driver = driver
        self.timeout = timeout
        self.wait = WebDriverWait(driver, timeout)

    def validate_profile(
        self, profile: BrowserProfile
    ) -> BrowserCompatibilityReport:
        """
        Validate all required capabilities for a browser profile.

        Args:
            profile: Browser profile to validate

        Returns:
            Comprehensive compatibility report
        """
        report = BrowserCompatibilityReport(
            browser=profile.browser,
            browser_version=profile.version,
            platform=self._get_platform_info(),
            test_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        for capability in profile.required_capabilities:
            result = self._test_capability(capability)
            report.add_result(result)

        return report

    def _get_platform_info(self) -> str:
        """Get platform information from the browser."""
        try:
            platform_info = self.driver.execute_script(
                "return navigator.platform + ' - ' + navigator.userAgent;"
            )
            return str(platform_info)
        except Exception:
            return "Unknown Platform"

    def _test_capability(
        self, capability: BrowserCapability
    ) -> CapabilityTestResult:
        """
        Test a specific browser capability.

        Args:
            capability: Capability to test

        Returns:
            Test result with details
        """
        start_time = time.time()

        try:
            if capability == BrowserCapability.JAVASCRIPT:
                result = self._test_javascript_support()
            elif capability == BrowserCapability.CSS_GRID:
                result = self._test_css_grid_support()
            elif capability == BrowserCapability.WEBGL:
                result = self._test_webgl_support()
            elif capability == BrowserCapability.FILE_UPLOAD:
                result = self._test_file_upload_support()
            elif capability == BrowserCapability.LOCAL_STORAGE:
                result = self._test_local_storage_support()
            elif capability == BrowserCapability.SESSION_STORAGE:
                result = self._test_session_storage_support()
            elif capability == BrowserCapability.RESPONSIVE_DESIGN:
                result = self._test_responsive_design_support()
            else:
                result = CapabilityTestResult(
                    capability=capability,
                    supported=False,
                    error_message=(
                        f"Test not implemented for {capability.value}"
                    ),
                )
        except Exception as e:
            result = CapabilityTestResult(
                capability=capability,
                supported=False,
                error_message=str(e),
            )

        result.test_duration = time.time() - start_time
        return result

    def _test_javascript_support(self) -> CapabilityTestResult:
        """Test JavaScript execution capability."""
        try:
            # Test basic JavaScript execution
            js_result = self.driver.execute_script(
                "return 'JavaScript working';"
            )

            # Test modern JavaScript features
            modern_js = self.driver.execute_script(
                """
                try {
                    // Test arrow functions, const/let, template literals
                    const test = () => `ES6 support: ${Date.now()}`;
                    return test();
                } catch (e) {
                    return 'ES6 not supported: ' + e.message;
                }
                """
            )

            es6_supported = "ES6 support:" in str(modern_js)

            return CapabilityTestResult(
                capability=BrowserCapability.JAVASCRIPT,
                supported=js_result == "JavaScript working",
                details={
                    "basic_js": js_result,
                    "es6_support": es6_supported,
                    "modern_js_result": str(modern_js),
                },
            )
        except JavascriptException as e:
            return CapabilityTestResult(
                capability=BrowserCapability.JAVASCRIPT,
                supported=False,
                error_message=f"JavaScript execution failed: {e}",
            )

    def _test_css_grid_support(self) -> CapabilityTestResult:
        """Test CSS Grid layout support."""
        try:
            grid_support = self.driver.execute_script(
                """
                var div = document.createElement('div');
                div.style.display = 'grid';
                return div.style.display === 'grid';
                """
            )
            subgrid_support = self.driver.execute_script(
                """
                try {
                    var div = document.createElement('div');
                    div.style.gridTemplateRows = 'subgrid';
                    return div.style.gridTemplateRows === 'subgrid';
                } catch (e) {
                    return false;
                }
                """
            )

            return CapabilityTestResult(
                capability=BrowserCapability.CSS_GRID,
                supported=bool(grid_support),
                details={
                    "basic_grid": bool(grid_support),
                    "subgrid": bool(subgrid_support),
                },
            )
        except Exception as e:
            return CapabilityTestResult(
                capability=BrowserCapability.CSS_GRID,
                supported=False,
                error_message=str(e),
            )

    def _test_webgl_support(self) -> CapabilityTestResult:
        """Test WebGL support for graphics rendering."""
        try:
            webgl_test = self.driver.execute_script(
                """
                try {
                    var canvas = document.createElement('canvas');
                    var gl = canvas.getContext('webgl') ||
                             canvas.getContext('experimental-webgl');
                    if (!gl) {
                        return {supported: false, version: null};
                    }
                    var version = gl.getParameter(gl.VERSION);
                    var vendor = gl.getParameter(gl.VENDOR);
                    var renderer = gl.getParameter(gl.RENDERER);
                    return {
                        supported: true,
                        version: version,
                        vendor: vendor,
                        renderer: renderer
                    };
                } catch (e) {
                    return {supported: false, error: e.message};
                }
                """
            )

            return CapabilityTestResult(
                capability=BrowserCapability.WEBGL,
                supported=webgl_test.get("supported", False),
                details=webgl_test,
            )
        except Exception as e:
            return CapabilityTestResult(
                capability=BrowserCapability.WEBGL,
                supported=False,
                error_message=str(e),
            )

    def _test_file_upload_support(self) -> CapabilityTestResult:
        """Test file upload input support."""
        try:
            # Create a temporary file input to test support
            file_input_test = self.driver.execute_script(
                """
                var input = document.createElement('input');
                input.type = 'file';
                input.accept = '.jpg,.jpeg,.png';
                input.multiple = true;

                // Test if file input properties are supported
                return {
                    type_supported: input.type === 'file',
                    accept_supported: input.accept === '.jpg,.jpeg,.png',
                    multiple_supported: input.multiple === true,
                    files_api: 'files' in input
                };
                """
            )
            drag_drop_test = self.driver.execute_script(
                """
                try {
                    var div = document.createElement('div');
                    return 'ondragover' in div && 'ondrop' in div;
                } catch (e) {
                    return false;
                }
                """
            )

            all_supported = file_input_test.get(
                "type_supported", False
            ) and file_input_test.get("files_api", False)

            return CapabilityTestResult(
                capability=BrowserCapability.FILE_UPLOAD,
                supported=all_supported,
                details={
                    "file_input": file_input_test,
                    "drag_drop": drag_drop_test,
                },
            )
        except Exception as e:
            return CapabilityTestResult(
                capability=BrowserCapability.FILE_UPLOAD,
                supported=False,
                error_message=str(e),
            )

    def _test_local_storage_support(self) -> CapabilityTestResult:
        """Test localStorage API support."""
        try:
            storage_test = self.driver.execute_script(
                """
                try {
                    if (typeof Storage === 'undefined') {
                        return {
                            supported: false,
                            reason: 'Storage API not available'
                        };
                    }
                    // Test localStorage operations
                    var testKey = 'crackseg_test_' + Date.now();
                    var testValue = 'test_value_' + Math.random();
                    localStorage.setItem(testKey, testValue);
                    var retrieved = localStorage.getItem(testKey);
                    localStorage.removeItem(testKey);
                    return {
                        supported: retrieved === testValue,
                        quota_test: 'quota' in navigator.storage || false
                    };
                } catch (e) {
                    return {supported: false, error: e.message};
                }
                """
            )

            return CapabilityTestResult(
                capability=BrowserCapability.LOCAL_STORAGE,
                supported=storage_test.get("supported", False),
                details=storage_test,
            )
        except Exception as e:
            return CapabilityTestResult(
                capability=BrowserCapability.LOCAL_STORAGE,
                supported=False,
                error_message=str(e),
            )

    def _test_session_storage_support(self) -> CapabilityTestResult:
        """Test sessionStorage API support."""
        try:
            session_test = self.driver.execute_script(
                """
                try {
                    if (typeof Storage === 'undefined') {
                        return {
                            supported: false,
                            reason: 'Storage API not available'
                        };
                    }
                    // Test sessionStorage operations
                    var testKey = 'crackseg_session_test_' + Date.now();
                    var testValue = 'session_value_' + Math.random();
                    sessionStorage.setItem(testKey, testValue);
                    var retrieved = sessionStorage.getItem(testKey);
                    sessionStorage.removeItem(testKey);
                    return {
                        supported: retrieved === testValue,
                        length_property: (
                            typeof sessionStorage.length === 'number'
                        )
                    };
                } catch (e) {
                    return {supported: false, error: e.message};
                }
                """
            )

            return CapabilityTestResult(
                capability=BrowserCapability.SESSION_STORAGE,
                supported=session_test.get("supported", False),
                details=session_test,
            )
        except Exception as e:
            return CapabilityTestResult(
                capability=BrowserCapability.SESSION_STORAGE,
                supported=False,
                error_message=str(e),
            )

    def _test_responsive_design_support(self) -> CapabilityTestResult:
        """Test responsive design features support."""
        try:
            responsive_test = self.driver.execute_script(
                """
                try {
                    return {
                        viewport_meta: !!(
                            document.querySelector('meta[name="viewport"]')
                        ),
                        media_queries: !!(window.matchMedia),
                        css_calc: CSS.supports('width', 'calc(100vw - 10px)'),
                        viewport_units: CSS.supports('width', '100vw'),
                        flexbox: CSS.supports('display', 'flex'),
                        grid: CSS.supports('display', 'grid')
                    };
                } catch (e) {
                    return {error: e.message};
                }
                """
            )

            # Check minimum required features
            required_features = ["media_queries", "flexbox"]
            supported = all(
                responsive_test.get(feature, False)
                for feature in required_features
            )

            return CapabilityTestResult(
                capability=BrowserCapability.RESPONSIVE_DESIGN,
                supported=supported,
                details=responsive_test,
            )
        except Exception as e:
            return CapabilityTestResult(
                capability=BrowserCapability.RESPONSIVE_DESIGN,
                supported=False,
                error_message=str(e),
            )


def validate_browser_compatibility(
    driver: WebDriver, profile: BrowserProfile
) -> BrowserCompatibilityReport:
    """
    Validate browser compatibility for a given profile.

    Args:
        driver: Selenium WebDriver instance
        profile: Browser profile with required capabilities

    Returns:
        Comprehensive compatibility report
    """
    validator = BrowserCapabilityValidator(driver)
    return validator.validate_profile(profile)


def save_compatibility_report(
    report: BrowserCompatibilityReport, file_path: str
) -> None:
    """
    Save compatibility report to JSON file.

    Args:
        report: Browser compatibility report
        file_path: Path to save the report
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Compatibility report saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save compatibility report: {e}")


def check_critical_compatibility(report: BrowserCompatibilityReport) -> bool:
    """
    Check if browser meets critical compatibility requirements.

    Args:
        report: Browser compatibility report

    Returns:
        True if browser meets critical requirements
    """
    critical_capabilities = {
        BrowserCapability.JAVASCRIPT,
        BrowserCapability.FILE_UPLOAD,
        BrowserCapability.LOCAL_STORAGE,
    }

    supported_critical = {
        result.capability
        for result in report.test_results
        if result.supported and result.capability in critical_capabilities
    }

    return len(supported_critical) == len(critical_capabilities)


def get_browser_compatibility_summary(
    reports: list[BrowserCompatibilityReport],
) -> dict[str, Any]:
    """
    Generate summary of browser compatibility across multiple reports.

    Args:
        reports: List of browser compatibility reports

    Returns:
        Summary dictionary with compatibility statistics
    """
    if not reports:
        return {"error": "No reports provided"}

    total_browsers = len(reports)
    compatible_browsers = sum(1 for r in reports if r.overall_compatibility)
    critical_compatible = sum(
        1 for r in reports if check_critical_compatibility(r)
    )

    average_score = (
        sum(r.get_compatibility_score() for r in reports) / total_browsers
    )

    capability_support = {}
    for capability in BrowserCapability:
        supported_count = sum(
            1
            for report in reports
            for result in report.test_results
            if result.capability == capability and result.supported
        )
        capability_support[capability.value] = {
            "supported_browsers": supported_count,
            "support_percentage": (supported_count / total_browsers) * 100,
        }

    return {
        "total_browsers_tested": total_browsers,
        "fully_compatible_browsers": compatible_browsers,
        "critical_compatible_browsers": critical_compatible,
        "average_compatibility_score": round(average_score, 2),
        "capability_support": capability_support,
        "browser_scores": {
            report.browser: report.get_compatibility_score()
            for report in reports
        },
    }
