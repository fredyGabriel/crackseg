"""Testing workflows for responsive design validation.

This module provides complete testing workflows that orchestrate viewport
changes,
layout validation, and result reporting for comprehensive responsive testing.
"""

import logging
import time
from pathlib import Path
from typing import Any, TypedDict

from selenium.webdriver.remote.webdriver import WebDriver

from tests.e2e.config.viewport_config import (
    ResponsiveDevice,
    ResponsiveTestMatrix,
)
from tests.e2e.utils.responsive.layout import (
    capture_layout_screenshot,
    check_horizontal_scrolling,
    get_layout_metrics,
    validate_responsive_layout,
)
from tests.e2e.utils.responsive.touch import (
    check_touch_capability,
)
from tests.e2e.utils.responsive.viewport import (
    set_viewport_size,
    wait_for_viewport_stabilization,
)

logger = logging.getLogger(__name__)


class TestSummary(TypedDict):
    """Type definition for test summary."""

    total_devices: int
    passed: int
    failed: int
    errors: int
    start_time: float
    end_time: float
    duration: float


class ErrorInfo(TypedDict):
    """Type definition for error information."""

    device: str
    error: str


class ScreenshotInfo(TypedDict):
    """Type definition for screenshot information."""

    device: str
    path: str
    status: str


class TestResults(TypedDict):
    """Type definition for complete test results."""

    summary: TestSummary
    device_results: dict[str, dict[str, Any]]
    screenshots: list[ScreenshotInfo]
    errors: list[ErrorInfo]


class DeviceResult(TypedDict):
    """Type definition for single device test result."""

    device: str
    viewport: str
    status: str
    tests: dict[str, Any]
    metrics: dict[str, Any]
    start_time: float
    end_time: float
    duration: float


def execute_responsive_test_suite(
    driver: WebDriver,
    test_matrix: ResponsiveTestMatrix,
    base_url: str,
    test_config: dict[str, Any] | None = None,
) -> TestResults:
    """Execute complete responsive test suite across multiple devices.

    Args:
        driver: WebDriver instance
        test_matrix: Test matrix with device configurations
        base_url: Base URL to test
        test_config: Additional test configuration

    Returns:
        Dictionary with comprehensive test results
    """
    try:
        logger.info("Starting responsive test suite execution")

        test_config = test_config or {}

        # Initialize with proper types
        results: TestResults = {
            "summary": {
                "total_devices": len(test_matrix.devices),
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "start_time": time.time(),
                "end_time": 0.0,
                "duration": 0.0,
            },
            "device_results": {},
            "screenshots": [],
            "errors": [],
        }

        # Navigate to base URL
        driver.get(base_url)
        time.sleep(2)  # Allow page to load

        # Test each device configuration
        for device in test_matrix.devices:
            logger.info(f"Testing device: {device.name}")

            device_result = _test_single_device(driver, device, test_config)

            results["device_results"][device.name] = device_result

            # Update summary
            if device_result["status"] == "passed":
                results["summary"]["passed"] += 1
            elif device_result["status"] == "failed":
                results["summary"]["failed"] += 1
            else:
                results["summary"]["errors"] += 1
                error_info: ErrorInfo = {
                    "device": device.name,
                    "error": device_result.get("error", "Unknown error"),
                }
                results["errors"].append(error_info)

            # Collect screenshots
            if "screenshot" in device_result:
                screenshot_info: ScreenshotInfo = {
                    "device": device.name,
                    "path": device_result["screenshot"],
                    "status": device_result["status"],
                }
                results["screenshots"].append(screenshot_info)

        # Finalize results
        results["summary"]["end_time"] = time.time()
        results["summary"]["duration"] = (
            results["summary"]["end_time"] - results["summary"]["start_time"]
        )

        logger.info(f"Responsive test suite completed: {results['summary']}")
        return results

    except Exception as e:
        logger.error(f"Responsive test suite failed: {e}")
        system_error: ErrorInfo = {"device": "system", "error": str(e)}
        error_summary: TestSummary = {
            "total_devices": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "start_time": time.time(),
            "end_time": time.time(),
            "duration": 0.0,
        }
        return {
            "summary": error_summary,
            "device_results": {},
            "screenshots": [],
            "errors": [system_error],
        }


def _test_single_device(
    driver: WebDriver,
    device: ResponsiveDevice,
    test_config: dict[str, Any],
) -> dict[str, Any]:
    """Test responsive behavior on a single device configuration.

    Args:
        driver: WebDriver instance
        device: Device configuration to test
        test_config: Test configuration options

    Returns:
        Dictionary with device test results
    """
    try:
        result: dict[str, Any] = {
            "device": device.name,
            "viewport": f"{device.viewport.width}x{device.viewport.height}",
            "status": "unknown",
            "tests": {},
            "metrics": {},
            "start_time": time.time(),
        }

        # Set viewport size
        viewport_success = set_viewport_size(driver, device.viewport)
        if not viewport_success:
            result["status"] = "error"
            result["error"] = "Failed to set viewport size"
            return result

        # Wait for stabilization
        stabilized = wait_for_viewport_stabilization(driver, device.viewport)
        if not stabilized:
            logger.warning(
                f"Viewport may not be fully stabilized for {device.name}"
            )

        # Capture layout metrics
        result["metrics"] = get_layout_metrics(driver)

        # Initialize tests dict
        tests: dict[str, Any] = {}

        # Run layout validation tests
        layout_checks = test_config.get("layout_checks", {})
        if layout_checks:
            tests["layout"] = validate_responsive_layout(driver, layout_checks)

        # Check horizontal scrolling
        tests["no_horizontal_scroll"] = check_horizontal_scrolling(driver)

        # Touch capability test (for mobile devices)
        if device.touch_capability.value != "none":
            tests["touch_capability"] = check_touch_capability(driver)

        result["tests"] = tests

        # Capture screenshot
        screenshot_dir = test_config.get(
            "screenshot_dir", "test-artifacts/screenshots"
        )
        screenshot_path = _capture_device_screenshot(
            driver, device, screenshot_dir, result["metrics"]
        )
        if screenshot_path:
            result["screenshot"] = screenshot_path

        # Determine overall status
        all_tests_passed = all(
            test_result is True
            for test_result in tests.values()
            if isinstance(test_result, bool)
        )

        layout_tests_passed = True
        if "layout" in tests:
            layout_result = tests["layout"]
            if isinstance(layout_result, dict):
                layout_tests_passed = all(
                    test_result is True
                    for test_result in layout_result.values()
                    if isinstance(test_result, bool)
                )

        if all_tests_passed and layout_tests_passed:
            result["status"] = "passed"
        else:
            result["status"] = "failed"

        result["end_time"] = time.time()
        result["duration"] = result["end_time"] - result["start_time"]

        return result

    except Exception as e:
        logger.error(f"Device test failed for {device.name}: {e}")
        return {
            "device": device.name,
            "status": "error",
            "error": str(e),
            "tests": {},
            "metrics": {},
        }


def _capture_device_screenshot(
    driver: WebDriver,
    device: ResponsiveDevice,
    screenshot_dir: str,
    metrics: dict[str, Any],
) -> str | None:
    """Capture screenshot for device test.

    Args:
        driver: WebDriver instance
        device: Device configuration
        screenshot_dir: Directory for screenshots
        metrics: Layout metrics for overlay

    Returns:
        Screenshot file path if successful, None otherwise
    """
    try:
        # Create screenshot directory
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = int(time.time())
        filename = f"{device.name.lower().replace(' ', '_')}_{timestamp}.png"
        filepath = Path(screenshot_dir) / filename

        # Prepare viewport info for overlay
        viewport_info = {
            "width": device.viewport.width,
            "height": device.viewport.height,
            "device": device.name,
        }

        # Capture screenshot
        success = capture_layout_screenshot(
            driver, str(filepath), viewport_info
        )

        if success:
            logger.debug(f"Screenshot captured: {filepath}")
            return str(filepath)
        else:
            logger.warning(f"Failed to capture screenshot for {device.name}")
            return None

    except Exception as e:
        logger.error(f"Screenshot capture failed for {device.name}: {e}")
        return None


def generate_test_report(
    results: dict[str, Any],
    output_path: str | None = None,
) -> str:
    """Generate HTML test report from results.

    Args:
        results: Test results dictionary
        output_path: Output file path (optional)

    Returns:
        Generated HTML report content
    """
    try:
        summary = results.get("summary", {})
        device_results = results.get("device_results", {})
        screenshots = results.get("screenshots", [])

        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Responsive Design Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .device {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .passed {{ border-left: 5px solid #4CAF50; }}
        .failed {{ border-left: 5px solid #f44336; }}
        .error {{ border-left: 5px solid #ff9800; }}
        .screenshot {{ max-width: 300px; margin: 10px 0; }}
        .metrics {{ font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <h1>Responsive Design Test Report</h1>

    <div class="summary">
        <h2>Test Summary</h2>
        <p><strong>Total Devices:</strong>
        {summary.get("total_devices", 0)}</p>
        <p><strong>Passed:</strong> {summary.get("passed", 0)}</p>
        <p><strong>Failed:</strong> {summary.get("failed", 0)}</p>
        <p><strong>Errors:</strong> {summary.get("errors", 0)}</p>
        <p><strong>Duration:</strong>
        {summary.get("duration", 0):.2f} seconds</p>
    </div>

    <h2>Device Test Results</h2>
        """

        # Add device results
        for device_name, device_result in device_results.items():
            status_class = device_result.get("status", "unknown")

            html_content += f"""
    <div class="device {status_class}">
        <h3>{device_name}</h3>
        <p><strong>Status:</strong>
        {device_result.get("status", "Unknown")}</p>
        <p><strong>Viewport:</strong>
        {device_result.get("viewport", "Unknown")}</p>
        <p><strong>Duration:</strong>
        {device_result.get("duration", 0):.2f}s</p>

        <h4>Test Results</h4>
        <ul>
            """

            # Add test results
            tests = device_result.get("tests", {})
            for test_name, test_result in tests.items():
                result_text = "✓ PASS" if test_result is True else "✗ FAIL"
                html_content += (
                    f"<li><strong>{test_name}:</strong> {result_text}</li>"
                )

            html_content += "</ul>"

            # Add screenshot if available
            screenshot = next(
                (s for s in screenshots if s["device"] == device_name), None
            )
            if screenshot:
                html_content += f"""
        <h4>Screenshot</h4>
        <img src="{screenshot["path"]}" class="screenshot"
        alt="{device_name} screenshot">
                """

            html_content += "</div>"

        html_content += """
</body>
</html>
        """

        # Save to file if path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Test report saved to: {output_path}")

        return html_content

    except Exception as e:
        logger.error(f"Failed to generate test report: {e}")
        return (
            f"<html><body><h1>Error generating report: {e}</h1></body></html>"
        )


def create_test_configuration(
    layout_checks: dict[str, Any] | None = None,
    screenshot_dir: str = "test-artifacts/screenshots",
    stabilization_delay: float = 1.0,
) -> dict[str, Any]:
    """Create test configuration for responsive testing.

    Args:
        layout_checks: Layout validation configuration
        screenshot_dir: Directory for screenshots
        stabilization_delay: Delay for viewport stabilization

    Returns:
        Test configuration dictionary
    """
    return {
        "layout_checks": layout_checks or {},
        "screenshot_dir": screenshot_dir,
        "stabilization_delay": stabilization_delay,
        "capture_screenshots": True,
        "generate_report": True,
    }
