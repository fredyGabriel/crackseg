"""
Viewport manipulation utilities for responsive testing. This module
provides functions for viewport resizing, orientation changes, and
viewport stabilization with proper waiting and error handling.
"""

import logging
import time

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver

from tests.e2e.config.viewport_config import Orientation, ViewportDimensions

logger = logging.getLogger(__name__)


def set_viewport_size(
    driver: WebDriver,
    viewport: ViewportDimensions,
    stabilization_delay: float = 0.5,
) -> bool:
    """
    Set browser viewport to specified dimensions. Args: driver: WebDriver
    instance viewport: Target viewport dimensions stabilization_delay:
    Time to wait for viewport stabilization Returns: True if successful,
    False otherwise
    """
    try:
        logger.debug(f"Setting viewport to {viewport}")

        # Get current window size
        current_size = driver.get_window_size()
        logger.debug(f"Current window size: {current_size}")

        # Calculate window size needed for desired viewport
        # Account for browser chrome (toolbars, etc.)
        viewport_adjustment = _calculate_viewport_adjustment(driver)

        target_width = viewport.width + viewport_adjustment["width"]
        target_height = viewport.height + viewport_adjustment["height"]

        # Set window size
        driver.set_window_size(target_width, target_height)

        # Wait for stabilization
        if stabilization_delay > 0:
            time.sleep(stabilization_delay)

        # Verify viewport size
        actual_viewport = get_actual_viewport_size(driver)
        tolerance = 10  # pixels

        width_match = (
            abs(actual_viewport["width"] - viewport.width) <= tolerance
        )
        height_match = (
            abs(actual_viewport["height"] - viewport.height) <= tolerance
        )

        if width_match and height_match:
            logger.debug(f"Viewport set successfully: {actual_viewport}")
            return True
        else:
            logger.warning(
                f"Viewport size mismatch. Expected: "
                f"{viewport.width}x{viewport.height}, "
                f"Actual: {actual_viewport['width']}x"
                f"{actual_viewport['height']}"
            )
            return False

    except WebDriverException as e:
        logger.error(f"Failed to set viewport size: {e}")
        return False


def get_actual_viewport_size(driver: WebDriver) -> dict[str, int]:
    """
    Get actual viewport size from browser. Args: driver: WebDriver
    instance Returns: Dictionary with width and height of actual viewport
    """
    try:
        # Use JavaScript to get viewport dimensions
        viewport_script = """
return { width: window.innerWidth ||
document.documentElement.clientWidth, height: window.innerHeight ||
document.documentElement.clientHeight };
"""

        result = driver.execute_script(viewport_script)
        return {"width": int(result["width"]), "height": int(result["height"])}

    except WebDriverException as e:
        logger.error(f"Failed to get viewport size: {e}")
        # Fallback to window size
        window_size = driver.get_window_size()
        return {"width": window_size["width"], "height": window_size["height"]}


def _calculate_viewport_adjustment(driver: WebDriver) -> dict[str, int]:
    """
    Calculate browser chrome size to adjust for viewport. Args: driver:
    WebDriver instance Returns: Dictionary with width and height
    adjustments needed
    """
    try:
        # Get window outer dimensions and viewport dimensions
        dimensions_script = """
return { outerWidth: window.outerWidth || 0, outerHeight:
window.outerHeight || 0, innerWidth: window.innerWidth ||
document.documentElement.clientWidth, innerHeight: window.innerHeight
|| document.documentElement.clientHeight };
"""

        dimensions = driver.execute_script(dimensions_script)

        # Calculate chrome size
        chrome_width = dimensions["outerWidth"] - dimensions["innerWidth"]
        chrome_height = dimensions["outerHeight"] - dimensions["innerHeight"]

        # Ensure positive values and reasonable bounds
        chrome_width = max(0, min(chrome_width, 200))  # Max 200px chrome
        chrome_height = max(0, min(chrome_height, 300))  # Max 300px chrome

        return {"width": chrome_width, "height": chrome_height}

    except WebDriverException:
        # Default fallback values for common browsers
        return {"width": 0, "height": 100}


def simulate_orientation_change(
    driver: WebDriver,
    target_orientation: Orientation,
    current_viewport: ViewportDimensions,
    change_delay: float = 1.0,
) -> ViewportDimensions | None:
    """
    Simulate device orientation change. Args: driver: WebDriver instance
    target_orientation: Target orientation current_viewport: Current
    viewport dimensions change_delay: Time to wait during orientation
    change Returns: New ViewportDimensions after orientation change, None
    if failed
    """
    try:
        logger.debug(
            f"Simulating orientation change to {target_orientation.value}"
        )

        # Get new dimensions for target orientation
        if target_orientation == Orientation.LANDSCAPE:
            new_viewport = current_viewport.get_landscape_dimensions()
        else:
            new_viewport = current_viewport.get_portrait_dimensions()

        # Skip if already in target orientation
        if (
            new_viewport.width == current_viewport.width
            and new_viewport.height == current_viewport.height
        ):
            logger.debug("Already in target orientation")
            return current_viewport

        # Set new viewport size
        success = set_viewport_size(driver, new_viewport, 0)

        if not success:
            logger.error("Failed to change viewport size for orientation")
            return None

        # Simulate orientation change delay
        if change_delay > 0:
            time.sleep(change_delay)

        # Trigger resize events
        _trigger_resize_events(driver)

        logger.debug(f"Orientation changed to {target_orientation.value}")
        return new_viewport

    except Exception as e:
        logger.error(f"Failed to simulate orientation change: {e}")
        return None


def _trigger_resize_events(driver: WebDriver) -> None:
    """
    Trigger browser resize events to notify responsive code. Args: driver:
    WebDriver instance
    """
    try:
        resize_script = """
// Trigger resize event window.dispatchEvent(new Event('resize')); //
Trigger orientationchange event if supported if ('onorientationchange'
in window) { window.dispatchEvent(new Event('orientationchange')); }
// Give time for handlers to execute return true;
"""

        driver.execute_script(resize_script)

        # Small delay for event processing
        time.sleep(0.1)

    except WebDriverException as e:
        logger.warning(f"Failed to trigger resize events: {e}")


def wait_for_viewport_stabilization(
    driver: WebDriver,
    expected_viewport: ViewportDimensions,
    timeout: float = 5.0,
    check_interval: float = 0.1,
) -> bool:
    """
    Wait for viewport to stabilize at expected dimensions. Args: driver:
    WebDriver instance expected_viewport: Expected viewport dimensions
    timeout: Maximum time to wait check_interval: Time between stability
    checks Returns: True if viewport stabilized, False if timeout
    """
    start_time = time.time()
    tolerance = 10  # pixels
    stable_count = 0
    required_stable_checks = 3  # Need 3 consecutive stable readings

    while time.time() - start_time < timeout:
        try:
            actual = get_actual_viewport_size(driver)

            width_stable = (
                abs(actual["width"] - expected_viewport.width) <= tolerance
            )
            height_stable = (
                abs(actual["height"] - expected_viewport.height) <= tolerance
            )

            if width_stable and height_stable:
                stable_count += 1
                if stable_count >= required_stable_checks:
                    logger.debug("Viewport stabilized")
                    return True
            else:
                stable_count = 0

            time.sleep(check_interval)

        except WebDriverException:
            stable_count = 0
            time.sleep(check_interval)

    logger.warning(f"Viewport failed to stabilize within {timeout}s")
    return False


def get_device_pixel_ratio(driver: WebDriver) -> float:
    """
    Get device pixel ratio from browser. Args: driver: WebDriver instance
    Returns: Device pixel ratio (1.0 if unable to determine)
    """
    try:
        ratio_script = "return window.devicePixelRatio || 1.0;"
        ratio = driver.execute_script(ratio_script)
        return float(ratio)

    except (WebDriverException, ValueError, TypeError):
        logger.warning("Could not determine device pixel ratio, using 1.0")
        return 1.0


def is_mobile_viewport(driver: WebDriver, mobile_threshold: int = 768) -> bool:
    """
    Check if current viewport is mobile-sized. Args: driver: WebDriver
    instance mobile_threshold: Width threshold for mobile detection
    Returns: True if viewport width is below mobile threshold
    """
    try:
        viewport = get_actual_viewport_size(driver)
        return viewport["width"] < mobile_threshold

    except Exception:
        return False
