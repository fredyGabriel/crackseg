"""
Touch simulation utilities for responsive testing. This module
provides functions for simulating touch interactions, gestures, and
mobile-specific user interactions for responsive design testing.
"""

import logging
import time

from selenium.common.exceptions import WebDriverException
from selenium.webdriver import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from tests.e2e.config.viewport_config import TouchCapability

logger = logging.getLogger(__name__)


def simulate_touch_tap(
    driver: WebDriver,
    element: WebElement,
    simulation_delay: float = 0.2,
) -> bool:
    """
    Simulate touch tap on an element. Args: driver: WebDriver instance
    element: Target element simulation_delay: Delay after touch simulation
    Returns: True if successful, False otherwise
    """
    try:
        logger.debug("Simulating touch tap")

        # Use ActionChains for touch-like interaction
        actions = ActionChains(driver)
        actions.move_to_element(element)
        actions.click(element)
        actions.perform()

        # Simulate touch delay
        if simulation_delay > 0:
            time.sleep(simulation_delay)

        # Trigger touch events via JavaScript
        _trigger_touch_events(driver, element, "tap")

        logger.debug("Touch tap simulated successfully")
        return True

    except WebDriverException as e:
        logger.error(f"Failed to simulate touch tap: {e}")
        return False


def simulate_swipe_gesture(
    driver: WebDriver,
    start_element: WebElement,
    direction: str,
    distance: int = 100,
    duration: float = 0.5,
) -> bool:
    """
    Simulate swipe gesture on an element. Args: driver: WebDriver instance
    start_element: Element to start swipe from direction: Swipe direction
    ('left', 'right', 'up', 'down') distance: Swipe distance in pixels
    duration: Duration of swipe gesture Returns: True if successful, False
    otherwise
    """
    try:
        logger.debug(f"Simulating {direction} swipe gesture")

        # Calculate swipe offsets
        offsets = _calculate_swipe_offsets(direction, distance)

        if not offsets:
            logger.error(f"Invalid swipe direction: {direction}")
            return False

        # Perform swipe using ActionChains
        actions = ActionChains(driver)
        actions.move_to_element(start_element)
        actions.click_and_hold(start_element)

        # Simulate drag motion
        steps = max(5, int(duration * 10))  # At least 5 steps
        step_x = offsets["x"] / steps
        step_y = offsets["y"] / steps
        step_delay = duration / steps

        for _i in range(steps):
            actions.move_by_offset(int(step_x), int(step_y))
            if step_delay > 0:
                time.sleep(step_delay / 2)  # Smaller delay for smooth motion

        actions.release()
        actions.perform()

        # Trigger touch events
        _trigger_touch_events(driver, start_element, "swipe", direction)

        logger.debug(f"Swipe gesture {direction} completed")
        return True

    except WebDriverException as e:
        logger.error(f"Failed to simulate swipe gesture: {e}")
        return False


def simulate_pinch_zoom(
    driver: WebDriver,
    element: WebElement,
    zoom_factor: float = 1.5,
    duration: float = 0.8,
) -> bool:
    """
    Simulate pinch zoom gesture on an element. Args: driver: WebDriver
    instance element: Target element zoom_factor: Zoom factor (>1 for zoom
    in, <1 for zoom out) duration: Duration of pinch gesture Returns: True
    if successful, False otherwise
    """
    try:
        logger.debug(f"Simulating pinch zoom (factor: {zoom_factor})")

        # JavaScript-based pinch simulation for better mobile compatibility
        pinch_script = f"""
var element = arguments[0]; var zoomFactor = {zoom_factor}; var
duration = {duration * 1000}; // Convert to milliseconds // Create
touch events var rect = element.getBoundingClientRect(); var centerX =
rect.left + rect.width / 2; var centerY = rect.top + rect.height / 2;
// Simulate pinch gesture var touchStartEvent = new
TouchEvent('touchstart', {{ bubbles: true, cancelable: true, touches:
[ new Touch({{ identifier: 0, target: element, clientX: centerX - 50,
clientY: centerY }}), new Touch({{ identifier: 1, target: element,
clientX: centerX + 50, clientY: centerY }}) ] }});
element.dispatchEvent(touchStartEvent); // Simulate zoom
setTimeout(function() {{ var scale = zoomFactor > 1 ? 'scale(' +
zoomFactor + ')' : 'scale(' + zoomFactor + ')';
element.style.transform = scale; var touchEndEvent = new
TouchEvent('touchend', {{ bubbles: true, cancelable: true,
changedTouches: [ new Touch({{ identifier: 0, target: element,
clientX: centerX - (50 * zoomFactor), clientY: centerY }}), new
Touch({{ identifier: 1, target: element, clientX: centerX + (50 *
zoomFactor), clientY: centerY }}) ] }});
element.dispatchEvent(touchEndEvent); }}, duration); return true;
"""

        result = driver.execute_script(pinch_script, element)

        # Wait for gesture completion
        time.sleep(duration + 0.2)

        logger.debug("Pinch zoom gesture completed")
        return bool(result)

    except WebDriverException as e:
        logger.error(f"Failed to simulate pinch zoom: {e}")
        return False


def simulate_long_press(
    driver: WebDriver,
    element: WebElement,
    press_duration: float = 1.0,
) -> bool:
    """
    Simulate long press gesture on an element. Args: driver: WebDriver
    instance element: Target element press_duration: Duration to hold
    press Returns: True if successful, False otherwise
    """
    try:
        logger.debug(f"Simulating long press ({press_duration}s)")

        # Use ActionChains for long press
        actions = ActionChains(driver)
        actions.move_to_element(element)
        actions.click_and_hold(element)
        actions.perform()

        # Hold for specified duration
        time.sleep(press_duration)

        # Release
        actions = ActionChains(driver)
        actions.release()
        actions.perform()

        # Trigger touch events
        _trigger_touch_events(driver, element, "longpress")

        logger.debug("Long press gesture completed")
        return True

    except WebDriverException as e:
        logger.error(f"Failed to simulate long press: {e}")
        return False


def _calculate_swipe_offsets(
    direction: str, distance: int
) -> dict[str, int] | None:
    """
    Calculate x,y offsets for swipe direction. Args: direction: Swipe
    direction distance: Swipe distance Returns: Dictionary with x,y
    offsets or None if invalid direction
    """
    direction_map = {
        "left": {"x": -distance, "y": 0},
        "right": {"x": distance, "y": 0},
        "up": {"x": 0, "y": -distance},
        "down": {"x": 0, "y": distance},
    }

    return direction_map.get(direction.lower())


def _trigger_touch_events(
    driver: WebDriver,
    element: WebElement,
    gesture_type: str,
    direction: str | None = None,
) -> None:
    """
    Trigger touch events to notify responsive JavaScript code. Args:
    driver: WebDriver instance element: Target element gesture_type: Type
    of gesture ('tap', 'swipe', 'longpress', 'pinch') direction: Direction
    for swipe gestures
    """
    try:
        touch_script = f"""
        var element = arguments[0];
        var gestureType = '{gesture_type}';
        var direction = '{direction or ""}';

        // Get element position
        var rect = element.getBoundingClientRect();
        var centerX = rect.left + rect.width / 2;
        var centerY = rect.top + rect.height / 2;

        // Create appropriate touch events based on gesture
        if (gestureType === 'tap') {{
            var touchEvent = new TouchEvent('touchstart', {{
                bubbles: true,
                cancelable: true,
                touches: [new Touch({{
                    identifier: 0,
                    target: element,
                    clientX: centerX,
                    clientY: centerY
                }})]
            }});
            element.dispatchEvent(touchEvent);

            setTimeout(function() {{
                var touchEndEvent = new TouchEvent('touchend', {{
                    bubbles: true,
                    cancelable: true,
                    changedTouches: [new Touch({{
                        identifier: 0,
                        target: element,
                        clientX: centerX,
                        clientY: centerY
                    }})]
                }});
                element.dispatchEvent(touchEndEvent);
            }}, 50);
        }}

        // Dispatch custom events for responsive frameworks
        element.dispatchEvent(new CustomEvent('responsive-' + gestureType, {{
            detail: {{ direction: direction, element: element }},
            bubbles: true
        }}));

        return true;
        """

        driver.execute_script(touch_script, element)

    except WebDriverException as e:
        logger.warning(f"Failed to trigger touch events: {e}")


def check_touch_capability(driver: WebDriver) -> TouchCapability:
    """
    Check touch capability of current browser/device. Args: driver:
    WebDriver instance Returns: TouchCapability level detected
    """
    try:
        capability_script = """
// Check for touch support var hasTouch = 'ontouchstart' in window ||
navigator.maxTouchPoints > 0 || navigator.msMaxTouchPoints > 0; //
Check for advanced gesture support var hasAdvancedTouch = 'TouchEvent'
in window && 'ontouchstart' in window; // Check for pointer events
(advanced touch) var hasPointerEvents = 'PointerEvent' in window;
return { hasTouch: hasTouch, hasAdvancedTouch: hasAdvancedTouch,
hasPointerEvents: hasPointerEvents, maxTouchPoints:
navigator.maxTouchPoints || 0 };
"""

        result = driver.execute_script(capability_script)

        if result["hasAdvancedTouch"] and result["maxTouchPoints"] > 1:
            return TouchCapability.ADVANCED
        elif result["hasTouch"]:
            return TouchCapability.BASIC
        else:
            return TouchCapability.NONE

    except WebDriverException:
        logger.warning("Could not determine touch capability")
        return TouchCapability.NONE


def set_mobile_user_agent(driver: WebDriver, user_agent: str) -> bool:
    """
    Set mobile user agent for touch simulation. Args: driver: WebDriver
    instance user_agent: Mobile user agent string Returns: True if
    successful, False otherwise
    """
    try:
        # This typically needs to be set during driver initialization
        # For runtime changes, we use CDP if available
        if hasattr(driver, "execute_cdp_cmd"):
            driver.execute_cdp_cmd(
                "Network.setUserAgentOverride", {"userAgent": user_agent}
            )
            logger.debug(f"User agent set to: {user_agent[:50]}...")
            return True
        else:
            logger.warning("Cannot change user agent at runtime")
            return False

    except Exception as e:
        logger.error(f"Failed to set user agent: {e}")
        return False
