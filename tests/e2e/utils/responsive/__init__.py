"""Responsive testing utilities package.

This package provides comprehensive utilities for testing responsive web
designs across multiple viewports, devices, and interaction patterns.

Main components:
- viewport: Viewport manipulation and sizing utilities
- touch: Touch interaction simulation for mobile testing
- layout: Layout validation and positioning checks
- testing: Complete test workflow orchestration

Example usage:
    from tests.e2e.utils.responsive import (
        execute_responsive_test_suite,
        set_viewport_size,
        simulate_touch_tap,
        validate_responsive_layout
    )

    # Execute full responsive test suite
    results = execute_responsive_test_suite(
        driver, test_matrix, "http://localhost:3000"
    )

    # Individual viewport testing
    success = set_viewport_size(driver, mobile_viewport)
    layout_valid = validate_responsive_layout(driver, layout_checks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver

# Viewport utilities
# Layout validation utilities
from tests.e2e.utils.responsive.layout import (
    capture_layout_screenshot,
    check_horizontal_scrolling,
    get_layout_metrics,
    validate_responsive_layout,
)

# Testing workflow utilities
from tests.e2e.utils.responsive.testing import (
    create_test_configuration,
    execute_responsive_test_suite,
    generate_test_report,
)

# Touch interaction utilities
from tests.e2e.utils.responsive.touch import (
    check_touch_capability,
    set_mobile_user_agent,
    simulate_long_press,
    simulate_pinch_zoom,
    simulate_swipe_gesture,
    simulate_touch_tap,
)
from tests.e2e.utils.responsive.viewport import (
    get_actual_viewport_size,
    get_device_pixel_ratio,
    is_mobile_viewport,
    set_viewport_size,
    simulate_orientation_change,
    wait_for_viewport_stabilization,
)

# Public API exports
__all__ = [
    # Viewport utilities
    "get_actual_viewport_size",
    "get_device_pixel_ratio",
    "is_mobile_viewport",
    "set_viewport_size",
    "simulate_orientation_change",
    "wait_for_viewport_stabilization",
    # Touch interaction utilities
    "check_touch_capability",
    "set_mobile_user_agent",
    "simulate_long_press",
    "simulate_pinch_zoom",
    "simulate_swipe_gesture",
    "simulate_touch_tap",
    # Layout validation utilities
    "capture_layout_screenshot",
    "check_horizontal_scrolling",
    "get_layout_metrics",
    "validate_responsive_layout",
    # Testing workflow utilities
    "create_test_configuration",
    "execute_responsive_test_suite",
    "generate_test_report",
]

# Version and package metadata
__version__ = "1.0.0"
__author__ = "CrackSeg Testing Team"
__description__ = "Responsive testing utilities for cross-browser validation"


# Quick access to commonly used functions
def quick_mobile_test(
    driver: WebDriver, viewport_width: int = 375, viewport_height: int = 667
) -> bool:
    """Quick mobile viewport test setup.

    Args:
        driver: WebDriver instance
        viewport_width: Mobile viewport width (default: iPhone SE)
        viewport_height: Mobile viewport height (default: iPhone SE)

    Returns:
        True if mobile viewport set successfully
    """
    from tests.e2e.config.viewport_config import ViewportDimensions

    mobile_viewport = ViewportDimensions(viewport_width, viewport_height)
    return set_viewport_size(driver, mobile_viewport)


def quick_desktop_test(
    driver: WebDriver,
    viewport_width: int = 1920,
    viewport_height: int = 1080,
) -> bool:
    """Quick desktop viewport test setup.

    Args:
        driver: WebDriver instance
        viewport_width: Desktop viewport width (default: Full HD)
        viewport_height: Desktop viewport height (default: Full HD)

    Returns:
        True if desktop viewport set successfully
    """
    from tests.e2e.config.viewport_config import ViewportDimensions

    desktop_viewport = ViewportDimensions(viewport_width, viewport_height)
    return set_viewport_size(driver, desktop_viewport)


def quick_responsive_check(
    driver: WebDriver, base_url: str, devices: list[Any] | None = None
) -> dict[str, Any]:
    """Quick responsive check across common devices.

    Args:
        driver: WebDriver instance
        base_url: URL to test
        devices: List of device configurations (uses defaults if None)

    Returns:
        Dictionary with test results
    """
    from tests.e2e.config.viewport_config import get_mobile_focused_matrix

    if devices is None:
        # Use mobile-focused matrix for quick testing
        test_matrix = get_mobile_focused_matrix()
    else:
        from tests.e2e.config.viewport_config import ResponsiveTestMatrix

        test_matrix = ResponsiveTestMatrix("Custom Matrix", devices)

    test_config = create_test_configuration(
        screenshot_dir="test-artifacts/quick-responsive",
        stabilization_delay=0.5,  # Faster for quick testing
    )

    return execute_responsive_test_suite(
        driver, test_matrix, base_url, test_config
    )


def create_basic_layout_checks() -> dict[str, Any]:
    """Create basic layout validation checks for common responsive patterns.

    Returns:
        Dictionary with standard layout validation configuration
    """
    return {
        "nav_visible": {
            "selector": "nav, .navbar, .navigation",
            "visible": True,
        },
        "mobile_menu": {
            "hamburger_selector": ".hamburger, .menu-toggle, .mobile-menu-btn",
            "menu_selector": ".mobile-menu, .nav-menu, .navbar-collapse",
        },
        "content_stacking": {
            "selectors": [".main-content", ".sidebar", ".content-area"]
        },
        "text_readability": {
            "min_font_size": 14,
            "selectors": ["p", "span", ".text", ".content"],
        },
    }


# Convenience functions for common testing patterns
def test_mobile_navigation(
    driver: WebDriver, hamburger_selector: str = ".hamburger"
) -> bool:
    """Test mobile navigation functionality.

    Args:
        driver: WebDriver instance
        hamburger_selector: CSS selector for hamburger menu button

    Returns:
        True if mobile navigation works correctly
    """
    # Set mobile viewport
    mobile_success = quick_mobile_test(driver)
    if not mobile_success:
        return False

    # Try to find and interact with hamburger menu
    try:
        from selenium.webdriver.common.by import By

        hamburger_elements = driver.find_elements(
            By.CSS_SELECTOR, hamburger_selector
        )

        if not hamburger_elements:
            return True  # No hamburger menu found, which is acceptable

        hamburger = hamburger_elements[0]
        if hamburger.is_displayed():
            # Simulate touch tap on hamburger
            return simulate_touch_tap(driver, hamburger)

        return True

    except Exception:
        return False


def test_touch_interactions(
    driver: WebDriver, test_element_selector: str = "body"
) -> dict[str, Any]:
    """Test basic touch interactions on an element.

    Args:
        driver: WebDriver instance
        test_element_selector: CSS selector for element to test

    Returns:
        Dictionary with touch test results
    """
    try:
        from selenium.webdriver.common.by import By

        elements = driver.find_elements(By.CSS_SELECTOR, test_element_selector)

        if not elements:
            return {"error": "Test element not found"}

        test_element = elements[0]

        results = {
            "touch_capability": check_touch_capability(driver),
            "tap_test": simulate_touch_tap(driver, test_element),
            "long_press_test": simulate_long_press(driver, test_element, 1.0),
            "swipe_test": simulate_swipe_gesture(
                driver, test_element, "left", 100
            ),
        }

        return results

    except Exception as e:
        return {"error": str(e)}


def validate_page_responsiveness(
    driver: WebDriver, layout_checks: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate overall page responsiveness with standard checks.

    Args:
        driver: WebDriver instance
        layout_checks: Custom layout checks (uses defaults if None)

    Returns:
        Dictionary with validation results
    """
    if layout_checks is None:
        layout_checks = create_basic_layout_checks()

    results = {
        "layout_validation": validate_responsive_layout(driver, layout_checks),
        "horizontal_scroll": check_horizontal_scrolling(driver),
        "layout_metrics": get_layout_metrics(driver),
    }

    return results
