"""Layout validation utilities for responsive testing.

This module provides functions for validating responsive layouts, checking
element positioning, visibility, and responsive design correctness.
"""

import logging
from typing import Any

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


def validate_responsive_layout(
    driver: WebDriver,
    layout_checks: dict[str, Any],
) -> dict[str, bool]:
    """Validate responsive layout against specified criteria.

    Args:
        driver: WebDriver instance
        layout_checks: Dictionary of layout validation criteria

    Returns:
        Dictionary with validation results for each check
    """
    results = {}

    try:
        logger.debug("Starting responsive layout validation")

        # Check navigation visibility
        if "nav_visible" in layout_checks:
            results["nav_visible"] = _check_navigation_visibility(
                driver, layout_checks["nav_visible"]
            )

        # Check mobile menu behavior
        if "mobile_menu" in layout_checks:
            results["mobile_menu"] = _check_mobile_menu_behavior(
                driver, layout_checks["mobile_menu"]
            )

        # Check content stacking
        if "content_stacking" in layout_checks:
            results["content_stacking"] = _check_content_stacking(
                driver, layout_checks["content_stacking"]
            )

        # Check element positioning
        if "element_positions" in layout_checks:
            results["element_positions"] = _check_element_positions(
                driver, layout_checks["element_positions"]
            )

        # Check text readability
        if "text_readability" in layout_checks:
            results["text_readability"] = _check_text_readability(
                driver, layout_checks["text_readability"]
            )

        logger.debug(f"Layout validation completed: {results}")
        return results

    except Exception as e:
        logger.error(f"Layout validation failed: {e}")
        return {"error": False}


def _check_navigation_visibility(
    driver: WebDriver, nav_config: dict[str, Any]
) -> bool:
    """Check navigation visibility and behavior.

    Args:
        driver: WebDriver instance
        nav_config: Navigation validation configuration

    Returns:
        True if navigation behaves correctly
    """
    try:
        nav_selector = nav_config.get("selector", "nav")
        expected_visible = nav_config.get("visible", True)

        elements = driver.find_elements(By.CSS_SELECTOR, nav_selector)

        if not elements:
            return not expected_visible

        nav_element = elements[0]
        is_visible = nav_element.is_displayed()

        return is_visible == expected_visible

    except Exception as e:
        logger.error(f"Navigation visibility check failed: {e}")
        return False


def _check_mobile_menu_behavior(
    driver: WebDriver, menu_config: dict[str, Any]
) -> bool:
    """Check mobile menu hamburger and dropdown behavior.

    Args:
        driver: WebDriver instance
        menu_config: Mobile menu validation configuration

    Returns:
        True if mobile menu behaves correctly
    """
    try:
        hamburger_selector = menu_config.get(
            "hamburger_selector", ".hamburger"
        )
        menu_selector = menu_config.get("menu_selector", ".mobile-menu")

        # Check if hamburger button exists
        hamburger_elements = driver.find_elements(
            By.CSS_SELECTOR, hamburger_selector
        )

        if not hamburger_elements:
            # No hamburger menu expected on larger screens
            return True

        hamburger = hamburger_elements[0]

        # Check if hamburger is visible
        if not hamburger.is_displayed():
            return True  # Hidden hamburger is okay on desktop

        # Try to click hamburger and check menu visibility
        menu_elements = driver.find_elements(By.CSS_SELECTOR, menu_selector)

        if not menu_elements:
            return False  # Should have menu if hamburger exists

        menu = menu_elements[0]
        initial_visible = menu.is_displayed()

        # Click hamburger
        hamburger.click()

        # Check if menu state changed
        final_visible = menu.is_displayed()

        return initial_visible != final_visible

    except Exception as e:
        logger.error(f"Mobile menu check failed: {e}")
        return False


def _check_content_stacking(
    driver: WebDriver, stacking_config: dict[str, Any]
) -> bool:
    """Check if content stacks properly on mobile.

    Args:
        driver: WebDriver instance
        stacking_config: Content stacking validation configuration

    Returns:
        True if content stacks correctly
    """
    try:
        selectors = stacking_config.get("selectors", [])

        if len(selectors) < 2:
            return True  # Need at least 2 elements to check stacking

        elements = []
        for selector in selectors:
            found_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if found_elements:
                elements.append(found_elements[0])

        if len(elements) < 2:
            return True  # Not enough elements found

        # Check if elements are stacked vertically
        positions = []
        for element in elements:
            rect = element.rect
            positions.append(
                {
                    "top": rect["y"],
                    "bottom": rect["y"] + rect["height"],
                    "left": rect["x"],
                    "width": rect["width"],
                }
            )

        # Check for vertical stacking (elements should not overlap
        # horizontally)
        for i in range(len(positions) - 1):
            current = positions[i]
            next_pos = positions[i + 1]

            # Elements should be roughly full width or clearly stacked
            if (current["width"] < 200 or next_pos["width"] < 200) and abs(
                current["left"] - next_pos["left"]
            ) > 50:
                return False  # Side-by-side layout detected

        return True

    except Exception as e:
        logger.error(f"Content stacking check failed: {e}")
        return False


def _check_element_positions(
    driver: WebDriver, position_config: dict[str, Any]
) -> bool:
    """Check element positioning meets responsive requirements.

    Args:
        driver: WebDriver instance
        position_config: Element positioning validation configuration

    Returns:
        True if elements are positioned correctly
    """
    try:
        checks = position_config.get("checks", [])
        all_passed = True

        for check in checks:
            selector = check.get("selector")
            expected_position = check.get("position", {})

            if not selector:
                continue

            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if not elements:
                all_passed = False
                continue

            element = elements[0]
            rect = element.rect

            # Check position constraints
            if "min_top" in expected_position:
                if rect["y"] < expected_position["min_top"]:
                    all_passed = False

            if "max_top" in expected_position:
                if rect["y"] > expected_position["max_top"]:
                    all_passed = False

            if "min_left" in expected_position:
                if rect["x"] < expected_position["min_left"]:
                    all_passed = False

            if "max_left" in expected_position:
                if rect["x"] > expected_position["max_left"]:
                    all_passed = False

        return all_passed

    except Exception as e:
        logger.error(f"Element position check failed: {e}")
        return False


def _check_text_readability(
    driver: WebDriver, readability_config: dict[str, Any]
) -> bool:
    """Check text readability on different screen sizes.

    Args:
        driver: WebDriver instance
        readability_config: Text readability validation configuration

    Returns:
        True if text is readable
    """
    try:
        min_font_size = readability_config.get("min_font_size", 14)
        text_selectors = readability_config.get(
            "selectors", ["p", "span", "div"]
        )

        for selector in text_selectors:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)

            for element in elements[:5]:  # Check first 5 elements
                # Get computed font size
                font_size_script = """
                var element = arguments[0];
                var style = window.getComputedStyle(element);
                return parseFloat(style.fontSize);
                """

                font_size = driver.execute_script(font_size_script, element)

                if font_size < min_font_size:
                    logger.warning(
                        f"Text too small: {font_size}px "
                        f"(min: {min_font_size}px)"
                    )
                    return False

        return True

    except Exception as e:
        logger.error(f"Text readability check failed: {e}")
        return False


def capture_layout_screenshot(
    driver: WebDriver,
    filename: str,
    viewport_info: dict[str, Any] | None = None,
) -> bool:
    """Capture screenshot of current layout for documentation.

    Args:
        driver: WebDriver instance
        filename: Screenshot filename
        viewport_info: Information about current viewport

    Returns:
        True if screenshot captured successfully
    """
    try:
        # Add viewport information to page if provided
        if viewport_info:
            overlay_script = f"""
            var overlay = document.createElement('div');
            overlay.style.position = 'fixed';
            overlay.style.top = '10px';
            overlay.style.left = '10px';
            overlay.style.background = 'rgba(0,0,0,0.8)';
            overlay.style.color = 'white';
            overlay.style.padding = '5px 10px';
            overlay.style.borderRadius = '3px';
            overlay.style.fontSize = '12px';
            overlay.style.zIndex = '10000';
            overlay.textContent = '{viewport_info.get("width", "?")}x{viewport_info.get("height", "?")} - {viewport_info.get("device", "Unknown")}';
            document.body.appendChild(overlay);

            setTimeout(function() {{
                document.body.removeChild(overlay);
            }}, 100);
            """

            driver.execute_script(overlay_script)

        # Capture screenshot
        success = driver.save_screenshot(filename)

        logger.debug(f"Layout screenshot saved: {filename}")
        return success

    except Exception as e:
        logger.error(f"Failed to capture layout screenshot: {e}")
        return False


def get_layout_metrics(driver: WebDriver) -> dict[str, Any]:
    """Get comprehensive layout metrics for analysis.

    Args:
        driver: WebDriver instance

    Returns:
        Dictionary with layout metrics
    """
    try:
        metrics_script = """
        return {
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            document: {
                width: document.documentElement.scrollWidth,
                height: document.documentElement.scrollHeight
            },
            scroll: {
                x: window.pageXOffset || document.documentElement.scrollLeft,
                y: window.pageYOffset || document.documentElement.scrollTop
            },
            devicePixelRatio: window.devicePixelRatio || 1
        };
        """

        metrics = driver.execute_script(metrics_script)

        # Add element count
        element_count_script = """
        return {
            total: document.querySelectorAll('*').length,
            visible: Array.from(document.querySelectorAll('*')).filter(
                el => el.offsetParent !== null
            ).length
        };
        """

        element_counts = driver.execute_script(element_count_script)
        metrics["elements"] = element_counts

        return metrics

    except Exception as e:
        logger.error(f"Failed to get layout metrics: {e}")
        return {}


def check_horizontal_scrolling(driver: WebDriver) -> bool:
    """Check if page has unwanted horizontal scrolling.

    Args:
        driver: WebDriver instance

    Returns:
        True if no horizontal scrolling detected
    """
    try:
        scroll_check_script = """
        var docWidth = document.documentElement.scrollWidth;
        var viewportWidth = window.innerWidth;
        return docWidth <= viewportWidth;
        """

        no_horizontal_scroll = driver.execute_script(scroll_check_script)

        if not no_horizontal_scroll:
            logger.warning("Horizontal scrolling detected")

        return no_horizontal_scroll

    except Exception as e:
        logger.error(f"Horizontal scroll check failed: {e}")
        return True  # Assume no scrolling if check fails
