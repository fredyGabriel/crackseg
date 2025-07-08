"""
Smoke tests for the main GUI application and its navigation.

This test file ensures that the main application initializes correctly
and that navigating to each page via the sidebar does not cause errors.
"""

# Add project root to allow imports
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.gui.utils.gui_config import PAGE_CONFIG

# Get all page names from the central configuration
ALL_PAGE_NAMES = list(PAGE_CONFIG.keys())


@pytest.fixture
def app_test() -> AppTest:
    """Fixture to create a new AppTest instance for each test."""
    at = AppTest.from_file("scripts/gui/app.py", default_timeout=30)
    at.run()
    return at


def test_main_app_initializes_without_error(app_test: AppTest):
    """Test that the main application initializes without errors."""
    assert not app_test.exception, "The main app failed to initialize."


def test_sidebar_navigation_and_page_rendering(app_test: AppTest):
    """
    Simulates navigating to available pages from the sidebar and checks for
    basic rendering and absence of errors.

    Note: Only tests pages that have navigation buttons available.
    Pages may not have buttons if they don't meet their requirements.
    """
    # Get all available navigation buttons in the sidebar
    available_nav_buttons = []
    for button in app_test.sidebar.button:
        if button.key and button.key.startswith("nav_btn_"):
            page_name = button.key.replace("nav_btn_", "")
            available_nav_buttons.append(page_name)

    # Skip test if no navigation buttons are available
    if not available_nav_buttons:
        pytest.skip("No navigation buttons available for testing")

    # Test each available page
    for page_name in available_nav_buttons:
        if page_name == "Home":
            continue  # Skip the initial page

        button_key = f"nav_btn_{page_name}"
        try:
            # Find and click the navigation button in the sidebar
            nav_button = app_test.sidebar.button(key=button_key)
            nav_button.click().run()

            # 1. Check for any exceptions during the page run
            assert not app_test.exception, (
                f"Navigating to page '{page_name}' caused an exception: "
                f"{app_test.exception}"
            )

            # 2. Check that the page rendered *something* (a title or a header)
            has_title = len(app_test.title) > 0
            has_header = len(app_test.header) > 0
            assert (
                has_title or has_header
            ), f"Page '{page_name}' did not render a title or header."

        except Exception as e:
            pytest.fail(
                f"Failed while trying to navigate to page '{page_name}'. "
                f"Button key: '{button_key}'. Error: {e}"
            )
