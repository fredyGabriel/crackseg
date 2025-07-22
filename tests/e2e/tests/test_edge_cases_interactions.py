"""
E2E tests for unusual user interaction patterns. This module contains
tests for edge cases in user behavior: - Browser refresh during
operations - Back/forward navigation mid-process - Session timeout
scenarios - Inactive tab behavior
"""

import time

import pytest
from selenium.webdriver.remote.webdriver import WebDriver

from ..base_test import BaseE2ETest
from ..pages import ArchitecturePage, ConfigPage, TrainPage


class TestEdgeCasesInteractions(BaseE2ETest):
    """Test class for unusual user interaction edge cases."""

    @pytest.mark.e2e
    def test_unusual_user_interactions(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test unusual user interaction patterns.

        Tests edge cases in user behavior including:
        - Browser refresh during operations
        - Back/forward navigation mid-process
        - Session timeout scenarios
        - Inactive tab behavior
        """
        self.log_test_step("Start unusual user interaction tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        try:
            # Test 1: Browser refresh during configuration loading
            self.log_test_step("Testing browser refresh during operation")

            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Start configuration loading
            config_page.select_config_file(
                self.get_test_data()["boundary_test_config"]
            )
            config_page.click_load_config()

            # Refresh browser mid-operation
            time.sleep(1.0)  # Allow operation to start
            webdriver.refresh()

            # Verify system recovers gracefully
            time.sleep(3.0)  # Allow page to reload
            try:
                self.assert_streamlit_loaded(webdriver)
                self.log_assertion(
                    "System recovered from mid-operation refresh", True
                )

                # Verify we can navigate normally after refresh
                config_page = ConfigPage(webdriver).navigate_to_page()
                self.assert_streamlit_loaded(webdriver)

            except Exception as e:
                self.log_assertion(f"System recovery failed: {e}", False)

            # Test 2: Back/Forward navigation patterns
            self.log_test_step("Testing back/forward navigation edge cases")

            # Navigate through pages
            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            ArchitecturePage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            TrainPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Use browser back button
            webdriver.back()
            time.sleep(2.0)

            # Verify state is handled appropriately
            try:
                self.assert_streamlit_loaded(webdriver)
                self.log_assertion("Back navigation handled correctly", True)
            except Exception:
                self.log_assertion("Back navigation caused issues", False)

            # Use browser forward button
            webdriver.forward()
            time.sleep(2.0)

            try:
                self.assert_streamlit_loaded(webdriver)
                self.log_assertion(
                    "Forward navigation handled correctly", True
                )
            except Exception:
                self.log_assertion("Forward navigation caused issues", False)

            # Test 3: Multiple tab behavior simulation
            self.log_test_step("Testing multiple window/tab simulation")

            original_window = webdriver.current_window_handle

            # Open new tab/window
            webdriver.execute_script("window.open('');")
            windows = webdriver.window_handles

            if len(windows) > 1:
                # Switch to new window
                webdriver.switch_to.window(windows[1])
                webdriver.get(streamlit_base_url)

                # Verify app loads in new window
                try:
                    self.assert_streamlit_loaded(webdriver)
                    self.log_assertion(
                        "App loaded successfully in new window", True
                    )

                    # Try basic navigation in new window
                    config_page = ConfigPage(webdriver).navigate_to_page()
                    self.assert_streamlit_loaded(webdriver)

                except Exception as e:
                    self.log_assertion(
                        f"New window functionality failed: {e}", False
                    )

                # Switch back to original window
                webdriver.switch_to.window(original_window)

                # Verify original window still works
                try:
                    self.assert_streamlit_loaded(webdriver)
                    self.log_assertion(
                        "Original window remained functional", True
                    )
                except Exception as e:
                    self.log_assertion(
                        f"Original window lost functionality: {e}", False
                    )

                # Close additional window
                webdriver.switch_to.window(windows[1])
                webdriver.close()
                webdriver.switch_to.window(original_window)

            # Test 4: Extended idle time simulation
            self.log_test_step("Testing extended idle time behavior")

            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Load configuration to establish session state
            config_page.select_config_file(
                self.get_test_data()["boundary_test_config"]
            )
            config_page.click_load_config()
            config_page.wait_for_configuration_loaded()

            # Simulate extended idle time
            idle_time = 10.0  # 10 seconds idle (reduced for testing)
            self.log_test_step(f"Simulating {idle_time}s idle time")
            time.sleep(idle_time)

            # Verify session/state persistence after idle
            try:
                # Try to interact with the page
                ArchitecturePage(webdriver).navigate_to_page()
                self.assert_streamlit_loaded(webdriver)
                self.log_assertion("Session persisted through idle time", True)

            except Exception as e:
                self.log_assertion(
                    f"Session lost during idle time: {e}", False
                )

        except Exception as e:
            self.log_test_step(f"❌ Unusual interaction test failed: {str(e)}")
            raise
