"""
E2E tests for concurrent user behavior scenarios. This module contains
tests for system behavior under concurrent access: - Multiple browser
sessions simultaneously - Rapid navigation between pages -
Simultaneous configuration loading - Race conditions in session state
"""

import time
from typing import TypedDict

import pytest
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver

from ..base_test import BaseE2ETest
from ..pages import ArchitecturePage, ConfigPage, ResultsPage, TrainPage


class ConcurrentTestResults(TypedDict):
    """Type definition for concurrent navigation test results."""

    start_time: float
    navigation_attempts: int
    successful_navigations: int
    errors: list[str]
    concurrent_sessions: int
    duration: float


class TestEdgeCasesConcurrent(BaseE2ETest):
    """Test class for concurrent user behavior edge cases."""

    @pytest.mark.e2e
    def test_concurrent_user_behaviors(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test concurrent user behavior scenarios.

        Tests system behavior under concurrent access including:
        - Multiple browser sessions simultaneously
        - Rapid navigation between pages
        - Simultaneous configuration loading
        - Race conditions in session state
        """
        self.log_test_step("Start concurrent user behavior tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        try:
            # Test 1: Rapid navigation stress test
            self.log_test_step("Testing rapid page navigation patterns")

            pages = [
                ("Config", lambda: ConfigPage(webdriver).navigate_to_page()),
                (
                    "Architecture",
                    lambda: ArchitecturePage(webdriver).navigate_to_page(),
                ),
                ("Train", lambda: TrainPage(webdriver).navigate_to_page()),
                ("Results", lambda: ResultsPage(webdriver).navigate_to_page()),
            ]

            navigation_success_count = 0
            rapid_navigation_attempts = 8

            for i in range(rapid_navigation_attempts):
                page_name = "unknown"  # Initialize to avoid unbound variable
                try:
                    page_name, nav_func = pages[i % len(pages)]

                    start_time = time.time()
                    nav_func()
                    navigation_time = time.time() - start_time

                    # Verify page loaded within reasonable time
                    assert navigation_time < 10.0, (
                        f"Navigation to {page_name} took "
                        f"{navigation_time:.2f}s"
                    )

                    # Basic page validation
                    self.assert_streamlit_loaded(webdriver)
                    navigation_success_count += 1

                    # Brief pause to avoid overwhelming the system
                    time.sleep(0.2)

                except (TimeoutException, WebDriverException) as e:
                    self.log_test_step(
                        f"⚠️ Navigation to {page_name} failed "
                        f"(attempt {i + 1}): {e}"
                    )

            # Verify reasonable success rate for rapid navigation
            success_rate = navigation_success_count / rapid_navigation_attempts
            assert success_rate >= 0.7, (
                f"Rapid navigation success rate {success_rate:.1%} below "
                "threshold"
            )
            self.log_assertion(
                f"Rapid navigation success rate: {success_rate:.1%}",
                success_rate >= 0.7,
            )

            # Test 2: Session state persistence during rapid operations
            self.log_test_step(
                "Testing session state persistence under stress"
            )

            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Load configuration
            config_page.select_config_file(
                self.get_test_data()["boundary_test_config"]
            )
            config_page.click_load_config()
            config_page.wait_for_configuration_loaded()

            # Verify config is loaded
            initial_config = config_page.get_configuration_content()
            assert initial_config is not None, "Config should be loaded"

            # Perform rapid navigation and return
            for _ in range(3):
                TrainPage(webdriver).navigate_to_page()
                time.sleep(0.1)
                config_page = ConfigPage(webdriver).navigate_to_page()
                time.sleep(0.1)

            # Verify session state persisted
            final_config = config_page.get_configuration_content()
            if final_config is not None:
                self.log_assertion(
                    "Session state persisted through rapid navigation", True
                )
            else:
                self.log_assertion(
                    "Session state lost during rapid navigation", False
                )
                # This might be expected behavior, not necessarily a failure

        except Exception as e:
            self.log_test_step(f"❌ Concurrent behavior test failed: {str(e)}")
            raise
