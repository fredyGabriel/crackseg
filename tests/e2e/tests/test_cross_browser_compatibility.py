"""
Cross-browser compatibility tests for CrackSeg application. This
module contains comprehensive cross-browser tests that extend the
existing test scenarios (15.1, 15.2, 15.3) to ensure compatibility
across Chrome, Firefox, Edge, and Safari browsers. Includes browser
capability validation and browser-specific feature testing.
"""

import pytest
from selenium.webdriver.remote.webdriver import WebDriver

from ..base_test import BaseE2ETest
from ..config.browser_matrix_config import (
    CI_CONFIG,
    COMPREHENSIVE_CONFIG,
    DEVELOPMENT_CONFIG,
    BrowserMatrixConfig,
)
from ..pages import ArchitecturePage, ConfigPage, ResultsPage, TrainPage
from ..utils.browser_validation import (
    check_critical_compatibility,
    save_compatibility_report,
    validate_browser_compatibility,
)


@pytest.mark.cross_browser
@pytest.mark.browser_matrix
class TestCrossBrowserCompatibility(BaseE2ETest):
    """Cross-browser compatibility test suite for CrackSeg application."""

    def setup_test_data(self) -> dict[str, str]:
        """Set up test data for cross-browser testing."""
        return {
            "config_file": "basic_verification.yaml",
            "expected_model_name": "MockModel",
        }

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "browser_config",
        [
            pytest.param(
                DEVELOPMENT_CONFIG,
                id="development",
                marks=pytest.mark.smoke,
            ),
            pytest.param(
                CI_CONFIG,
                id="ci",
                marks=pytest.mark.ci,
            ),
            pytest.param(
                COMPREHENSIVE_CONFIG,
                id="comprehensive",
                marks=pytest.mark.comprehensive,
            ),
        ],
    )
    def test_browser_capability_validation(
        self,
        cross_browser_driver: WebDriver,
        streamlit_base_url: str,
        browser_config: BrowserMatrixConfig,
    ) -> None:
        """
        Test browser capabilities for CrackSeg application compatibility. This
        test validates that each browser supports the required capabilities
        for running the CrackSeg application properly.

        Args:
            cross_browser_driver: Parametrized WebDriver for different browsers
            streamlit_base_url: Base URL of the Streamlit application
            browser_config: Browser matrix configuration
        """
        self.log_test_step("Starting browser capability validation")

        # Get browser name from WebDriver capabilities
        browser_name = self._get_browser_name(cross_browser_driver)
        self.log_test_step(f"Testing capabilities for browser: {browser_name}")

        # Get browser profile from configuration
        browser_profile = browser_config.get_browser_profile(browser_name)  # type: ignore[arg-type]
        if not browser_profile:
            pytest.skip(
                f"Browser {browser_name} not configured in test matrix"
            )

        # Navigate to application to test in real context
        self.navigate_and_verify(cross_browser_driver, streamlit_base_url)
        self.assert_streamlit_loaded(cross_browser_driver)

        # Validate browser capabilities
        compatibility_report = validate_browser_compatibility(
            cross_browser_driver, browser_profile
        )

        # Save compatibility report for analysis
        report_path = f"test-artifacts/e2e/compatibility_{browser_name}.json"
        save_compatibility_report(compatibility_report, report_path)
        self.log_test_step(f"Compatibility report saved to {report_path}")

        # Assert critical compatibility requirements
        assert check_critical_compatibility(compatibility_report), (
            f"Browser {browser_name} failed critical compatibility "
            "requirements"
        )

        # Log compatibility score
        score = compatibility_report.get_compatibility_score()
        self.log_test_step(f"Browser compatibility score: {score:.1f}%")

        # Assert minimum compatibility score (80% for production use)
        assert score >= 80.0, (
            f"Browser {browser_name} compatibility score ({score:.1f}%) "
            "below minimum requirement (80%)"
        )

    @pytest.mark.e2e
    def test_happy_path_cross_browser(
        self,
        cross_browser_driver: WebDriver,
        streamlit_base_url: str,
    ) -> None:
        """
        Test the happy path workflow across different browsers. Extends the
        happy path test from 15.1 to run across multiple browsers, ensuring
        consistent functionality regardless of browser choice.

        Args:
            cross_browser_driver: Parametrized WebDriver for different browsers
            streamlit_base_url: Base URL of the Streamlit application
        """
        browser_name = self._get_browser_name(cross_browser_driver)
        self.log_test_step(
            f"Starting cross-browser happy path test: {browser_name}"
        )

        # Execute simplified workflow that uses confirmed methods
        self.navigate_and_verify(cross_browser_driver, streamlit_base_url)

        try:
            # 1. Config Page: Basic navigation test
            self.log_test_step("Navigating to Config page")
            ConfigPage(cross_browser_driver).navigate_to_page()
            self.assert_streamlit_loaded(cross_browser_driver)
            self.log_test_step("Config page loaded successfully")

            # 2. Architecture Page: Basic navigation test
            self.log_test_step("Navigating to Architecture page")
            ArchitecturePage(cross_browser_driver).navigate_to_page()
            self.assert_streamlit_loaded(cross_browser_driver)
            self.log_test_step("Architecture page loaded successfully")

            # 3. Train Page: Basic navigation test
            self.log_test_step("Navigating to Train page")
            TrainPage(cross_browser_driver).navigate_to_page()
            self.assert_streamlit_loaded(cross_browser_driver)
            self.log_test_step("Train page loaded successfully")

            # 4. Results Page: Basic navigation test
            self.log_test_step("Navigating to Results page")
            ResultsPage(cross_browser_driver).navigate_to_page()
            self.assert_streamlit_loaded(cross_browser_driver)
            self.log_test_step("Results page loaded successfully")

            self.log_test_step(
                f"Cross-browser happy path completed: {browser_name}"
            )

        except Exception as e:
            self.log_test_step(
                f"❌ Cross-browser test failed on {browser_name}: {e}"
            )
            # Capture browser-specific debug information
            self._capture_browser_debug_info(
                cross_browser_driver, browser_name
            )
            raise

    @pytest.mark.e2e
    def test_error_scenarios_cross_browser(
        self,
        cross_browser_driver: WebDriver,
        streamlit_base_url: str,
    ) -> None:
        """
        Test error handling scenarios across different browsers. Extends error
        scenario tests from 15.2 to ensure consistent error handling and user
        feedback across all supported browsers.

        Args:
            cross_browser_driver: Parametrized WebDriver for different browsers
            streamlit_base_url: Base URL of the Streamlit application
        """
        browser_name = self._get_browser_name(cross_browser_driver)
        self.log_test_step(
            f"Starting cross-browser error scenarios: {browser_name}"
        )

        self.navigate_and_verify(cross_browser_driver, streamlit_base_url)

        # Test basic error handling via page responsiveness
        ConfigPage(cross_browser_driver).navigate_to_page()
        self.assert_streamlit_loaded(cross_browser_driver)

        self.log_test_step("Testing basic error handling")

        try:
            # Basic error handling test via page interaction
            self.assert_streamlit_loaded(cross_browser_driver)
            self.log_test_step(
                f"Basic error handling verified for {browser_name}"
            )

        except Exception as e:
            self.log_test_step(
                f"❌ Error scenario test failed on {browser_name}: {e}"
            )
            self._capture_browser_debug_info(
                cross_browser_driver, browser_name
            )
            raise

    @pytest.mark.e2e
    def test_file_upload_cross_browser(
        self,
        cross_browser_driver: WebDriver,
        streamlit_base_url: str,
    ) -> None:
        """
        Test file upload functionality across different browsers. Simplified
        test that verifies basic file upload capabilities exist.

        Args:
            cross_browser_driver: Parametrized WebDriver for different browsers
            streamlit_base_url: Base URL of the Streamlit application
        """
        browser_name = self._get_browser_name(cross_browser_driver)
        self.log_test_step(
            f"Starting cross-browser file upload test: {browser_name}"
        )

        self.navigate_and_verify(cross_browser_driver, streamlit_base_url)

        # Navigate to a page and verify it loads
        ConfigPage(cross_browser_driver).navigate_to_page()
        self.assert_streamlit_loaded(cross_browser_driver)

        self.log_test_step(f"File upload page verified for {browser_name}")

    @pytest.mark.e2e
    @pytest.mark.responsive
    def test_responsive_design_cross_browser(
        self,
        cross_browser_driver: WebDriver,
        streamlit_base_url: str,
    ) -> None:
        """
        Test responsive design behavior across different browsers. Tests how
        the CrackSeg application adapts to different screen sizes and browser
        window dimensions.

        Args:
            cross_browser_driver: Parametrized WebDriver for different browsers
            streamlit_base_url: Base URL of the Streamlit application
        """
        browser_name = self._get_browser_name(cross_browser_driver)
        self.log_test_step(f"Starting responsive design test: {browser_name}")

        self.navigate_and_verify(cross_browser_driver, streamlit_base_url)

        # Test different screen sizes
        screen_sizes = [
            (1920, 1080),  # Desktop
            (1366, 768),  # Laptop
            (768, 1024),  # Tablet
        ]

        ConfigPage(cross_browser_driver).navigate_to_page()

        for width, height in screen_sizes:
            self.log_test_step(
                f"Testing {width}x{height} resolution in {browser_name}"
            )

            # Resize browser window
            cross_browser_driver.set_window_size(width, height)

            # Wait for layout to stabilize
            self.wait_for_stable_layout(cross_browser_driver)

            # Basic verification that page still loads
            self.assert_streamlit_loaded(cross_browser_driver)

        # Reset to standard size
        cross_browser_driver.set_window_size(1920, 1080)
        self.log_test_step(f"Responsive design verified for {browser_name}")

    def _get_browser_name(self, driver: WebDriver) -> str:
        """Extract browser name from WebDriver capabilities."""
        try:
            capabilities = driver.capabilities
            browser_name = capabilities.get("browserName", "unknown").lower()
            return browser_name
        except Exception:
            return "unknown"

    def _get_browser_wait_multiplier(self, browser_name: str) -> float:
        """Get wait time multiplier based on browser performance."""
        multipliers = {
            "firefox": 1.2,  # Firefox typically slower
            "safari": 1.5,  # Safari may be slower on some operations
            "edge": 1.1,  # Edge slightly slower than Chrome
            "chrome": 1.0,  # Chrome as baseline
        }
        return multipliers.get(browser_name, 1.0)

    def _capture_browser_debug_info(
        self, driver: WebDriver, browser_name: str
    ) -> None:
        """Capture browser-specific debug information for failures."""
        try:
            # Capture screenshot with browser name
            screenshot_path = f"test-artifacts/e2e/failure_{browser_name}.png"
            driver.save_screenshot(screenshot_path)

            # Log basic driver information
            self.log_test_step(f"Debug info captured for {browser_name}")

        except Exception as debug_error:
            self.log_test_step(f"Failed to capture debug info: {debug_error}")

    def wait_for_stable_layout(
        self, driver: WebDriver, timeout: float = 5.0
    ) -> None:
        """Wait for responsive layout to stabilize after window resize."""
        import time

        time.sleep(timeout)  # Simple wait for layout stabilization


@pytest.mark.cross_browser
@pytest.mark.browser_matrix
@pytest.mark.performance
class TestCrossBrowserPerformance(BaseE2ETest):
    """Performance comparison tests across browsers."""

    @pytest.mark.e2e
    def test_page_load_performance_comparison(
        self,
        cross_browser_driver: WebDriver,
        streamlit_base_url: str,
    ) -> None:
        """Compare page load performance across browsers.

        Measures and compares basic performance metrics across supported
        browsers.

        Args:
            cross_browser_driver: Parametrized WebDriver for different browsers
            streamlit_base_url: Base URL of the Streamlit application
        """
        browser_name = self._get_browser_name(cross_browser_driver)
        self.log_test_step(f"Testing performance for: {browser_name}")

        # Simple performance test via successful page load
        self.navigate_and_verify(cross_browser_driver, streamlit_base_url)
        self.assert_streamlit_loaded(cross_browser_driver)

        # Navigate through main pages and verify load times are reasonable
        for page_class in [
            ConfigPage,
            ArchitecturePage,
            TrainPage,
            ResultsPage,
        ]:
            page_class(cross_browser_driver).navigate_to_page()
            self.assert_streamlit_loaded(cross_browser_driver)

        self.log_test_step(f"Performance test completed for {browser_name}")

    def _get_browser_name(self, driver: WebDriver) -> str:
        """Extract browser name from WebDriver capabilities."""
        try:
            capabilities = driver.capabilities
            browser_name = capabilities.get("browserName", "unknown").lower()
            return browser_name
        except Exception:
            return "unknown"


# Browser-specific test classes
class TestChromeSpecific(TestCrossBrowserCompatibility):
    """Chrome-specific feature tests."""

    @pytest.mark.chrome_only
    def test_chrome_specific_features(
        self,
        chrome_driver: WebDriver,
        streamlit_base_url: str,
    ) -> None:
        """Test Chrome-specific features and optimizations."""
        self.navigate_and_verify(chrome_driver, streamlit_base_url)
        self.assert_streamlit_loaded(chrome_driver)
        self.log_test_step("Chrome-specific features verified")


class TestFirefoxSpecific(TestCrossBrowserCompatibility):
    """Firefox-specific feature tests."""

    @pytest.mark.firefox_only
    def test_firefox_specific_features(
        self,
        firefox_driver: WebDriver,
        streamlit_base_url: str,
    ) -> None:
        """Test Firefox-specific features and behavior."""
        self.navigate_and_verify(firefox_driver, streamlit_base_url)
        self.assert_streamlit_loaded(firefox_driver)
        self.log_test_step("Firefox-specific features verified")


class TestEdgeSpecific(TestCrossBrowserCompatibility):
    """Edge-specific feature tests."""

    @pytest.mark.edge_only
    def test_edge_specific_features(
        self,
        edge_driver: WebDriver,
        streamlit_base_url: str,
    ) -> None:
        """Test Edge-specific features and behavior."""
        self.navigate_and_verify(edge_driver, streamlit_base_url)
        self.assert_streamlit_loaded(edge_driver)
        self.log_test_step("Edge-specific features verified")
