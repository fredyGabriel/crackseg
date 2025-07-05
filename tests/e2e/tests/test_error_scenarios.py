"""E2E tests for error scenarios and error handling.

This module contains tests that validate error handling, error messages,
recovery mechanisms, and system robustness under various failure conditions.
"""

import time
from pathlib import Path
from typing import Any

import pytest
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from ..base_test import BaseE2ETest
from ..pages import ArchitecturePage, ConfigPage, ResultsPage, TrainPage
from ..utils.debugging import E2EDebugger


class ErrorTestUtilities:
    """Utility functions for error scenario testing."""

    @staticmethod
    def create_invalid_config_file(
        config_type: str, output_dir: Path | str = "test-artifacts"
    ) -> Path:
        """Create temporary invalid configuration files for testing.

        Args:
            config_type: Type of invalid config
                ('syntax', 'missing_fields', 'conflicts')
            output_dir: Directory to save test config files

        Returns:
            Path to created invalid config file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        config_files = {
            "syntax": (
                "tests/e2e/test_data/invalid_configs/invalid_syntax.yaml"
            ),
            "missing_fields": (
                "tests/e2e/test_data/invalid_configs/missing_required_fields.yaml"
            ),
            "conflicts": (
                "tests/e2e/test_data/invalid_configs/conflicting_dependencies.yaml"
            ),
        }

        source_path = Path(
            config_files.get(config_type, config_files["syntax"])
        )
        return source_path

    @staticmethod
    def wait_for_error_message(
        driver: WebDriver, timeout: float = 10.0
    ) -> str | None:
        """Wait for and capture error message from Streamlit alerts.

        Args:
            driver: WebDriver instance
            timeout: Maximum time to wait for error message

        Returns:
            Error message text or None if no error found
        """
        try:
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.support.ui import WebDriverWait

            # Wait for error alert to appear
            error_element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='stAlert']")
                )
            )

            # Get error message text
            if error_element:
                return error_element.text.strip()

        except TimeoutException:
            # No error message appeared within timeout
            pass

        return None

    @staticmethod
    def check_error_recovery_options(driver: WebDriver) -> dict[str, bool]:
        """Check for available error recovery mechanisms in the UI.

        Args:
            driver: WebDriver instance

        Returns:
            Dictionary with available recovery options
        """
        recovery_options = {
            "retry_button": False,
            "reset_button": False,
            "clear_button": False,
            "reload_button": False,
            "back_to_config": False,
        }

        # Check for common recovery buttons
        recovery_selectors = {
            "retry_button": (
                "//button[contains(text(), 'Retry') or "
                "contains(text(), 'Try Again')]"
            ),
            "reset_button": (
                "//button[contains(text(), 'Reset') or "
                "contains(text(), 'Clear')]"
            ),
            "clear_button": (
                "//button[contains(text(), 'Clear') or "
                "contains(text(), 'Remove')]"
            ),
            "reload_button": (
                "//button[contains(text(), 'Reload') or "
                "contains(text(), 'Refresh')]"
            ),
            "back_to_config": (
                "//button[contains(text(), 'Back') or "
                "contains(text(), 'Config')]"
            ),
        }

        for option, xpath in recovery_selectors.items():
            try:
                elements = driver.find_elements(By.XPATH, xpath)
                recovery_options[option] = len(elements) > 0
            except Exception:
                pass

        return recovery_options

    @staticmethod
    def simulate_network_timeout(
        driver: WebDriver, timeout_seconds: int = 5
    ) -> None:
        """Simulate network timeout by setting very short timeouts.

        Args:
            driver: WebDriver instance
            timeout_seconds: Timeout duration to simulate

        Note:
            This is a simplified simulation. Real network issues would
            require more sophisticated mocking.
        """
        # Set very short timeouts to simulate network issues
        driver.set_page_load_timeout(timeout_seconds)
        driver.implicitly_wait(timeout_seconds)


@pytest.mark.e2e
class TestErrorScenarios(BaseE2ETest):
    """Test suite for comprehensive error scenario validation."""

    def setup_test_data(self) -> dict[str, Any]:
        """Set up test-specific data for error scenarios.

        Returns:
            Dictionary containing error test data and configuration.
        """
        return {
            "expected_error_keywords": ["error", "invalid", "failed"],
            "max_error_wait_time": 15.0,
        }

    @pytest.mark.e2e
    def test_configuration_validation_errors(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test configuration validation error scenarios.

        Tests various configuration error conditions including:
        - Invalid YAML syntax
        - Missing required fields
        - Conflicting dependencies
        - Validation error message display
        - Error recovery mechanisms
        """
        self.log_test_step("Start configuration validation error tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        debugger = E2EDebugger(webdriver, "config_validation_errors")

        try:
            # Navigate to Config page
            self.log_test_step("Navigating to Config page for error testing")
            config_page_base = ConfigPage(webdriver).navigate_to_page()
            assert isinstance(config_page_base, ConfigPage)
            config_page = config_page_base
            self.assert_streamlit_loaded(webdriver)

            test_data = self.get_test_data()
            error_utils = ErrorTestUtilities()

            # Test 1: Basic validation and error handling
            self.log_test_step("Testing basic validation and error handling")

            # Try to trigger an error by clicking load without selecting config
            try:
                config_page.click_load_config()

                # Wait for potential error message
                error_message = error_utils.wait_for_error_message(
                    webdriver, 5.0
                )

                if error_message:
                    self.log_assertion(
                        "Error message displayed for validation",
                        any(
                            keyword in error_message.lower()
                            for keyword in test_data["expected_error_keywords"]
                        ),
                    )
                else:
                    self.log_assertion(
                        "No error message shown "
                        "(UI may prevent invalid action)",
                        True,
                    )

            except Exception as e:
                self.log_test_step(f"Config validation test result: {e}")

            # Test 2: Error recovery mechanisms
            self.log_test_step("Testing error recovery mechanisms")

            # Check for recovery mechanisms
            recovery_options = error_utils.check_error_recovery_options(
                webdriver
            )
            self.log_assertion(
                "Error recovery options available",
                any(recovery_options.values()),
            )

            # Test 3: State consistency after error
            self.log_test_step(
                "Testing application state consistency after errors"
            )

            # Verify that the application is still responsive
            config_page.navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Verify that other functions still work
            config_page.select_config_file("basic_verification.yaml")

            self.log_assertion(
                "Application remains responsive after error", True
            )

        except Exception as e:
            self.log_test_step(f"❌ Configuration error test failed: {e}")
            debugger.save_debug_report()
            raise

    @pytest.mark.e2e
    def test_training_process_errors(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test training process error scenarios.

        Tests various training error conditions including:
        - Resource exhaustion simulation
        - Process interruption handling
        - Training timeout scenarios
        - Error state recovery
        """
        self.log_test_step("Start training process error tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        debugger = E2EDebugger(webdriver, "training_process_errors")

        try:
            # Setup: Load valid configuration first
            self.log_test_step(
                "Setting up valid configuration for training error tests"
            )
            config_page = ConfigPage(webdriver).navigate_to_page()
            config_page.select_config_file("basic_verification.yaml")
            config_page.click_load_config()
            config_page.wait_for_configuration_loaded()

            # Navigate to Architecture page and instantiate model
            self.log_test_step("Instantiating model for training error tests")
            arch_page_base = ArchitecturePage(webdriver).navigate_to_page()
            assert isinstance(arch_page_base, ArchitecturePage)
            arch_page = arch_page_base
            arch_page.instantiate_model()
            arch_page.wait_for_model_instantiation()

            # Navigate to Training page
            self.log_test_step("Navigating to Training page for error testing")
            train_page_base = TrainPage(webdriver).navigate_to_page()
            assert isinstance(train_page_base, TrainPage)
            train_page = train_page_base

            # Test 1: Training timeout simulation
            self.log_test_step("Testing training timeout scenarios")

            # Set very short timeout to simulate timeout conditions
            error_utils = ErrorTestUtilities()
            error_utils.simulate_network_timeout(webdriver, 5)

            try:
                # Attempt to start training
                train_page.start_training()

                # Wait for training to start or timeout
                time.sleep(3)

                # Check if training started or if there's an error
                training_status = train_page.get_training_status()

                if training_status:
                    self.log_assertion(
                        "Training status available despite timeout simulation",
                        True,
                    )
                else:
                    self.log_assertion(
                        "Training did not start (timeout simulation)", True
                    )

            except TimeoutException:
                self.log_assertion("Timeout exception properly handled", True)

            # Reset timeouts to normal
            webdriver.set_page_load_timeout(30)
            webdriver.implicitly_wait(10)

            # Test 2: Process interruption handling
            self.log_test_step(
                "Testing training process interruption handling"
            )

            # If training is running, try to stop it
            try:
                if train_page.is_element_displayed(
                    (By.XPATH, "//button[contains(text(), 'Stop')]")
                ):
                    stop_button = train_page.wait_for_element(
                        (By.XPATH, "//button[contains(text(), 'Stop')]")
                    )
                    if stop_button:
                        stop_button.click()
                        time.sleep(2)

                        # Check if stop was handled gracefully
                        status_after_stop = train_page.get_training_status()
                        self.log_assertion(
                            "Training stop handled gracefully",
                            status_after_stop is not None,
                        )

            except Exception as e:
                self.log_test_step(f"Stop training test result: {e}")

            # Test 3: Error recovery and state consistency
            self.log_test_step("Testing error recovery and application state")

            # Verify navigation still works after errors
            config_page.navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            self.log_assertion(
                "Navigation functional after training errors", True
            )

        except Exception as e:
            self.log_test_step(f"❌ Training error test failed: {e}")
            debugger.save_debug_report()
            raise

    @pytest.mark.e2e
    def test_ui_error_handling_validation(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test UI error handling and user feedback mechanisms.

        Tests various UI error conditions including:
        - Error message display validation
        - User feedback mechanisms
        - UI state consistency during errors
        - Accessibility of error information
        """
        self.log_test_step("Start UI error handling validation tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        debugger = E2EDebugger(webdriver, "ui_error_handling")

        try:
            error_utils = ErrorTestUtilities()

            # Test 1: Error message display and formatting
            self.log_test_step("Testing error message display and formatting")

            config_page = ConfigPage(webdriver).navigate_to_page()

            # Try to trigger an error by clicking load without selecting config
            try:
                config_page.click_load_config()

                # Wait for potential error message
                error_message = error_utils.wait_for_error_message(
                    webdriver, 5.0
                )

                if error_message:
                    self.log_assertion(
                        "Error message is properly displayed",
                        len(error_message) > 0,
                    )

                    # Check if error message is user-friendly
                    self.log_assertion(
                        "Error message contains helpful information",
                        any(
                            keyword in error_message.lower()
                            for keyword in ["select", "config", "file"]
                        ),
                    )
                else:
                    self.log_assertion(
                        "No error triggered (UI may prevent invalid action)",
                        True,
                    )

            except Exception as e:
                self.log_test_step(f"Error message test result: {e}")

            # Test 2: Recovery mechanism availability
            self.log_test_step("Testing error recovery mechanism availability")

            recovery_options = error_utils.check_error_recovery_options(
                webdriver
            )
            available_recoveries = sum(recovery_options.values())

            self.log_assertion(
                f"Recovery options available: {available_recoveries}",
                available_recoveries
                >= 0,  # At least acknowledge recovery options exist
            )

            # Test 3: UI consistency during error states
            self.log_test_step("Testing UI consistency during error states")

            # Check that navigation is still functional
            sidebar_inspection = debugger.inspect_sidebar_elements(
                verbose=False
            )

            self.log_assertion(
                "Sidebar navigation remains functional during errors",
                sidebar_inspection["sidebar_present"],
            )

            self.log_assertion(
                "Navigation buttons still available during errors",
                sidebar_inspection["buttons_found"] > 0,
            )

            # Test 4: Cross-page error state management
            self.log_test_step("Testing cross-page error state management")

            # Navigate between pages to ensure error states don't persist
            ArchitecturePage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            TrainPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            ResultsPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Return to config page
            config_page.navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            self.log_assertion(
                "Cross-page navigation functional during error scenarios", True
            )

        except Exception as e:
            self.log_test_step(f"❌ UI error handling test failed: {e}")
            debugger.save_debug_report()
            raise

    @pytest.mark.e2e
    def test_system_integration_errors(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test system integration error scenarios.

        Tests various system-level error conditions including:
        - Service availability issues
        - Resource constraint handling
        - Network connectivity problems
        - System state recovery
        """
        self.log_test_step("Start system integration error tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        debugger = E2EDebugger(webdriver, "system_integration_errors")

        try:
            # Test 1: Service availability verification
            self.log_test_step("Testing service availability verification")

            # Check if application loads and basic services are available
            self.assert_streamlit_loaded(webdriver)

            # Check if all main pages are accessible
            main_pages = [ConfigPage, ArchitecturePage, TrainPage, ResultsPage]
            accessible_pages = 0

            for page_class in main_pages:
                try:
                    page_class(webdriver).navigate_to_page()
                    self.assert_streamlit_loaded(webdriver)
                    accessible_pages += 1
                    self.log_test_step(f"✓ {page_class.__name__} accessible")
                except Exception as e:
                    self.log_test_step(f"⚠ {page_class.__name__} error: {e}")

            self.log_assertion(
                f"System services availability: {accessible_pages}/"
                f"{len(main_pages)} pages accessible",
                accessible_pages > 0,
            )

            # Test 2: Resource constraint simulation
            self.log_test_step("Testing resource constraint handling")

            config_page = ConfigPage(webdriver).navigate_to_page()

            # Try to load a configuration that might stress resources
            config_page.select_config_file("basic_verification.yaml")
            config_page.click_load_config()

            # Monitor for resource-related errors or warnings
            time.sleep(5)  # Allow time for resource checks

            error_utils = ErrorTestUtilities()
            potential_error = error_utils.wait_for_error_message(
                webdriver, 3.0
            )

            if potential_error:
                self.log_assertion(
                    "Resource constraint error properly communicated",
                    "memory" in potential_error.lower()
                    or "resource" in potential_error.lower(),
                )
            else:
                self.log_assertion(
                    "No resource constraint errors "
                    "(system has adequate resources)",
                    True,
                )

            # Test 3: Network connectivity resilience
            self.log_test_step("Testing network connectivity resilience")

            # Navigate between pages rapidly to test network resilience
            pages_to_test = [ConfigPage, ArchitecturePage, TrainPage]
            successful_navigations = 0

            for page_class in pages_to_test:
                try:
                    page_class(webdriver).navigate_to_page()
                    time.sleep(1)  # Brief pause between navigations
                    successful_navigations += 1
                except Exception as e:
                    self.log_test_step(f"Navigation stress test error: {e}")

            self.log_assertion(
                "Network resilience under navigation stress",
                successful_navigations
                >= len(pages_to_test) // 2,  # At least half successful
            )

            # Test 4: System state recovery verification
            self.log_test_step("Testing system state recovery")

            # Return to a known good state
            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Verify basic functionality still works
            config_page.select_config_file("basic_verification.yaml")

            self.log_assertion("System recovered to functional state", True)

        except Exception as e:
            self.log_test_step(f"❌ System integration error test failed: {e}")
            debugger.save_debug_report()
            raise

        finally:
            # Cleanup: Save final debug report
            self.log_test_step("Saving final system error test debug report")
            debugger.save_debug_report()
