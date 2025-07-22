"""
E2E Happy Path Tests for CrackSeg Application. This module contains
tests that simulate a successful user journey through the main
functionalities of the CrackSeg application, from configuration to
viewing results. Performance monitoring is optionally integrated via
pytest markers.
"""

from typing import Any

import pytest
from selenium.webdriver.remote.webdriver import WebDriver

from ..base_test import BaseE2ETest
from ..pages import ArchitecturePage, ConfigPage, ResultsPage, TrainPage


@pytest.mark.e2e
class TestHappyPathWorkflow(BaseE2ETest):
    """Test suite for the main successful user workflow."""

    def setup_test_data(self) -> dict[str, Any]:
        """
        Set up test-specific data for the happy path workflow. Returns:
        Dictionary containing test data.
        """
        return {
            "config_file": "basic_verification.yaml",
            "expected_model_name": "MockModel",
        }

    @pytest.mark.e2e
    def test_full_workflow(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """
        Test the full user workflow from config to results. This test
        implements the complete 'happy path' workflow using robust navigation
        strategies and comprehensive error handling. Args: webdriver: The
        Selenium WebDriver instance. streamlit_base_url: The base URL of the
        Streamlit application.
        """
        self.log_test_step("Start of the happy path E2E test")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        try:
            # 1. Config Page: Load configuration
            self.log_test_step("Navigating to Config page")
            config_page_base = ConfigPage(webdriver).navigate_to_page()
            assert isinstance(config_page_base, ConfigPage)
            config_page = config_page_base
            self.assert_streamlit_loaded(webdriver)

            config_file = self.get_test_data()["config_file"]
            self.log_test_step(f"Selecting configuration: {config_file}")
            config_page.select_config_file(config_file)
            self.log_test_step("Clicking 'Load Configuration'")
            config_page.click_load_config()

            # Verify configuration is loaded
            config_page.wait_for_configuration_loaded()
            loaded_config_text = config_page.get_configuration_content()
            assert (
                loaded_config_text is not None
            ), "Config content should not be None"
            assert (
                "model" in loaded_config_text and "data" in loaded_config_text
            ), "Configuration content should be displayed"
            self.log_assertion(
                "Configuration loaded and displayed",
                "model" in loaded_config_text,
            )

            # 2. Architecture Page: Instantiate and verify model
            self.log_test_step("Navigating to Architecture page")
            arch_page_base = ArchitecturePage(webdriver).navigate_to_page()
            assert isinstance(arch_page_base, ArchitecturePage)
            arch_page = arch_page_base
            self.assert_streamlit_loaded(webdriver)

            self.log_test_step("Instantiating model from loaded config")
            arch_page.instantiate_model()

            # Verify model is instantiated
            arch_page.wait_for_model_instantiation()
            model_summary = arch_page.get_model_summary()
            expected_model_name = self.get_test_data()["expected_model_name"]
            assert (
                model_summary is not None
            ), "Model summary should not be None"
            assert (
                expected_model_name in model_summary
            ), f"Model summary should contain '{expected_model_name}'"
            self.log_assertion(
                "Model instantiated and summary displayed",
                expected_model_name in model_summary,
            )

            # 3. Train Page: Start training and verify logs
            self.log_test_step("Navigating to Train page")
            train_page_base = TrainPage(webdriver).navigate_to_page()
            assert isinstance(train_page_base, TrainPage)
            train_page = train_page_base
            self.assert_streamlit_loaded(webdriver)

            self.log_test_step("Starting training process")
            train_page.start_training()

            # Verify training has started by checking for live logs
            train_page.wait_for_training_to_start()
            status = train_page.get_training_status()
            assert status is not None, "Training status should be available"
            assert (
                "running" in status.lower()
            ), "Training status should be 'running'"
            self.log_assertion(
                "Training started and status is running",
                "running" in status.lower(),
            )

            # 4. Navigate to Results page and verify results are displayed
            # (This assumes training completes and produces results)
            self.log_test_step("Navigating to Results page")
            results_page_base = ResultsPage(webdriver).navigate_to_page()
            assert isinstance(results_page_base, ResultsPage)
            results_page = results_page_base
            self.assert_streamlit_loaded(webdriver)

            # Verify results are available
            results_page.wait_for_results_loaded()
            assert (
                results_page.is_results_gallery_displayed()
            ), "Results gallery should be available after training"
            self.log_assertion("Results are displayed on the page", True)

            self.log_test_step("End of the happy path E2E test")

        except Exception as e:
            # Enhanced error handling with debugging information
            self.log_test_step(f"❌ Test failed with error: {str(e)}")

            # Capture debugging information using Selenium
            try:
                from selenium.webdriver.common.by import By

                # Check if sidebar is present
                sidebar_elements = webdriver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stSidebar']"
                )

                if sidebar_elements:
                    # Get all buttons in sidebar
                    sidebar_buttons = webdriver.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stSidebar'] button"
                    )

                    button_texts = []
                    for btn in sidebar_buttons[:5]:  # Limit to first 5 buttons
                        try:
                            text = btn.text.strip()
                            if text:
                                button_texts.append(text)
                        except Exception:
                            button_texts.append("[unreadable]")

                    sidebar_info = (
                        f"Found {len(sidebar_buttons)} buttons: "
                        f"{', '.join(button_texts)}"
                    )
                else:
                    sidebar_info = "Sidebar not found in DOM"

                self.log_test_step(f"🔍 Debug info - {sidebar_info}")

            except Exception as debug_error:
                self.log_test_step(
                    f"⚠️ Could not extract debug info: {debug_error}"
                )

            # Re-raise the original exception
            raise

    @pytest.mark.e2e
    @pytest.mark.performance
    def test_full_workflow_with_performance_monitoring(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """
        Test the full user workflow with comprehensive performance monitoring.
        This test demonstrates the same happy path workflow with optional
        performance monitoring enabled via the @pytest.mark.performance
        decorator. All functionality remains identical, but performance
        metrics are captured. Args: webdriver: The Selenium WebDriver
        instance. streamlit_base_url: The base URL of the Streamlit
        application.
        """
        self.log_test_step("Start of performance-monitored happy path test")

        # Performance monitoring is automatically enabled via the marker
        assert (
            self._performance_enabled
        ), "Performance monitoring should be enabled"

        # Start performance monitoring
        self.start_performance_monitoring()

        # Navigate with performance measurement
        nav_metrics = self.navigate_and_verify(
            webdriver, streamlit_base_url, measure_performance=True
        )

        if nav_metrics:
            self.log_test_step(
                f"Initial navigation completed in "
                f"{nav_metrics['load_complete']:.2f}s"
            )

        try:
            # Take memory snapshot at start
            initial_memory = self.monitor_memory_usage_snapshot()

            # 1. Config Page: Load configuration with performance tracking
            self.log_test_step("Navigating to Config page")

            # Measure navigation performance
            self.track_interaction_performance(
                webdriver,
                "page_navigation",
                "Config page navigation",
                threshold_ms=2000.0,
            )

            config_page_base = ConfigPage(webdriver).navigate_to_page()
            assert isinstance(config_page_base, ConfigPage)
            config_page = config_page_base
            self.assert_streamlit_loaded(webdriver)

            config_file = self.get_test_data()["config_file"]
            self.log_test_step(f"Selecting configuration: {config_file}")

            # Measure file selection performance
            self.track_interaction_performance(
                webdriver,
                "file_selection",
                f"Config file selection: {config_file}",
                threshold_ms=1000.0,
            )

            config_page.select_config_file(config_file)

            self.log_test_step("Clicking 'Load Configuration'")

            # Measure button click performance
            self.track_interaction_performance(
                webdriver,
                "button_click",
                "Load Configuration button",
                threshold_ms=500.0,
            )

            config_page.click_load_config()

            # Verify configuration is loaded
            config_page.wait_for_configuration_loaded()
            loaded_config_text = config_page.get_configuration_content()
            assert (
                loaded_config_text is not None
            ), "Config content should not be None"
            assert (
                "model" in loaded_config_text and "data" in loaded_config_text
            ), "Configuration content should be displayed"
            self.log_assertion(
                "Configuration loaded and displayed",
                "model" in loaded_config_text,
            )

            # Memory check after config load
            self.monitor_memory_usage_snapshot()

            # 2. Architecture Page: Instantiate and verify model
            self.log_test_step("Navigating to Architecture page")
            arch_page_base = ArchitecturePage(webdriver).navigate_to_page()
            assert isinstance(arch_page_base, ArchitecturePage)
            arch_page = arch_page_base
            self.assert_streamlit_loaded(webdriver)

            self.log_test_step("Instantiating model from loaded config")

            # Measure model instantiation performance
            self.track_interaction_performance(
                webdriver,
                "model_instantiation",
                "Model instantiation process",
                threshold_ms=5000.0,
            )

            arch_page.instantiate_model()

            # Verify model is instantiated
            arch_page.wait_for_model_instantiation()
            model_summary = arch_page.get_model_summary()
            expected_model_name = self.get_test_data()["expected_model_name"]
            assert (
                model_summary is not None
            ), "Model summary should not be None"
            assert (
                expected_model_name in model_summary
            ), f"Model summary should contain '{expected_model_name}'"
            self.log_assertion(
                "Model instantiated and summary displayed",
                expected_model_name in model_summary,
            )

            # Memory check after model instantiation
            self.monitor_memory_usage_snapshot()

            # 3. Train Page: Start training and verify logs
            self.log_test_step("Navigating to Train page")
            train_page_base = TrainPage(webdriver).navigate_to_page()
            assert isinstance(train_page_base, TrainPage)
            train_page = train_page_base
            self.assert_streamlit_loaded(webdriver)

            self.log_test_step("Starting training process")

            # Measure training start performance
            self.track_interaction_performance(
                webdriver,
                "training_start",
                "Training process initiation",
                threshold_ms=3000.0,
            )

            train_page.start_training()

            # Verify training has started by checking for live logs
            train_page.wait_for_training_to_start()
            status = train_page.get_training_status()
            assert status is not None, "Training status should be available"
            assert (
                "running" in status.lower()
            ), "Training status should be 'running'"
            self.log_assertion(
                "Training started and status is running",
                "running" in status.lower(),
            )

            # 4. Navigate to Results page and verify results are displayed
            self.log_test_step("Navigating to Results page")
            results_page_base = ResultsPage(webdriver).navigate_to_page()
            assert isinstance(results_page_base, ResultsPage)
            results_page = results_page_base
            self.assert_streamlit_loaded(webdriver)

            # Verify results are available
            results_page.wait_for_results_loaded()
            assert (
                results_page.is_results_gallery_displayed()
            ), "Results gallery should be available after training"
            self.log_assertion("Results are displayed on the page", True)

            # Final memory check
            final_memory = self.monitor_memory_usage_snapshot()

            # Log memory usage progression
            if initial_memory and final_memory:
                total_memory_increase = (
                    final_memory["rss_memory_mb"]
                    - initial_memory["rss_memory_mb"]
                )
                self.log_test_step(
                    f"Total memory increase during workflow: "
                    f"{total_memory_increase:.1f}MB"
                )

            self.log_test_step("End of performance-monitored happy path test")

            # Generate comprehensive performance report
            performance_report = self.generate_performance_report()

            if performance_report:
                # Log performance summary
                avg_load_time = performance_report.get(
                    "avg_page_load_time", "N/A"
                )
                peak_memory = performance_report.get(
                    "peak_memory_usage", "N/A"
                )
                total_duration = performance_report.get(
                    "total_duration", "N/A"
                )

                self.log_test_step(
                    f"Performance Summary - "
                    f"Total Duration: {total_duration}s, "
                    f"Avg Page Load: {avg_load_time}s, "
                    f"Peak Memory: {peak_memory}MB"
                )

        except Exception as e:
            # Enhanced error handling with performance context
            self.log_test_step(
                f"❌ Performance test failed with error: {str(e)}"
            )

            # Generate emergency performance report for debugging
            if self._performance_enabled:
                try:
                    emergency_report = self.generate_performance_report()
                    if emergency_report:
                        duration = emergency_report.get(
                            "total_duration", "N/A"
                        )
                        memory = emergency_report.get(
                            "peak_memory_usage", "N/A"
                        )
                        self.log_test_step(
                            f"🔍 Performance context at failure: "
                            f"Duration: {duration}s, Memory: {memory}MB"
                        )
                except Exception:
                    pass  # Don't fail the test further if reporting fails

            # Re-raise the original exception
            raise
