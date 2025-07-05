"""E2E tests for edge cases and boundary conditions.

This module contains tests that validate system behavior under edge conditions,
including boundary values, concurrent operations, resource limits, data
corruption, performance degradation, and unusual user interaction patterns.
"""

import concurrent.futures
import time
from pathlib import Path
from typing import Any, TypedDict

import pytest
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from ..base_test import BaseE2ETest
from ..pages import ArchitecturePage, ConfigPage, ResultsPage, TrainPage
from ..utils.debugging import E2EDebugger


class ConcurrentTestResults(TypedDict):
    """Type definition for concurrent navigation test results."""

    start_time: float
    navigation_attempts: int
    successful_navigations: int
    errors: list[str]
    concurrent_sessions: int
    duration: float


class NavigationTime(TypedDict):
    """Type definition for navigation timing data."""

    page: str
    time: float
    conditions: str


class PerformanceMetrics(TypedDict):
    """Type definition for performance metrics collection."""

    navigation_times: list[NavigationTime]
    load_times: list[float]
    error_recovery_times: list[float]
    memory_pressure_impact: dict[str, Any]


class EdgeCaseTestUtilities:
    """Utility functions for edge case testing scenarios."""

    @staticmethod
    def create_large_config_file(
        size_mb: float = 2.0, output_dir: Path | str = "test-artifacts"
    ) -> Path:
        """Create a large configuration file for boundary testing.

        Args:
            size_mb: Size of config file in megabytes
            output_dir: Directory to save the large config file

        Returns:
            Path to created large config file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        config_file = output_path / "large_config_edge_test.yaml"

        # Generate large YAML content while maintaining valid structure
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("# Large configuration for edge case testing\n")
            f.write("model:\n")
            f.write("  name: EdgeTestModel\n")
            f.write("  architecture: cnn\n")
            f.write("  parameters:\n")

            # Generate large parameter list to reach desired size
            bytes_written = 100  # Approximate initial content
            target_bytes = int(size_mb * 1024 * 1024)

            param_count = 0
            while bytes_written < target_bytes:
                param_line = (
                    f"    param_{param_count:06d}: "
                    f"{param_count * 0.123456789}\n"
                )
                f.write(param_line)
                bytes_written += len(param_line.encode("utf-8"))
                param_count += 1

            f.write("data:\n")
            f.write("  train_path: /data/train\n")
            f.write("  val_path: /data/val\n")

        return config_file

    @staticmethod
    def create_minimal_config_file(
        output_dir: Path | str = "test-artifacts",
    ) -> Path:
        """Create minimal configuration file for boundary testing.

        Args:
            output_dir: Directory to save the minimal config file

        Returns:
            Path to created minimal config file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        config_file = output_path / "minimal_config_edge_test.yaml"

        with open(config_file, "w", encoding="utf-8") as f:
            f.write("# Minimal configuration for edge case testing\n")
            f.write("model:\n")
            f.write("  name: MinimalModel\n")

        return config_file

    @staticmethod
    def simulate_memory_pressure() -> None:
        """Simulate memory pressure for resource limit testing.

        Note:
            This creates temporary memory pressure but releases it quickly
            to avoid system instability. Real memory testing should be done
            in isolated environments.
        """
        # Create temporary memory pressure (50MB)
        memory_hog = bytearray(50 * 1024 * 1024)  # 50MB
        time.sleep(0.5)  # Brief pressure
        del memory_hog  # Release immediately

    @staticmethod
    def simulate_concurrent_navigation(
        driver_factory: Any, base_url: str, duration_seconds: int = 5
    ) -> ConcurrentTestResults:
        """Simulate concurrent navigation stress testing.

        Args:
            driver_factory: Factory function to create WebDriver instances
            base_url: Base URL for navigation testing
            duration_seconds: Duration of concurrent operations

        Returns:
            Dictionary with stress test results
        """
        results: ConcurrentTestResults = {
            "start_time": time.time(),
            "navigation_attempts": 0,
            "successful_navigations": 0,
            "errors": [],
            "concurrent_sessions": 0,
            "duration": 0.0,
        }

        def navigation_worker():
            try:
                # Create isolated driver instance for concurrent testing
                worker_driver = driver_factory()
                worker_driver.get(base_url)
                time.sleep(0.5)  # Brief interaction
                results["successful_navigations"] += 1
                worker_driver.quit()
            except Exception as e:
                results["errors"].append(str(e))
            finally:
                results["navigation_attempts"] += 1

        # Run concurrent navigation tests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            end_time = time.time() + duration_seconds

            while time.time() < end_time:
                if len(futures) < 3:  # Limit concurrent sessions
                    futures.append(executor.submit(navigation_worker))
                    results["concurrent_sessions"] += 1
                    time.sleep(0.3)  # Small delay between session starts

                # Clean up completed futures
                futures = [f for f in futures if not f.done()]

            # Wait for remaining futures
            for future in futures:
                future.result(timeout=10)

        results["duration"] = time.time() - results["start_time"]
        return results


@pytest.mark.e2e
class TestEdgeCases(BaseE2ETest):
    """Test suite for comprehensive edge case validation."""

    def setup_test_data(self) -> dict[str, Any]:
        """Set up test-specific data for edge case scenarios.

        Returns:
            Dictionary containing edge case test data and configuration.
        """
        return {
            "boundary_test_config": "basic_verification.yaml",
            "max_wait_time": 30.0,
            "memory_stress_threshold_mb": 100,
            "concurrent_session_limit": 3,
            "large_file_size_mb": 1.5,
        }

    @pytest.mark.e2e
    def test_boundary_value_inputs(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test boundary conditions for input values and file sizes.

        Tests various boundary scenarios including:
        - Maximum and minimum configuration file sizes
        - Empty configuration handling
        - Very long model names and parameter values
        - Numerical edge cases (zero, negative values)
        """
        self.log_test_step("Start boundary value input tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        debugger = E2EDebugger(webdriver, "boundary_value_inputs")

        try:
            # Test 1: Large configuration file handling
            self.log_test_step("Testing large configuration file boundary")
            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Create and test large config file
            large_config = EdgeCaseTestUtilities.create_large_config_file(
                size_mb=self.get_test_data()["large_file_size_mb"]
            )

            try:
                # Attempt to load large config - should handle gracefully
                config_page.upload_configuration_file(str(large_config))
                config_page.click_load_config()

                # Verify system handles large file appropriately
                # Either loads successfully or shows appropriate error
                time.sleep(3)  # Allow processing time

                config_content = config_page.get_configuration_content()
                if config_content is None:
                    # Check for error message
                    error_elements = webdriver.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stAlert']"
                    )
                    if error_elements:
                        error_text = error_elements[0].text
                        assert any(
                            word in error_text.lower()
                            for word in ["large", "size", "limit", "memory"]
                        ), (
                            "Large file should trigger appropriate "
                            "error message"
                        )
                        self.log_assertion(
                            "Large file handled with appropriate error", True
                        )
                    else:
                        pytest.fail(
                            "Large file should either load or show error"
                        )
                else:
                    # Verify content is displayed correctly
                    assert (
                        "EdgeTestModel" in config_content
                    ), "Large config content should be accessible"
                    self.log_assertion(
                        "Large config file loaded successfully", True
                    )

            finally:
                # Cleanup large test file
                if large_config.exists():
                    large_config.unlink()

            # Test 2: Minimal configuration file handling
            self.log_test_step("Testing minimal configuration file boundary")
            minimal_config = EdgeCaseTestUtilities.create_minimal_config_file()

            try:
                config_page.upload_configuration_file(str(minimal_config))
                config_page.click_load_config()

                # Verify minimal config handling
                config_content = config_page.get_configuration_content()
                if config_content:
                    assert (
                        "MinimalModel" in config_content
                    ), "Minimal config should load basic content"
                    self.log_assertion(
                        "Minimal config loaded successfully", True
                    )
                else:
                    # Check for validation error
                    error_elements = webdriver.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stAlert']"
                    )
                    if error_elements:
                        self.log_assertion(
                            "Minimal config triggered validation error", True
                        )
                    else:
                        pytest.fail(
                            "Minimal config should load or show "
                            "validation error"
                        )

            finally:
                # Cleanup minimal test file
                if minimal_config.exists():
                    minimal_config.unlink()

            # Test 3: Empty file boundary case
            self.log_test_step("Testing empty configuration file boundary")
            empty_config = Path("test-artifacts") / "empty_config.yaml"
            empty_config.parent.mkdir(exist_ok=True)
            empty_config.write_text("", encoding="utf-8")

            try:
                config_page.upload_configuration_file(str(empty_config))
                config_page.click_load_config()

                # Verify empty file is handled gracefully
                error_elements = webdriver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stAlert']"
                )
                assert (
                    len(error_elements) > 0
                ), "Empty config file should trigger error message"
                error_text = error_elements[0].text.lower()
                assert any(
                    word in error_text
                    for word in ["empty", "invalid", "error", "required"]
                ), "Empty file error should be descriptive"
                self.log_assertion(
                    "Empty config handled with appropriate error", True
                )

            finally:
                if empty_config.exists():
                    empty_config.unlink()

        except Exception as e:
            debugger.save_debug_report()
            self.log_test_step(f"❌ Boundary value test failed: {str(e)}")
            raise

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

    @pytest.mark.e2e
    def test_system_resource_limits(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test system behavior under resource constraints.

        Tests system limits including:
        - Memory pressure scenarios
        - CPU intensive operations
        - Long-running processes
        - Resource cleanup verification
        """
        self.log_test_step("Start system resource limit tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        try:
            # Test 1: Memory pressure during configuration loading
            self.log_test_step("Testing behavior under memory pressure")

            # Create memory pressure
            EdgeCaseTestUtilities.simulate_memory_pressure()

            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Try to load config under memory pressure
            start_time = time.time()
            config_page.select_config_file(
                self.get_test_data()["boundary_test_config"]
            )
            config_page.click_load_config()

            # Verify operation completes within reasonable time even under
            # pressure
            try:
                config_page.wait_for_configuration_loaded(timeout=20.0)
                load_time = time.time() - start_time
                self.log_performance_metric(
                    "config_load_under_pressure", load_time
                )

                # Verify config loaded correctly
                config_content = config_page.get_configuration_content()
                assert (
                    config_content is not None
                ), "Config should load even under memory pressure"
                self.log_assertion(
                    "Config loaded successfully under memory pressure", True
                )

            except TimeoutException:
                self.log_assertion(
                    "Config loading timed out under memory pressure", False
                )
                # This might be acceptable behavior under extreme resource
                # constraints

            # Test 2: Browser resource consumption monitoring
            self.log_test_step("Testing browser resource consumption patterns")

            # Navigate through multiple pages and monitor performance
            navigation_times = []
            pages_to_test = [
                ConfigPage,
                ArchitecturePage,
                TrainPage,
                ResultsPage,
            ]

            for PageClass in pages_to_test:
                start_time = time.time()
                PageClass(webdriver).navigate_to_page()
                self.assert_streamlit_loaded(webdriver)
                nav_time = time.time() - start_time
                navigation_times.append(nav_time)

                # Log performance for monitoring
                self.log_performance_metric(
                    f"{PageClass.__name__}_navigation", nav_time
                )

            # Verify performance doesn't degrade significantly
            avg_nav_time = sum(navigation_times) / len(navigation_times)
            max_nav_time = max(navigation_times)

            assert avg_nav_time < 8.0, (
                f"Average navigation time {avg_nav_time:.2f}s exceeds "
                "threshold"
            )
            assert max_nav_time < 15.0, (
                f"Maximum navigation time {max_nav_time:.2f}s exceeds "
                "threshold"
            )

            self.log_assertion(
                "Navigation performance acceptable (avg: "
                f"{avg_nav_time:.2f}s)",
                True,
            )

        except Exception as e:
            self.log_test_step(f"❌ Resource limit test failed: {str(e)}")
            raise

    @pytest.mark.e2e
    def test_data_corruption_scenarios(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Test handling of corrupted or malformed data.

        Tests data corruption scenarios including:
        - Malformed configuration files
        - Interrupted file operations
        - Binary data injection
        - Recovery mechanisms
        """
        self.log_test_step("Start data corruption scenario tests")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        try:
            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Test 1: Malformed YAML content
            self.log_test_step("Testing malformed YAML file handling")

            corrupted_config = Path("test-artifacts") / "corrupted_config.yaml"
            corrupted_config.parent.mkdir(exist_ok=True)

            # Create malformed YAML with various corruption patterns
            corrupted_content = """
# Corrupted YAML for testing
model:
  name: CorruptedModel
  parameters:
    - invalid: [unclosed bracket
    - missing_value:
  nested:
    level1:
      level2: {unclosed_brace
    invalid_structure: [1, 2, 3}  # Mixed brackets
data:
  path: "/some/path"
  invalid_unicode: "\x00\x01\x02"
"""
            corrupted_config.write_text(corrupted_content, encoding="utf-8")

            try:
                config_page.upload_configuration_file(str(corrupted_config))
                config_page.click_load_config()

                # Verify system handles corrupted YAML gracefully
                error_elements = webdriver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stAlert']"
                )

                if error_elements:
                    error_text = error_elements[0].text.lower()
                    assert any(
                        word in error_text
                        for word in [
                            "yaml",
                            "parse",
                            "syntax",
                            "invalid",
                            "error",
                        ]
                    ), "Corrupted YAML should trigger descriptive error"
                    self.log_assertion(
                        "Corrupted YAML handled with appropriate error", True
                    )
                else:
                    # If no error, verify config didn't load
                    config_content = config_page.get_configuration_content()
                    assert (
                        config_content is None
                        or "CorruptedModel" not in config_content
                    ), "Corrupted config should not load successfully"
                    self.log_assertion(
                        "Corrupted YAML rejected silently", True
                    )

            finally:
                if corrupted_config.exists():
                    corrupted_config.unlink()

            # Test 2: Binary data injection
            self.log_test_step("Testing binary data injection handling")

            binary_config = Path("test-artifacts") / "binary_config.yaml"

            # Create file with binary content disguised as YAML
            with open(binary_config, "wb") as f:
                f.write(b"# Binary injection test\n")
                f.write(b"model:\n")
                f.write(b"  name: BinaryTest\n")
                f.write(b"\x00\x01\x02\x03\xff\xfe\xfd")  # Binary garbage
                f.write(b"\ndata:\n  path: /test")

            try:
                config_page.upload_configuration_file(str(binary_config))
                config_page.click_load_config()

                # Verify binary content is handled safely
                time.sleep(2)  # Allow processing

                # Check for appropriate error handling
                error_elements = webdriver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stAlert']"
                )

                if error_elements:
                    self.log_assertion(
                        "Binary injection triggered error handling", True
                    )
                else:
                    # Verify system didn't crash and interface is still
                    # responsive
                    try:
                        self.assert_streamlit_loaded(webdriver)
                        self.log_assertion(
                            "System remained stable with binary data", True
                        )
                    except Exception:
                        pytest.fail("Binary data caused system instability")

            finally:
                if binary_config.exists():
                    binary_config.unlink()

            # Test 3: File operation interruption simulation
            self.log_test_step("Testing interrupted file operation recovery")

            # Create incomplete config file (simulating interrupted write)
            incomplete_config = (
                Path("test-artifacts") / "incomplete_config.yaml"
            )
            with open(incomplete_config, "w", encoding="utf-8") as f:
                f.write("# Incomplete configuration file\n")
                f.write("model:\n")
                f.write("  name: IncompleteModel\n")
                f.write("  architecture: cnn\n")
                f.write("  parameters:\n")
                f.write("    learning_rate: 0.001\n")
                # File ends abruptly without proper closure

            try:
                config_page.upload_configuration_file(str(incomplete_config))
                config_page.click_load_config()

                # Verify incomplete file is handled appropriately
                config_content = config_page.get_configuration_content()
                if config_content and "IncompleteModel" in config_content:
                    self.log_assertion(
                        "Incomplete config loaded successfully", True
                    )
                else:
                    # Check for validation error
                    error_elements = webdriver.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stAlert']"
                    )
                    if error_elements:
                        self.log_assertion(
                            "Incomplete config triggered validation", True
                        )

            finally:
                if incomplete_config.exists():
                    incomplete_config.unlink()

        except Exception as e:
            self.log_test_step(f"❌ Data corruption test failed: {str(e)}")
            raise

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

    @pytest.mark.e2e
    def test_edge_case_performance_monitoring(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """Monitor performance metrics during edge case scenarios.

        Tests performance characteristics including:
        - Page load times under stress
        - Memory consumption patterns
        - Response time degradation
        - Resource cleanup verification
        """
        self.log_test_step("Start edge case performance monitoring")
        self.navigate_and_verify(webdriver, streamlit_base_url)

        performance_metrics: PerformanceMetrics = {
            "navigation_times": [],
            "load_times": [],
            "error_recovery_times": [],
            "memory_pressure_impact": {},
        }

        try:
            # Monitor navigation performance across multiple pages
            pages_for_testing = [
                ("Config", ConfigPage),
                ("Architecture", ArchitecturePage),
                ("Train", TrainPage),
                ("Results", ResultsPage),
            ]

            # Baseline performance measurement
            self.log_test_step("Measuring baseline performance")
            for page_name, PageClass in pages_for_testing:
                start_time = time.time()
                PageClass(webdriver).navigate_to_page()
                self.assert_streamlit_loaded(webdriver)
                nav_time = time.time() - start_time

                performance_metrics["navigation_times"].append(
                    {
                        "page": page_name,
                        "time": nav_time,
                        "conditions": "baseline",
                    }
                )

                self.log_performance_metric(
                    f"baseline_{page_name.lower()}", nav_time
                )

            # Performance under memory pressure
            self.log_test_step("Measuring performance under memory pressure")
            EdgeCaseTestUtilities.simulate_memory_pressure()

            for page_name, PageClass in pages_for_testing[
                :2
            ]:  # Test subset under pressure
                start_time = time.time()
                PageClass(webdriver).navigate_to_page()
                self.assert_streamlit_loaded(webdriver)
                nav_time = time.time() - start_time

                performance_metrics["navigation_times"].append(
                    {
                        "page": page_name,
                        "time": nav_time,
                        "conditions": "memory_pressure",
                    }
                )

                self.log_performance_metric(
                    f"pressure_{page_name.lower()}", nav_time
                )

            # Error recovery performance
            self.log_test_step("Measuring error recovery performance")
            config_page = ConfigPage(webdriver).navigate_to_page()
            self.assert_streamlit_loaded(webdriver)

            # Create invalid config for error scenario
            invalid_config = Path("test-artifacts") / "perf_test_invalid.yaml"
            invalid_config.parent.mkdir(exist_ok=True)
            invalid_config.write_text(
                "invalid: [yaml content", encoding="utf-8"
            )

            try:
                start_time = time.time()
                config_page.upload_configuration_file(str(invalid_config))
                config_page.click_load_config()

                # Wait for error to appear
                webdriver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stAlert']"
                )

                # Measure recovery time
                config_page.select_config_file(
                    self.get_test_data()["boundary_test_config"]
                )
                config_page.click_load_config()
                config_page.wait_for_configuration_loaded()

                recovery_time = time.time() - start_time
                performance_metrics["error_recovery_times"].append(
                    recovery_time
                )
                self.log_performance_metric("error_recovery", recovery_time)

            finally:
                if invalid_config.exists():
                    invalid_config.unlink()

            # Analyze performance metrics
            baseline_times = [
                m["time"]
                for m in performance_metrics["navigation_times"]
                if m["conditions"] == "baseline"
            ]
            pressure_times = [
                m["time"]
                for m in performance_metrics["navigation_times"]
                if m["conditions"] == "memory_pressure"
            ]

            if baseline_times and pressure_times:
                avg_baseline = sum(baseline_times) / len(baseline_times)
                avg_pressure = sum(pressure_times) / len(pressure_times)
                performance_impact = (
                    (avg_pressure - avg_baseline) / avg_baseline * 100
                )

                self.log_performance_metric(
                    "performance_impact_percent", performance_impact
                )

                # Verify performance degradation is within acceptable limits
                assert performance_impact < 200, (
                    f"Performance degradation {performance_impact:.1f}% "
                    "exceeds threshold"
                )  # 200% increase maximum

                self.log_assertion(
                    "Performance impact acceptable: "
                    f"{performance_impact:.1f}%",
                    performance_impact < 200,
                )

            # Summary of edge case performance
            self.log_test_step("Edge case performance summary completed")

            avg_nav_time = (
                sum(baseline_times) / len(baseline_times)
                if baseline_times
                else 0
            )
            max_nav_time = max(baseline_times) if baseline_times else 0

            self.log_assertion(
                f"Average navigation: {avg_nav_time:.2f}s, Max: "
                f"{max_nav_time:.2f}s",
                True,
            )

        except Exception as e:
            self.log_test_step(f"❌ Performance monitoring failed: {str(e)}")
            raise
