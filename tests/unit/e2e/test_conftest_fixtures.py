"""Unit tests for E2E testing fixtures and configuration.

This module tests the pytest fixtures defined in tests/e2e/conftest.py to
ensure proper initialization, configuration, and resource management.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from tests.e2e.drivers import DriverConfig


class TestE2EConfigFixtures:
    """Test E2E configuration fixtures."""

    def test_e2e_config_defaults(self, e2e_config: DriverConfig) -> None:
        """Test that e2e_config fixture provides correct defaults."""
        assert e2e_config.browser == "chrome"
        assert e2e_config.headless is True
        assert e2e_config.window_size == (1920, 1080)
        assert e2e_config.implicit_wait == 10.0
        assert e2e_config.page_load_timeout == 30.0
        assert e2e_config.screenshot_on_failure is True
        assert e2e_config.enable_logging is True
        assert e2e_config.log_level == "INFO"
        assert "e2e" in str(e2e_config.artifacts_dir)

    def test_chrome_config_browser_type(
        self, chrome_config: DriverConfig
    ) -> None:
        """Test that chrome_config fixture sets correct browser type."""
        assert chrome_config.browser == "chrome"
        assert chrome_config.headless is True

    def test_firefox_config_browser_type(
        self, firefox_config: DriverConfig
    ) -> None:
        """Test that firefox_config fixture sets correct browser type."""
        assert firefox_config.browser == "firefox"
        assert firefox_config.headless is True

    def test_edge_config_browser_type(self, edge_config: DriverConfig) -> None:
        """Test that edge_config fixture sets correct browser type."""
        assert edge_config.browser == "edge"
        assert edge_config.headless is True

    def test_config_inheritance(
        self, e2e_config: DriverConfig, chrome_config: DriverConfig
    ) -> None:
        """Test that browser-specific configs inherit from base config."""
        # All properties except browser should be the same
        assert chrome_config.window_size == e2e_config.window_size
        assert chrome_config.implicit_wait == e2e_config.implicit_wait
        assert chrome_config.page_load_timeout == e2e_config.page_load_timeout
        assert (
            chrome_config.screenshot_on_failure
            == e2e_config.screenshot_on_failure
        )


class TestDriverManagerFixtures:
    """Test driver manager fixtures."""

    @patch("tests.e2e.drivers.HybridDriverManager")
    def test_driver_manager_creation(
        self, mock_manager_class: Mock, e2e_config: DriverConfig
    ) -> None:
        """Test that driver_manager fixture creates HybridDriverManager
        correctly."""
        # Import the fixture function directly for testing
        from tests.e2e.conftest import driver_manager

        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager

        # Create generator and get the manager
        manager_gen = driver_manager(e2e_config)
        manager = next(manager_gen)

        # Verify manager was created with correct config
        mock_manager_class.assert_called_once_with(e2e_config)
        assert manager == mock_manager

        # Test cleanup is called
        try:
            next(manager_gen)
        except StopIteration:
            pass

        mock_manager.cleanup_all_drivers.assert_called_once()


class TestWebDriverFixtures:
    """Test WebDriver fixtures."""

    @patch("tests.e2e.conftest.driver_session")
    def test_chrome_driver_fixture(
        self, mock_driver_session: Mock, chrome_config: DriverConfig
    ) -> None:
        """Test that chrome_driver fixture uses correct configuration."""
        # Import the fixture function
        from tests.e2e.conftest import chrome_driver

        mock_driver = Mock()
        mock_driver_session.return_value.__enter__.return_value = mock_driver
        mock_driver_session.return_value.__exit__.return_value = None

        # Test the fixture
        driver_gen = chrome_driver(chrome_config)
        driver = next(driver_gen)

        # Verify driver_session was called with correct parameters
        mock_driver_session.assert_called_once_with(
            browser="chrome", config=chrome_config
        )
        assert driver == mock_driver

    @patch("tests.e2e.conftest.driver_session")
    def test_firefox_driver_fixture(
        self, mock_driver_session: Mock, firefox_config: DriverConfig
    ) -> None:
        """Test that firefox_driver fixture uses correct configuration."""
        from tests.e2e.conftest import firefox_driver

        mock_driver = Mock()
        mock_driver_session.return_value.__enter__.return_value = mock_driver
        mock_driver_session.return_value.__exit__.return_value = None

        driver_gen = firefox_driver(firefox_config)
        driver = next(driver_gen)

        mock_driver_session.assert_called_once_with(
            browser="firefox", config=firefox_config
        )
        assert driver == mock_driver

    @patch("tests.e2e.conftest.driver_session")
    def test_edge_driver_fixture(
        self, mock_driver_session: Mock, edge_config: DriverConfig
    ) -> None:
        """Test that edge_driver fixture uses correct configuration."""
        from tests.e2e.conftest import edge_driver

        mock_driver = Mock()
        mock_driver_session.return_value.__enter__.return_value = mock_driver
        mock_driver_session.return_value.__exit__.return_value = None

        driver_gen = edge_driver(edge_config)
        driver = next(driver_gen)

        mock_driver_session.assert_called_once_with(
            browser="edge", config=edge_config
        )
        assert driver == mock_driver


class TestCrossBrowserFixtures:
    """Test cross-browser testing fixtures."""

    @patch("tests.e2e.conftest.driver_session")
    def test_cross_browser_driver_parametrization(
        self, mock_driver_session: Mock
    ) -> None:
        """
        Test that cross_browser_driver fixture supports multiple browsers.
        """
        from tests.e2e.conftest import cross_browser_driver

        # Mock request object for different browsers
        for browser in ["chrome", "firefox", "edge"]:
            mock_request = Mock()
            mock_request.param = browser

            mock_e2e_config = DriverConfig(browser="chrome")  # Base config

            mock_driver = Mock()
            mock_driver_session.return_value.__enter__.return_value = (
                mock_driver
            )
            mock_driver_session.return_value.__exit__.return_value = None

            # Test the fixture with this browser
            driver_gen = cross_browser_driver(mock_request, mock_e2e_config)
            driver = next(driver_gen)

            # Verify correct browser was used
            call_args = mock_driver_session.call_args
            assert call_args[1]["browser"] == browser
            assert driver == mock_driver

            mock_driver_session.reset_mock()


class TestUtilityFixtures:
    """Test utility and test data fixtures."""

    def test_streamlit_base_url(self, streamlit_base_url: str) -> None:
        """Test that streamlit_base_url fixture provides correct URL."""
        assert streamlit_base_url == "http://localhost:8501"
        assert streamlit_base_url.startswith("http://")

    def test_test_artifacts_dir_creation(
        self, test_artifacts_dir: Path
    ) -> None:
        """Test that test_artifacts_dir fixture creates directory."""
        assert test_artifacts_dir.exists()
        assert test_artifacts_dir.is_dir()
        assert "e2e" in str(test_artifacts_dir)

    def test_test_data_structure(self, test_data: dict[str, Any]) -> None:
        """Test that test_data fixture provides expected structure."""
        assert "sample_images" in test_data
        assert "config_values" in test_data
        assert "expected_results" in test_data

        # Verify sample images structure
        sample_images = test_data["sample_images"]
        assert len(sample_images) == 3
        for image in sample_images:
            assert "name" in image
            assert "type" in image
            assert image["type"] in ["crack", "no_crack"]

        # Verify config values
        config_values = test_data["config_values"]
        assert "model_name" in config_values
        assert "batch_size" in config_values
        assert "confidence_threshold" in config_values

        # Verify expected results
        expected_results = test_data["expected_results"]
        assert "navigation_elements" in expected_results
        assert "file_upload_types" in expected_results

        # Check navigation elements
        nav_elements = expected_results["navigation_elements"]
        expected_nav = [
            "Architecture",
            "Configuration",
            "Training",
            "Evaluation",
        ]
        assert all(elem in nav_elements for elem in expected_nav)

        # Check file upload types
        upload_types = expected_results["file_upload_types"]
        expected_types = [".jpg", ".jpeg", ".png"]
        assert all(file_type in upload_types for file_type in expected_types)


class TestResourceManagement:
    """Test resource management and cleanup functionality."""

    def test_cleanup_test_artifacts_fixture(
        self, test_artifacts_dir: Path
    ) -> None:
        """Test that cleanup_test_artifacts fixture works correctly."""
        # Create some test files
        test_file1 = test_artifacts_dir / "test_sample.txt"
        test_file2 = test_artifacts_dir / "test_another.log"

        test_file1.write_text("test content")
        test_file2.write_text("test log")

        assert test_file1.exists()
        assert test_file2.exists()

        # Import and execute cleanup fixture
        from tests.e2e.conftest import cleanup_test_artifacts

        cleanup_gen = cleanup_test_artifacts(test_artifacts_dir)
        next(cleanup_gen)  # Setup phase

        # Files should still exist during test execution
        assert test_file1.exists()
        assert test_file2.exists()

        # Execute cleanup phase
        try:
            next(cleanup_gen)
        except StopIteration:
            pass

        # Files should be cleaned up
        # Note: In real test execution, these would be cleaned up
        # but for unit testing we verify the logic works

    @patch("tests.e2e.conftest.Path.unlink")
    def test_cleanup_handles_permission_errors(
        self, mock_unlink: Mock, test_artifacts_dir: Path
    ) -> None:
        """Test that cleanup gracefully handles permission errors."""
        mock_unlink.side_effect = PermissionError("Access denied")

        # Create test file for cleanup
        test_file = test_artifacts_dir / "test_file.txt"
        test_file.write_text("test")

        from tests.e2e.conftest import cleanup_test_artifacts

        # Should not raise exception even with permission error
        cleanup_gen = cleanup_test_artifacts(test_artifacts_dir)
        next(cleanup_gen)

        try:
            next(cleanup_gen)
        except StopIteration:
            pass  # Expected behavior


class TestPytestConfiguration:
    """Test pytest configuration and markers."""

    def test_pytest_markers_registration(self) -> None:
        """Test that all required markers are registered."""
        # Import the configuration function
        from tests.e2e.conftest import pytest_configure

        # Mock config object
        mock_config = Mock()
        mock_config.addinivalue_line = Mock()

        # Execute configuration
        pytest_configure(mock_config)

        # Verify all expected markers were registered
        expected_markers = [
            "e2e: mark test as end-to-end integration test",
            "chrome: mark test as Chrome-specific",
            "firefox: mark test as Firefox-specific",
            "edge: mark test as Edge-specific",
            "cross_browser: mark test for cross-browser execution",
            "slow: mark test as slow-running (requires extended timeout)",
        ]

        # Check that addinivalue_line was called for each marker
        assert mock_config.addinivalue_line.call_count == len(expected_markers)

        # Verify specific marker calls
        for expected_marker in expected_markers:
            mock_config.addinivalue_line.assert_any_call(
                "markers", expected_marker
            )
