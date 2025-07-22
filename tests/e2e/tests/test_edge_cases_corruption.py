"""
E2E tests for data corruption scenarios. This module contains tests
for handling corrupted or malformed data: - Malformed configuration
files - Interrupted file operations - Binary data injection - Recovery
mechanisms
"""

import time
from pathlib import Path

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from ..base_test import BaseE2ETest
from ..pages import ConfigPage


class TestEdgeCasesCorruption(BaseE2ETest):
    """Test class for data corruption edge cases."""

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
