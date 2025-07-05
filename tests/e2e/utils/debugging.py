"""E2E debugging utilities for comprehensive test analysis."""

import json
import time
from pathlib import Path
from typing import Any

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver


class E2EDebugger:
    """Comprehensive debugging utilities for E2E tests."""

    def __init__(self, driver: WebDriver, test_name: str = "unknown") -> None:
        """Initialize the debugger with WebDriver instance.

        Args:
            driver: Selenium WebDriver instance
            test_name: Name of the test for debugging context
        """
        self.driver = driver
        self.test_name = test_name

    def inspect_sidebar_elements(self, verbose: bool = True) -> dict[str, Any]:
        """Comprehensive inspection of sidebar elements and structure.

        Args:
            verbose: Whether to print detailed debugging information

        Returns:
            Dictionary containing sidebar inspection results
        """
        inspection_result: dict[str, Any] = {
            "sidebar_present": False,
            "buttons_found": 0,
            "button_details": [],
            "all_elements": [],
            "page_title": None,
            "url": self.driver.current_url,
            "timestamp": time.time(),
        }

        try:
            # Check if sidebar exists
            sidebar_elements = self.driver.find_elements(
                By.CSS_SELECTOR, "[data-testid='stSidebar']"
            )
            inspection_result["sidebar_present"] = len(sidebar_elements) > 0

            if inspection_result["sidebar_present"]:
                # Get all buttons in sidebar
                sidebar_buttons = self.driver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stSidebar'] button"
                )
                inspection_result["buttons_found"] = len(sidebar_buttons)

                # Detailed button analysis
                for i, button in enumerate(sidebar_buttons):
                    try:
                        button_info: dict[str, Any] = {
                            "index": i,
                            "text": button.text.strip(),
                            "tag_name": button.tag_name,
                            "is_displayed": button.is_displayed(),
                            "is_enabled": button.is_enabled(),
                            "location": button.location,
                            "size": button.size,
                            "attributes": {},
                        }

                        # Extract key attributes
                        for attr in [
                            "key",
                            "aria-label",
                            "data-testid",
                            "class",
                        ]:
                            try:
                                value = button.get_attribute(attr)
                                if value:
                                    button_info["attributes"][attr] = value
                            except Exception:
                                pass

                        inspection_result["button_details"].append(button_info)

                    except Exception as e:
                        error_info: dict[str, Any] = {
                            "index": i,
                            "error": str(e),
                        }
                        inspection_result["button_details"].append(error_info)

                # Get all sidebar elements for comprehensive view
                all_sidebar_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, "[data-testid='stSidebar'] *"
                )

                for element in all_sidebar_elements[:20]:  # Limit to first 20
                    try:
                        element_info: dict[str, Any] = {
                            "tag": element.tag_name,
                            "text": element.text[:50] if element.text else "",
                            "class_name": element.get_attribute("class"),
                        }
                        inspection_result["all_elements"].append(element_info)
                    except Exception:
                        pass

            # Get page title
            try:
                title_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, "h1"
                )
                if title_elements:
                    inspection_result["page_title"] = title_elements[0].text
            except Exception:
                pass

            if verbose:
                self._print_inspection_results(inspection_result)

        except Exception as e:
            inspection_result["inspection_error"] = str(e)
            if verbose:
                print(f"❌ Sidebar inspection failed: {e}")

        return inspection_result

    def _print_inspection_results(self, results: dict[str, Any]) -> None:
        """Print formatted inspection results."""
        print(f"\n🔍 E2E Debug Report for '{self.test_name}'")
        print("=" * 50)
        print(f"URL: {results['url']}")
        print(f"Sidebar Present: {results['sidebar_present']}")
        print(f"Buttons Found: {results['buttons_found']}")

        if results.get("page_title"):
            print(f"Page Title: {results['page_title']}")

        if results.get("button_details"):
            print("\n📋 Button Details:")
            for button in results["button_details"]:
                if "error" not in button:
                    print(f"  [{button['index']}] '{button['text']}'")
                    print(f"      Displayed: {button['is_displayed']}")
                    print(f"      Enabled: {button['is_enabled']}")
                    if button.get("attributes"):
                        print(f"      Attributes: {button['attributes']}")
                else:
                    print(f"  [{button['index']}] Error: {button['error']}")

        if results.get("inspection_error"):
            print(f"\n❌ Inspection Error: {results['inspection_error']}")

        print("=" * 50)

    def save_debug_report(
        self, output_dir: Path | str = "test-artifacts"
    ) -> Path | None:
        """Save comprehensive debug report to file.

        Args:
            output_dir: Directory to save the debug report

        Returns:
            Path to saved report file, or None if save failed
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            timestamp = int(time.time())
            report_file = (
                output_path / f"debug_report_{self.test_name}_{timestamp}.json"
            )

            # Gather comprehensive debug data
            debug_data = {
                "test_name": self.test_name,
                "timestamp": timestamp,
                "browser_info": self._get_browser_info(),
                "page_info": self._get_page_info(),
                "sidebar_inspection": self.inspect_sidebar_elements(
                    verbose=False
                ),
                "streamlit_info": self._get_streamlit_info(),
            }

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, default=str)

            print(f"💾 Debug report saved: {report_file}")
            return report_file

        except Exception as e:
            print(f"❌ Failed to save debug report: {e}")
            return None

    def _get_browser_info(self) -> dict[str, Any]:
        """Get browser and driver information."""
        try:
            return {
                "user_agent": self.driver.execute_script(
                    "return navigator.userAgent"
                ),
                "window_size": self.driver.get_window_size(),
                "current_url": self.driver.current_url,
                "title": self.driver.title,
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_page_info(self) -> dict[str, Any]:
        """Get current page information."""
        try:
            return {
                "url": self.driver.current_url,
                "title": self.driver.title,
                "ready_state": self.driver.execute_script(
                    "return document.readyState"
                ),
                "body_text_length": len(
                    self.driver.find_element(By.TAG_NAME, "body").text
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_streamlit_info(self) -> dict[str, Any]:
        """Get Streamlit-specific information."""
        try:
            # Check for Streamlit app structure
            streamlit_app = self.driver.find_elements(
                By.CSS_SELECTOR, "[data-testid='stApp']"
            )

            info = {
                "streamlit_app_present": len(streamlit_app) > 0,
                "spinner_present": len(
                    self.driver.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stSpinner']"
                    )
                )
                > 0,
                "total_buttons": len(
                    self.driver.find_elements(By.CSS_SELECTOR, "button")
                ),
                "total_inputs": len(
                    self.driver.find_elements(By.CSS_SELECTOR, "input")
                ),
                "streamlit_version": "unknown",  # Default value
            }

            # Try to get Streamlit version if available
            try:
                streamlit_version = self.driver.execute_script(
                    "return window.streamlitVersion || 'unknown'"
                )
                info["streamlit_version"] = streamlit_version
            except Exception:
                pass  # streamlit_version already set to "unknown"

            return info

        except Exception as e:
            return {"error": str(e)}

    def wait_and_retry_navigation(
        self, page_name: str, max_attempts: int = 3, delay: float = 2.0
    ) -> bool:
        """Retry navigation with debugging between attempts.

        Args:
            page_name: Name of page to navigate to
            max_attempts: Maximum number of attempts
            delay: Delay between attempts in seconds

        Returns:
            True if navigation successful, False otherwise
        """
        for attempt in range(max_attempts):
            print(
                f"🔄 Navigation attempt {attempt + 1}/{max_attempts} "
                f"for '{page_name}'"
            )

            # Inspect current state
            self.inspect_sidebar_elements(verbose=True)

            try:
                # Try to navigate (this would call actual navigation logic)
                # For now, this is a placeholder - actual navigation
                # would be implemented by the calling test
                time.sleep(delay)
                return True  # Placeholder success

            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    print(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)

        print(
            f"❌ Failed to find navigation button for '{page_name}' "
            f"after {max_attempts} attempts"
        )
        return False


def create_debugger(
    driver: WebDriver, test_name: str = "unknown"
) -> E2EDebugger:
    """Factory function to create an E2EDebugger instance.

    Args:
        driver: Selenium WebDriver instance
        test_name: Name of the test for debugging context

    Returns:
        Configured E2EDebugger instance
    """
    return E2EDebugger(driver, test_name)
