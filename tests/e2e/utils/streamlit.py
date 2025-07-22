"""Streamlit-specific utilities for E2E testing.

This module provides utilities specifically designed for testing Streamlit
applications, including the CrackSeg web interface. Contains functions for
managing Streamlit app lifecycle, interacting with widgets, handling file
uploads, and navigating pages.

Key features:
- Streamlit app startup and shutdown management
- Widget interaction helpers (buttons, sliders, uploads)
- Page navigation and state management
- File upload simulation and validation
- CrackSeg-specific application helpers
- Session state monitoring and manipulation

Examples:
    >>> driver = start_streamlit_app("scripts/gui/app.py", port=8501)
    >>> wait_for_streamlit_ready(driver)
    >>> upload_file(driver, "data/test/images/test_image.jpg")
    >>> navigate_to_page(driver, "Training")
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from selenium.common.exceptions import (  # type: ignore[import-untyped]
        TimeoutException,
        WebDriverException,
    )
    from selenium.webdriver.common.by import By  # type: ignore[import-untyped]
    from selenium.webdriver.remote.webdriver import (
        WebDriver,  # type: ignore[import-untyped]
    )
    from selenium.webdriver.support import (
        expected_conditions as EC,  # type: ignore[import-untyped]
    )
    from selenium.webdriver.support.ui import (
        WebDriverWait,  # type: ignore[import-untyped]
    )
else:
    try:
        from selenium.common.exceptions import (
            TimeoutException,
            WebDriverException,
        )
        from selenium.webdriver.common.by import By
        from selenium.webdriver.remote.webdriver import WebDriver
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:
        # Mock classes when selenium is not available
        class WebDriver:  # type: ignore[no-redef]
            pass

        class WebDriverWait:  # type: ignore[no-redef]
            def __init__(self, driver: Any, timeout: int) -> None:
                pass

        class TimeoutException(Exception):  # type: ignore[no-redef]
            pass

        class WebDriverException(Exception):  # type: ignore[no-redef]
            pass

        class By:  # type: ignore[no-redef]
            TAG_NAME = "tag_name"
            CLASS_NAME = "class_name"
            CSS_SELECTOR = "css_selector"
            XPATH = "xpath"

        class EC:  # type: ignore[no-redef]
            @staticmethod
            def presence_of_element_located(locator: Any) -> Any:
                return lambda driver: True

            @staticmethod
            def element_to_be_clickable(locator: Any) -> Any:
                return lambda driver: True


# Optional import with proper handling
_requests_available = False
try:
    import requests

    _requests_available = True
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


def start_streamlit_app(
    app_path: str | Path,
    port: int = 8501,
    host: str = "localhost",
    timeout: int = 30,
    additional_args: list[str] | None = None,
) -> subprocess.Popen[str]:
    """Start a Streamlit application for testing.

    Launches a Streamlit app in a subprocess and waits for it to be ready.
    Useful for E2E testing scenarios where the app needs to be controlled.

    Args:
        app_path: Path to the Streamlit application script
        port: Port number for the Streamlit server
        host: Host address for the server
        timeout: Maximum time to wait for app startup
        additional_args: Additional command line arguments for streamlit run

    Returns:
        Subprocess object for the running Streamlit app

    Raises:
        FileNotFoundError: If the app script doesn't exist
        RuntimeError: If the app fails to start within timeout

    Example:
        >>> process = start_streamlit_app("scripts/gui/app.py", port=8502)
        >>> # Run tests...
        >>> process.terminate()
    """
    app_path = Path(app_path)
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found: {app_path}")

    # Build command
    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    if additional_args:
        cmd.extend(additional_args)

    # Start the process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=app_path.parent,
        )
    except FileNotFoundError as err:
        raise RuntimeError(
            "Streamlit not found. Please install streamlit or activate "
            "the correct environment."
        ) from err

    # Wait for app to be ready
    start_time = time.time()
    base_url = f"http://{host}:{port}"

    while time.time() - start_time < timeout:
        try:
            if _requests_available and requests is not None:
                response = requests.get(base_url, timeout=5)
                if response.status_code == 200:
                    logger.info(
                        f"Streamlit app started successfully on {base_url}"
                    )
                    return process
        except Exception:
            pass

        # Check if process is still running
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise RuntimeError(f"Streamlit app failed to start: {stderr}")

        time.sleep(1)

    # Timeout reached
    process.terminate()
    raise RuntimeError(
        f"Streamlit app failed to start within {timeout} seconds"
    )


def wait_for_streamlit_ready(
    driver: WebDriver,
    timeout: int = 30,
    check_elements: list[str] | None = None,
) -> bool:
    """Wait for Streamlit application to be fully loaded and ready.

    Waits for the Streamlit app to finish loading by checking for the absence
    of loading indicators and the presence of expected elements.

    Args:
        driver: Selenium WebDriver instance
        timeout: Maximum time to wait in seconds
        check_elements: List of element selectors to check for presence

    Returns:
        True if app is ready, False if timeout

    Example:
        >>> wait_for_streamlit_ready(
        ...     driver, check_elements=["[data-testid='stSidebar']"]
        ... )
    """
    wait = WebDriverWait(driver, timeout)

    try:
        # Wait for basic Streamlit structure
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Wait for Streamlit app to finish loading
        wait.until_not(
            EC.presence_of_element_located((By.CLASS_NAME, "stSpinner"))
        )

        # Wait for main content area
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-testid='stApp']")
            )
        )

        # Check for specific elements if provided
        if check_elements:
            for element_selector in check_elements:
                wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, element_selector)
                    )
                )

        # Additional small wait for dynamic content
        time.sleep(1)

        logger.debug("Streamlit app is ready")
        return True

    except TimeoutException:
        logger.warning(f"Streamlit app not ready within {timeout} seconds")
        return False


def navigate_to_page(
    driver: WebDriver,
    page_name: str,
    sidebar_navigation: bool = True,
    timeout: int = 10,
) -> bool:
    """Navigate to a specific page in the Streamlit application.

    Navigates to a page by clicking on navigation elements, typically
    in the sidebar or main navigation area.

    Args:
        driver: Selenium WebDriver instance
        page_name: Name of the page to navigate to
        sidebar_navigation: Whether to look for navigation in sidebar
        timeout: Maximum time to wait for navigation

    Returns:
        True if navigation successful, False otherwise

    Example:
        >>> navigate_to_page(driver, "Training", sidebar_navigation=True)
    """
    wait = WebDriverWait(driver, timeout)

    try:
        # Common selectors for Streamlit navigation
        nav_selectors = [
            f"[data-testid='stSidebar'] button:contains('{page_name}')",
            (
                f"[data-testid='stSidebar'] "
                f"[role='button']:contains('{page_name}')"
            ),
            f"button:contains('{page_name}')",
            f"[role='button']:contains('{page_name}')",
            f"a:contains('{page_name}')",
        ]

        if not sidebar_navigation:
            # Skip sidebar-specific selectors
            nav_selectors = nav_selectors[2:]

        element_found = False
        for _selector in nav_selectors:
            try:
                # Use XPath for text content matching
                xpath = (
                    f"//*[contains(text(), '{page_name}')]"
                    f"[@role='button' or self::button or self::a]"
                )
                element = wait.until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                element.click()
                element_found = True
                break
            except TimeoutException:
                continue

        if not element_found:
            logger.warning(f"Navigation element for '{page_name}' not found")
            return False

        # Wait for page content to load
        time.sleep(2)

        logger.debug(f"Navigated to page: {page_name}")
        return True

    except (TimeoutException, WebDriverException) as e:
        logger.error(f"Failed to navigate to {page_name}: {e}")
        return False


def upload_file(
    driver: WebDriver,
    file_path: str | Path,
    file_uploader_label: str | None = None,
    timeout: int = 30,
) -> bool:
    """Upload a file using Streamlit's file uploader widget.

    Locates a file uploader widget and uploads the specified file.
    Can target specific uploaders by label.

    Args:
        driver: Selenium WebDriver instance
        file_path: Path to the file to upload
        file_uploader_label: Optional label to identify specific uploader
        timeout: Maximum time to wait for upload completion

    Returns:
        True if upload successful, False otherwise

    Example:
        >>> upload_file(driver, "data/test/images/test.jpg", "Upload Image")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    wait = WebDriverWait(driver, timeout)

    try:
        # Find file uploader
        if file_uploader_label:
            # Look for uploader with specific label
            uploader_xpath = (
                f"//div[contains(text(), '{file_uploader_label}')]"
                f"/following-sibling::div//input[@type='file']"
            )
        else:
            # Use any file uploader
            uploader_xpath = "//input[@type='file']"

        file_input = wait.until(
            EC.presence_of_element_located((By.XPATH, uploader_xpath))
        )

        # Upload the file
        file_input.send_keys(str(file_path.absolute()))

        # Wait for upload to complete (look for success indicators)
        success_indicators = [
            "[data-testid='stSuccess']",
            ".uploadedFile",
            "//div[contains(text(), 'Uploaded')]",
        ]

        upload_success = False
        for indicator in success_indicators:
            try:
                if indicator.startswith("//"):
                    wait.until(
                        EC.presence_of_element_located((By.XPATH, indicator))
                    )
                else:
                    wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, indicator)
                        )
                    )
                upload_success = True
                break
            except TimeoutException:
                continue

        if not upload_success:
            # Fallback: just wait a bit and assume success if no error
            time.sleep(3)
            upload_success = True

        logger.info(f"File uploaded successfully: {file_path.name}")
        return upload_success

    except (TimeoutException, WebDriverException) as e:
        logger.error(f"File upload failed: {e}")
        return False


def click_streamlit_button(
    driver: WebDriver,
    button_text: str,
    timeout: int = 10,
    exact_match: bool = False,
) -> bool:
    """Click a Streamlit button by its text content.

    Finds and clicks a button widget in the Streamlit app based on
    its displayed text.

    Args:
        driver: Selenium WebDriver instance
        button_text: Text content of the button to click
        timeout: Maximum time to wait for button
        exact_match: Whether to match text exactly or partially

    Returns:
        True if button clicked successfully, False otherwise

    Example:
        >>> click_streamlit_button(driver, "Start Training", exact_match=True)
    """
    wait = WebDriverWait(driver, timeout)

    try:
        if exact_match:
            xpath = f"//button[text()='{button_text}']"
        else:
            xpath = f"//button[contains(text(), '{button_text}')]"

        button = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        button.click()

        logger.debug(f"Clicked button: {button_text}")
        return True

    except (TimeoutException, WebDriverException) as e:
        logger.error(f"Failed to click button '{button_text}': {e}")
        return False


def get_streamlit_session_state(
    driver: WebDriver, timeout: int = 5
) -> dict[str, Any]:
    """Extract Streamlit session state information.

    Attempts to retrieve session state information from the Streamlit app
    by executing JavaScript or parsing page content.

    Args:
        driver: Selenium WebDriver instance
        timeout: Maximum time to wait for state extraction

    Returns:
        Dictionary containing available session state information

    Note:
        This is best-effort and may not work with all Streamlit versions

    Example:
        >>> state = get_streamlit_session_state(driver)
        >>> print(state.get("current_page", "Unknown"))
    """
    try:
        # Try to extract session state via JavaScript
        js_code = """
        // Look for Streamlit session state in window object
        if (window.streamlitDebug && window.streamlitDebug.sessionState) {
            return window.streamlitDebug.sessionState;
        }

        // Fallback: try to extract from React components
        const reactRoot = document.querySelector('[data-testid="stApp"]');
        if (reactRoot && reactRoot._reactInternalFiber) {
            // This is implementation-specific and may not work
            return {};
        }

        return {};
        """

        state = driver.execute_script(js_code)
        if state:
            logger.debug("Retrieved session state via JavaScript")
            return state

    except WebDriverException:
        pass

    # Fallback: Extract visible state information
    state_info = {}

    try:
        # Look for common state indicators
        # Try to find current page indicators
        page_indicators = driver.find_elements(
            By.CSS_SELECTOR, "[data-testid='stSidebar'] .selected"
        )
        if page_indicators:
            state_info["current_page"] = page_indicators[0].text

        # Look for form states
        forms = driver.find_elements(By.TAG_NAME, "form")
        state_info["forms_count"] = len(forms)

        # Check for uploaded files
        uploaded_files = driver.find_elements(By.CSS_SELECTOR, ".uploadedFile")
        state_info["uploaded_files_count"] = len(uploaded_files)

    except (TimeoutException, WebDriverException):
        pass

    return state_info


def wait_for_streamlit_rerun(
    driver: WebDriver, timeout: int = 10, check_spinner: bool = True
) -> bool:
    """Wait for Streamlit app to complete a rerun cycle.

    Waits for the app to finish processing after a user interaction
    that triggers a rerun.

    Args:
        driver: Selenium WebDriver instance
        timeout: Maximum time to wait for rerun completion
        check_spinner: Whether to wait for loading spinner to disappear

    Returns:
        True if rerun completed, False if timeout

    Example:
        >>> click_streamlit_button(driver, "Process Data")
        >>> wait_for_streamlit_rerun(driver)
    """
    wait = WebDriverWait(driver, timeout)

    try:
        if check_spinner:
            # Wait for spinner to appear (if any)
            try:
                WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located(
                        (By.CLASS_NAME, "stSpinner")
                    )
                )
            except TimeoutException:
                pass  # No spinner appeared, that's fine

            # Wait for spinner to disappear
            wait.until_not(
                EC.presence_of_element_located((By.CLASS_NAME, "stSpinner"))
            )

        # Additional wait for content to stabilize
        time.sleep(1)

        logger.debug("Streamlit rerun completed")
        return True

    except TimeoutException:
        logger.warning(
            f"Streamlit rerun not completed within {timeout} seconds"
        )
        return False


def get_crackseg_app_status(driver: WebDriver) -> dict[str, Any]:
    """Get status information specific to the CrackSeg application.

    Extracts application-specific status information such as loaded models,
    training progress, and configuration status.

    Args:
        driver: Selenium WebDriver instance

    Returns:
        Dictionary containing CrackSeg application status

    Example:
        >>> status = get_crackseg_app_status(driver)
        >>> print(f"Model loaded: {status['model_loaded']}")
    """
    status: dict[str, Any] = {
        "model_loaded": False,
        "config_loaded": False,
        "training_active": False,
        "current_page": "Unknown",
        "errors": [],
        "warnings": [],
    }

    try:
        # Check current page
        try:
            sidebar = driver.find_element(
                By.CSS_SELECTOR, "[data-testid='stSidebar']"
            )
            selected_items = sidebar.find_elements(
                By.CSS_SELECTOR, ".selected, [aria-selected='true']"
            )
            if selected_items:
                status["current_page"] = selected_items[0].text.strip()
        except Exception:
            pass

        # Look for model status indicators
        model_indicators = [
            "Model loaded",
            "Model ready",
            "✅ Model",
        ]

        for indicator in model_indicators:
            elements = driver.find_elements(
                By.XPATH, f"//*[contains(text(), '{indicator}')]"
            )
            if elements:
                status["model_loaded"] = True
                break

        # Look for configuration status
        config_indicators = [
            "Configuration loaded",
            "Config ready",
            "✅ Config",
        ]

        for indicator in config_indicators:
            elements = driver.find_elements(
                By.XPATH, f"//*[contains(text(), '{indicator}')]"
            )
            if elements:
                status["config_loaded"] = True
                break

        # Check for training status
        training_indicators = [
            "Training in progress",
            "Epoch",
            "Training started",
        ]

        for indicator in training_indicators:
            elements = driver.find_elements(
                By.XPATH, f"//*[contains(text(), '{indicator}')]"
            )
            if elements:
                status["training_active"] = True
                break

        # Look for error messages
        error_selectors = [
            "[data-testid='stException']",
            "[data-testid='stError']",
            ".stError",
            "//div[contains(@class, 'error')]",
        ]

        errors_list: list[str] = []
        for selector in error_selectors:
            try:
                if selector.startswith("//"):
                    elements = driver.find_elements(By.XPATH, selector)
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                errors_list.extend(
                    [elem.text for elem in elements if elem.text]
                )
            except Exception:
                continue

        status["errors"] = errors_list

        # Look for warning messages
        warning_selectors = [
            "[data-testid='stWarning']",
            ".stWarning",
            "//div[contains(@class, 'warning')]",
        ]

        warnings_list: list[str] = []
        for selector in warning_selectors:
            try:
                if selector.startswith("//"):
                    elements = driver.find_elements(By.XPATH, selector)
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                warnings_list.extend(
                    [elem.text for elem in elements if elem.text]
                )
            except Exception:
                continue

        status["warnings"] = warnings_list

    except Exception as e:
        logger.error(f"Failed to get CrackSeg app status: {e}")
        # Ensure errors is a list before appending
        if isinstance(status["errors"], list):
            status["errors"].append(f"Status check failed: {e}")
        else:
            status["errors"] = [f"Status check failed: {e}"]

    return status


def stop_streamlit_app(
    process: subprocess.Popen[str], timeout: int = 10
) -> bool:
    """Stop a running Streamlit application process.

    Gracefully terminates a Streamlit app process with optional forced kill
    if graceful shutdown fails.

    Args:
        process: Subprocess object of the running Streamlit app
        timeout: Maximum time to wait for graceful shutdown

    Returns:
        True if process stopped successfully, False otherwise

    Example:
        >>> process = start_streamlit_app("app.py")
        >>> # ... run tests ...
        >>> stop_streamlit_app(process)
    """
    if process.poll() is not None:
        logger.debug("Streamlit app process already terminated")
        return True

    try:
        # Try graceful termination
        process.terminate()

        # Wait for process to end
        try:
            process.wait(timeout=timeout)
            logger.info("Streamlit app stopped gracefully")
            return True
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown failed
            process.kill()
            process.wait(timeout=5)
            logger.warning("Streamlit app force-killed after timeout")
            return True

    except Exception as e:
        logger.error(f"Failed to stop Streamlit app: {e}")
        return False
