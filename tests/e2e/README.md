# E2E Testing Framework - Fixture Usage Guide

This directory contains the End-to-End (E2E) testing framework for the CrackSeg Streamlit
application, implemented as part of **Task 14.2 - Test Fixture Creation and Configuration**.

## Overview

The E2E testing framework provides pytest fixtures that leverage the HybridDriverManager system
(from subtask 14.1) to enable comprehensive browser automation testing of the Streamlit application.

## Architecture

```txt
tests/e2e/
├── conftest.py                    # Main fixture definitions
├── drivers/                       # WebDriver management system (14.1)
├── test_fixture_usage_example.py  # Usage examples
└── README.md                      # This documentation
```

## Key Features

- **Browser Configuration Fixtures**: Pre-configured settings for Chrome, Firefox, and Edge
- **WebDriver Fixtures**: Function-scoped driver instances with automatic cleanup
- **Cross-Browser Testing**: Parametrized fixtures for multi-browser test execution
- **Resource Management**: Automatic cleanup of artifacts and drivers
- **Test Data Fixtures**: Mock data for Streamlit application testing
- **Screenshot on Failure**: Automatic capture of test failure screenshots

## Fixture Categories

### 1. Configuration Fixtures

#### `e2e_config` (session scope)

Default E2E testing configuration optimized for headless browser testing.

```python
def test_with_config(e2e_config):
    assert e2e_config.browser == "chrome"
    assert e2e_config.headless is True
    assert e2e_config.window_size == (1920, 1080)
```

#### Browser-Specific Configs

- `chrome_config`: Chrome-optimized configuration
- `firefox_config`: Firefox-optimized configuration
- `edge_config`: Edge-optimized configuration

### 2. WebDriver Fixtures

#### Single Browser Fixtures

```python
def test_chrome_functionality(chrome_driver):
    chrome_driver.get("http://localhost:8501")
    assert "Streamlit" in chrome_driver.title

def test_firefox_functionality(firefox_driver):
    firefox_driver.get("http://localhost:8501")
    # Test Firefox-specific behavior

def test_edge_functionality(edge_driver):
    edge_driver.get("http://localhost:8501")
    # Test Edge-specific behavior
```

#### Default WebDriver

```python
def test_with_default_driver(webdriver):
    """Uses HybridDriverManager with default browser."""
    webdriver.get("http://localhost:8501")
    # Test with automatically managed driver
```

#### Cross-Browser Testing

```python
@pytest.mark.cross_browser
def test_cross_browser_compatibility(cross_browser_driver):
    """Runs automatically on Chrome, Firefox, and Edge."""
    cross_browser_driver.get("http://localhost:8501")
    # Test will run 3 times (once per browser)
```

### 3. Test Data and Utility Fixtures

#### Application URL

```python
def test_navigation(chrome_driver, streamlit_base_url):
    chrome_driver.get(streamlit_base_url)  # http://localhost:8501
```

#### Test Data

```python
def test_file_upload(chrome_driver, test_data):
    sample_images = test_data["sample_images"]
    config_values = test_data["config_values"]
    expected_results = test_data["expected_results"]
```

#### Artifacts Management

```python
def test_with_artifacts(chrome_driver, test_artifacts_dir):
    # Save manual screenshot
    screenshot_path = test_artifacts_dir / "manual_test.png"
    chrome_driver.save_screenshot(str(screenshot_path))
```

### 4. Resource Management Fixtures

#### Automatic Cleanup (autouse)

```python
# cleanup_test_artifacts runs automatically after each test
# No manual invocation needed
```

#### Screenshot on Failure

```python
def test_with_failure_screenshot(chrome_driver, screenshot_on_failure):
    # If this test fails, screenshot will be automatically captured
    chrome_driver.get("http://localhost:8501")
    assert False  # Will trigger screenshot capture
```

## Usage Examples

### Basic E2E Test

```python
import pytest
from selenium.webdriver.common.by import By

@pytest.mark.e2e
def test_streamlit_loads(chrome_driver, streamlit_base_url):
    """Test that Streamlit application loads successfully."""
    chrome_driver.get(streamlit_base_url)
    assert chrome_driver.find_element(By.TAG_NAME, "body")
```

### Cross-Browser Test

```python
@pytest.mark.e2e
@pytest.mark.cross_browser
def test_navigation_works_everywhere(cross_browser_driver, streamlit_base_url, test_data):
    """Test navigation across all supported browsers."""
    cross_browser_driver.get(streamlit_base_url)

    expected_nav = test_data["expected_results"]["navigation_elements"]
    page_text = cross_browser_driver.page_source

    # Verify navigation elements exist
    for nav_item in expected_nav:
        assert nav_item.lower() in page_text.lower()
```

### File Upload Test

```python
@pytest.mark.e2e
def test_file_upload_functionality(chrome_driver, streamlit_base_url, test_data):
    """Test file upload functionality with test images."""
    chrome_driver.get(streamlit_base_url)

    sample_images = test_data["sample_images"]
    crack_images = [img for img in sample_images if img["type"] == "crack"]

    # Test with crack image upload
    # Implementation would depend on Streamlit file upload component
```

### Configuration Validation Test

```python
@pytest.mark.e2e
def test_browser_configuration(webdriver, e2e_config):
    """Test that browser is configured correctly."""
    expected_width, expected_height = e2e_config.window_size
    actual_size = webdriver.get_window_size()

    assert abs(actual_size["width"] - expected_width) <= 50
    assert abs(actual_size["height"] - expected_height) <= 100
```

## Pytest Markers

The framework registers several custom markers:

- `@pytest.mark.e2e`: End-to-end integration test
- `@pytest.mark.chrome`: Chrome-specific test
- `@pytest.mark.firefox`: Firefox-specific test
- `@pytest.mark.edge`: Edge-specific test
- `@pytest.mark.cross_browser`: Cross-browser execution
- `@pytest.mark.slow`: Slow-running test (extended timeout)

## Running Tests

### All E2E Tests

```bash
pytest tests/e2e/ -m e2e
```

### Browser-Specific Tests

```bash
pytest tests/e2e/ -m chrome
pytest tests/e2e/ -m firefox
pytest tests/e2e/ -m edge
```

### Cross-Browser Tests Only

```bash
pytest tests/e2e/ -m cross_browser
```

### Exclude Slow Tests

```bash
pytest tests/e2e/ -m "e2e and not slow"
```

## Integration with Existing Infrastructure

### HybridDriverManager Integration

The fixtures leverage the complete WebDriver management system from subtask 14.1:

- Docker Grid prioritization with WebDriverManager fallback
- Automatic driver lifecycle management
- Robust error handling and retry logic
- Cross-platform compatibility

### Docker Infrastructure Integration

Seamless integration with existing Docker testing infrastructure from Task 13:

- Uses existing Selenium Grid setup when available
- Falls back to local WebDriverManager when Docker is unavailable
- Supports both local development and CI/CD environments

### Project Standards Compliance

- **Type Safety**: Complete Python 3.12+ type annotations
- **Documentation**: Comprehensive docstrings following project standards
- **Error Handling**: Robust exception handling with graceful degradation
- **Resource Management**: Automatic cleanup prevents resource leaks

## Configuration Options

### Environment Variables

The fixtures respect standard WebDriver environment variables:

- `SELENIUM_HUB_HOST`: Selenium Grid host (default: localhost)
- `SELENIUM_HUB_PORT`: Selenium Grid port (default: 4444)
- `DOCKER_CONTAINER`: Docker environment detection

### Custom Configuration

```python
# Override default configuration
custom_config = DriverConfig(
    browser="firefox",
    headless=False,  # Run with visible browser
    window_size=(1366, 768),
    implicit_wait=15.0
)

with driver_session("firefox", config=custom_config) as driver:
    # Use custom configuration
    pass
```

## Troubleshooting

### Common Issues

1. **WebDriver Not Found**
   - Ensure HybridDriverManager is properly configured
   - Check Docker Grid availability or install WebDriverManager

2. **Test Artifacts Not Cleaned**
   - Verify `cleanup_test_artifacts` fixture is working
   - Check permissions on test artifacts directory

3. **Screenshot on Failure Not Working**
   - Ensure `screenshot_on_failure` fixture is included in test
   - Check that WebDriver is still active at failure time

4. **Cross-Browser Tests Failing**
   - Verify all target browsers are installed and accessible
   - Check browser version compatibility with WebDriver

### Debug Mode

Run with verbose output to see fixture execution:

```bash
pytest tests/e2e/ -v -s --capture=no
```

## Implementation Details

### Total Implementation

- **conftest.py**: 435 lines of comprehensive fixture definitions
- **Unit tests**: 285 lines of fixture validation tests
- **Usage examples**: 250+ lines of practical usage demonstrations
- **Documentation**: Complete usage guide and troubleshooting

### Quality Assurance

- Type-safe implementation with Python 3.12+ annotations
- Comprehensive error handling and resource management
- Integration with existing project infrastructure
- Following established pytest and project patterns

### Ready for Next Subtask

The fixture system is fully prepared for **subtask 14.3 - Base Test Class Implementation**, providing:

- Robust browser management foundation
- Standardized test data and configuration
- Proven resource cleanup mechanisms
- Cross-browser testing capabilities

---

**Subtask 14.2 Status**: ✅ **COMPLETED**
**Implementation**: Test Fixture Creation and Configuration
**Integration**: HybridDriverManager (14.1) + Docker Infrastructure (Task 13)
**Next**: Ready for subtask 14.3 - Base Test Class Implementation
