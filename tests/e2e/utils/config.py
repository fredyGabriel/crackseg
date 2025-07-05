"""Configuration utilities for E2E testing.

This module provides configuration management utilities specifically designed
for E2E testing of the CrackSeg Streamlit application. Includes functions for
loading test configurations, managing browser settings, and handling
environment-specific configurations.

Key features:
- Test configuration loading and validation
- Browser configuration management
- Environment-specific test settings
- CrackSeg application configuration handling
- Configuration file generation for tests

Examples:
    >>> config = load_test_config("selenium_config.yaml")
    >>> browser_config = get_browser_config("chrome", headless=True)
    >>> streamlit_config = get_streamlit_test_config()
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_test_config(
    config_path: str | Path,
    section: str | None = None,
    validate_required: list[str] | None = None,
) -> dict[str, Any]:
    """Load and validate test configuration from file.

    Supports both YAML and JSON configuration files with optional
    section extraction and required field validation.

    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)
        section: Optional section name to extract from config
        validate_required: List of required keys to validate

    Returns:
        Dictionary containing configuration data

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If required fields are missing or file format invalid

    Example:
        >>> config = load_test_config(
        ...     "tests/config/selenium.yaml",
        ...     section="browser_settings",
        ...     validate_required=["driver_path", "timeout"]
        ... )
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config format: {config_path.suffix}"
                )
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid configuration file format: {e}") from e

    # Extract specific section if requested
    if section:
        if section not in config_data:
            raise ValueError(f"Section '{section}' not found in config")
        config_data = config_data[section]

    # Validate required fields
    if validate_required:
        missing_fields = [
            field for field in validate_required if field not in config_data
        ]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    logger.debug(f"Loaded configuration from {config_path}")
    return config_data


def get_browser_config(
    browser: str = "chrome",
    headless: bool = True,
    window_size: tuple[int, int] = (1920, 1080),
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate browser configuration for Selenium WebDriver.

    Creates standardized browser configuration dictionary suitable
    for WebDriver initialization with CrackSeg testing requirements.

    Args:
        browser: Browser type ('chrome', 'firefox', 'edge')
        headless: Whether to run browser in headless mode
        window_size: Browser window dimensions (width, height)
        **kwargs: Additional browser-specific options

    Returns:
        Dictionary with browser configuration options

    Example:
        >>> config = get_browser_config("chrome", headless=False,
        ...                            window_size=(1440, 900))
    """
    base_config = {
        "browser": browser.lower(),
        "headless": headless,
        "window_size": window_size,
        "implicit_wait": kwargs.get("implicit_wait", 10),
        "page_load_timeout": kwargs.get("page_load_timeout", 30),
    }

    # Browser-specific configurations
    if browser.lower() == "chrome":
        chrome_options = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-web-security",
            "--allow-running-insecure-content",
        ]
        if headless:
            chrome_options.append("--headless")

        base_config.update(
            {
                "chrome_options": chrome_options,
                "download_dir": kwargs.get("download_dir", None),
                "prefs": kwargs.get("prefs", {}),
            }
        )

    elif browser.lower() == "firefox":
        firefox_options = []
        if headless:
            firefox_options.append("--headless")

        base_config.update(
            {
                "firefox_options": firefox_options,
                "profile_preferences": kwargs.get("profile_preferences", {}),
            }
        )

    elif browser.lower() == "edge":
        edge_options = ["--no-sandbox", "--disable-dev-shm-usage"]
        if headless:
            edge_options.append("--headless")

        base_config.update(
            {
                "edge_options": edge_options,
            }
        )

    # Add any additional custom options
    base_config.update(
        {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "implicit_wait",
                "page_load_timeout",
                "download_dir",
                "prefs",
                "profile_preferences",
            ]
        }
    )

    return base_config


def get_streamlit_test_config(
    port: int = 8501,
    host: str = "localhost",
    timeout: int = 30,
    debug: bool = False,
) -> dict[str, Any]:
    """Generate Streamlit application test configuration.

    Creates configuration dictionary for testing Streamlit applications
    with appropriate timeouts and debugging settings.

    Args:
        port: Streamlit application port
        host: Streamlit application host
        timeout: Default timeout for operations in seconds
        debug: Enable debug mode for detailed logging

    Returns:
        Dictionary with Streamlit test configuration

    Example:
        >>> config = get_streamlit_test_config(port=8502, debug=True)
    """
    return {
        "streamlit": {
            "host": host,
            "port": port,
            "base_url": f"http://{host}:{port}",
            "timeout": timeout,
            "debug": debug,
        },
        "test_settings": {
            "wait_for_app_start": timeout,
            "element_wait_timeout": 10,
            "page_load_timeout": 20,
            "screenshot_on_failure": True,
            "save_page_source": debug,
        },
        "crackseg_specific": {
            "expected_pages": [
                "Home",
                "Configuration",
                "Training",
                "Architecture",
            ],
            "file_upload_timeout": 60,
            "model_load_timeout": 120,
            "training_timeout": 300,
        },
    }


def get_test_environment_config() -> dict[str, Any]:
    """Get test environment-specific configuration.

    Reads environment variables and system settings to create
    appropriate test configuration for the current environment.

    Returns:
        Dictionary with environment-specific test settings

    Example:
        >>> env_config = get_test_environment_config()
        >>> is_ci = env_config["ci"]["enabled"]
    """
    # Detect CI environment
    ci_indicators = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
    ]
    is_ci = any(os.getenv(indicator) for indicator in ci_indicators)

    # Get system information
    is_windows = os.name == "nt"
    is_linux = os.name == "posix" and "linux" in sys.platform.lower()
    is_macos = os.name == "posix" and "darwin" in sys.platform.lower()

    return {
        "ci": {
            "enabled": is_ci,
            "provider": _detect_ci_provider(),
        },
        "system": {
            "windows": is_windows,
            "linux": is_linux,
            "macos": is_macos,
            "platform": sys.platform,
        },
        "paths": {
            "home": str(Path.home()),
            "tmp": str(Path.cwd() / "test-artifacts"),
            "downloads": (
                str(Path.home() / "Downloads")
                if not is_ci
                else str(Path.cwd() / "downloads")
            ),
        },
        "timeouts": {
            "default": 60 if is_ci else 30,
            "extended": 300 if is_ci else 120,
            "file_operations": 30,
        },
        "browser_defaults": {
            "headless": is_ci,
            "window_size": (1920, 1080) if not is_ci else (1280, 720),
        },
    }


def _detect_ci_provider() -> str | None:
    """Detect which CI provider is being used."""
    providers = {
        "GITHUB_ACTIONS": "github",
        "GITLAB_CI": "gitlab",
        "TRAVIS": "travis",
        "CIRCLECI": "circle",
        "APPVEYOR": "appveyor",
        "AZURE_PIPELINES": "azure",
    }

    for env_var, provider in providers.items():
        if os.getenv(env_var):
            return provider

    return None


def create_test_config_file(
    output_path: str | Path,
    browser_configs: list[str] | None = None,
    environment_configs: dict[str, Any] | None = None,
    streamlit_config: dict[str, Any] | None = None,
) -> Path:
    """Create a comprehensive test configuration file.

    Generates a complete test configuration file combining browser,
    environment, and Streamlit settings for E2E testing.

    Args:
        output_path: Path where to save the configuration file
        browser_configs: List of browser types to include
        environment_configs: Custom environment configuration overrides
        streamlit_config: Custom Streamlit configuration overrides

    Returns:
        Path to the created configuration file

    Example:
        >>> config_path = create_test_config_file(
        ...     "test_config.yaml",
        ...     browser_configs=["chrome", "firefox"],
        ...     environment_configs={"timeout": 60}
        ... )
    """
    output_path = Path(output_path)

    # Default browser configurations
    if browser_configs is None:
        browser_configs = ["chrome", "firefox"]

    # Build complete configuration
    config_data = {
        "browsers": {
            browser: get_browser_config(browser) for browser in browser_configs
        },
        "environment": get_test_environment_config(),
        "streamlit": get_streamlit_test_config(),
    }

    # Apply custom overrides
    if environment_configs:
        config_data["environment"].update(environment_configs)

    if streamlit_config:
        config_data["streamlit"].update(streamlit_config)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write configuration file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if output_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif output_path.suffix.lower() == ".json":
                json.dump(config_data, f, indent=2)
            else:
                # Default to YAML
                output_path = output_path.with_suffix(".yaml")
                with open(output_path, "w", encoding="utf-8") as yaml_f:
                    yaml.dump(
                        config_data, yaml_f, default_flow_style=False, indent=2
                    )
    except (OSError, yaml.YAMLError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to write configuration file: {e}") from e

    logger.info(f"Test configuration written to {output_path}")
    return output_path


def validate_crackseg_config(config_data: dict[str, Any]) -> bool:
    """Validate CrackSeg-specific configuration requirements.

    Checks that configuration contains all necessary settings
    for CrackSeg application testing.

    Args:
        config_data: Configuration dictionary to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> config = load_test_config("crackseg_test.yaml")
        >>> is_valid = validate_crackseg_config(config)
    """
    required_sections = ["browsers", "streamlit", "environment"]
    missing_sections = [
        section for section in required_sections if section not in config_data
    ]
    if missing_sections:
        raise ValueError(f"Missing required sections: {missing_sections}")

    # Validate Streamlit configuration
    streamlit_config = config_data.get("streamlit", {})
    required_streamlit = ["host", "port", "timeout"]
    missing_streamlit = [
        key for key in required_streamlit if key not in streamlit_config
    ]
    if missing_streamlit:
        raise ValueError(f"Missing Streamlit config: {missing_streamlit}")

    # Validate at least one browser configuration
    browser_config = config_data.get("browsers", {})
    if not browser_config:
        raise ValueError("At least one browser configuration is required")

    # Validate browser configurations
    for browser_name, browser_settings in browser_config.items():
        required_browser = ["browser", "headless", "window_size"]
        missing_browser = [
            key for key in required_browser if key not in browser_settings
        ]
        if missing_browser:
            raise ValueError(
                f"Missing {browser_name} config: {missing_browser}"
            )

    logger.debug("CrackSeg configuration validation passed")
    return True


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Performs deep merge of configuration dictionaries with later
    configurations taking precedence over earlier ones.

    Args:
        *configs: Variable number of configuration dictionaries

    Returns:
        Merged configuration dictionary

    Example:
        >>> base_config = {"timeout": 30, "browser": {"headless": True}}
        >>> override_config = {"timeout": 60}
        >>> merged = merge_configs(base_config, override_config)
    """

    def _deep_merge(
        base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Recursively merge dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    if not configs:
        return {}

    result = configs[0].copy()
    for config in configs[1:]:
        result = _deep_merge(result, config)

    return result
