"""Environment utility functions for configuration management."""

import logging
from pathlib import Path

from .env_config import EnvironmentConfig

logger = logging.getLogger(__name__)


def load_env_file(env_file: Path) -> dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to environment file.

    Returns:
        Dictionary of environment variables.

    Raises:
        FileNotFoundError: If environment file doesn't exist.
        ValueError: If file format is invalid.
    """
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    env_vars = {}

    try:
        with open(env_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Parse KEY=VALUE format
                if "=" not in line:
                    logger.warning(
                        f"Skipping invalid line {line_num} in {env_file}: "
                        f"{line}"
                    )
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove inline comments (everything after # symbol)
                if "#" in value:
                    value = value.split("#")[0].strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                env_vars[key] = value

    except Exception as e:
        raise ValueError(
            f"Error reading environment file {env_file}: {e}"
        ) from e

    logger.info(f"Loaded {len(env_vars)} variables from {env_file}")
    return env_vars


def env_vars_to_config(
    env_vars: dict[str, str], environment: str
) -> EnvironmentConfig:
    """
    Convert environment variables to EnvironmentConfig object.

    Args:
        env_vars: Dictionary of environment variables.
        environment: Target environment type.

    Returns:
        Configured EnvironmentConfig object.
    """

    # Helper function to get boolean values
    def get_bool(key: str, default: bool) -> bool:
        value = env_vars.get(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    # Helper function to get integer values
    def get_int(key: str, default: int) -> int:
        try:
            return int(env_vars.get(key, str(default)))
        except ValueError:
            logger.warning(
                f"Invalid integer value for {key}, using default: {default}"
            )
            return default

    # Extract secrets separately
    secrets = {}
    secret_keys = [
        "API_KEY",
        "SECRET_KEY",
        "JWT_SECRET",
        "DATABASE_PASSWORD",
    ]
    for key in secret_keys:
        if key in env_vars:
            secrets[key] = env_vars[key]

    # Extract feature flags
    feature_flags = {}
    for key, _value in env_vars.items():
        if key.startswith("FEATURE_"):
            feature_flags[key] = get_bool(key, True)

    # Environment-specific defaults
    is_local = environment == "local"
    is_production = environment == "production"

    # Create configuration object with environment-specific defaults
    config = EnvironmentConfig(
        # Environment identification
        node_env=env_vars.get("NODE_ENV", environment),
        crackseg_env=env_vars.get("CRACKSEG_ENV", environment),
        project_name=env_vars.get("PROJECT_NAME", "crackseg"),
        # Application configuration (environment-specific defaults)
        streamlit_server_headless=get_bool(
            "STREAMLIT_SERVER_HEADLESS", not is_local
        ),
        streamlit_server_port=get_int("STREAMLIT_SERVER_PORT", 8501),
        streamlit_server_address=env_vars.get(
            "STREAMLIT_SERVER_ADDRESS",
            "localhost" if is_local else "0.0.0.0",
        ),
        streamlit_browser_stats=get_bool(
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS", False
        ),
        # Development features (local development defaults)
        debug=get_bool("DEBUG", is_local),
        log_level=env_vars.get("LOG_LEVEL", "DEBUG" if is_local else "INFO"),
        development_mode=get_bool("DEVELOPMENT_MODE", is_local),
        hot_reload_enabled=get_bool("HOT_RELOAD_ENABLED", is_local),
        # Testing configuration (environment-specific defaults)
        test_browser=env_vars.get("TEST_BROWSER", "chrome"),
        test_parallel_workers=env_vars.get(
            "TEST_PARALLEL_WORKERS", "1" if is_local else "auto"
        ),
        test_timeout=get_int("TEST_TIMEOUT", 120 if is_local else 300),
        test_headless=get_bool("TEST_HEADLESS", not is_local),
        test_debug=get_bool("TEST_DEBUG", is_local),
        coverage_enabled=get_bool("COVERAGE_ENABLED", True),
        html_report_enabled=get_bool("HTML_REPORT_ENABLED", True),
        # Service endpoints (containerized vs local defaults)
        selenium_hub_host=env_vars.get(
            "SELENIUM_HUB_HOST",
            "localhost" if is_local else "selenium-hub",
        ),
        selenium_hub_port=get_int("SELENIUM_HUB_PORT", 4444),
        streamlit_host=env_vars.get(
            "STREAMLIT_HOST", "localhost" if is_local else "streamlit-app"
        ),
        streamlit_port=get_int("STREAMLIT_PORT", 8501),
        # Paths (environment-specific defaults)
        project_root=env_vars.get("PROJECT_ROOT", "." if is_local else "/app"),
        test_results_path=env_vars.get("TEST_RESULTS_PATH", "./test-results"),
        test_data_path=env_vars.get("TEST_DATA_PATH", "./test-data"),
        test_artifacts_path=env_vars.get(
            "TEST_ARTIFACTS_PATH", "./test-artifacts"
        ),
        selenium_videos_path=env_vars.get(
            "SELENIUM_VIDEOS_PATH", "./selenium-videos"
        ),
        # ML/Training (resource-specific defaults)
        pytorch_cuda_alloc_conf=env_vars.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            (
                "max_split_size_mb:1024"
                if is_production
                else "max_split_size_mb:512"
            ),
        ),
        cuda_visible_devices=env_vars.get("CUDA_VISIBLE_DEVICES", "0"),
        model_cache_dir=env_vars.get("MODEL_CACHE_DIR", "./cache/models"),
        dataset_cache_dir=env_vars.get(
            "DATASET_CACHE_DIR", "./cache/datasets"
        ),
        # Performance
        pytest_opts=env_vars.get(
            "PYTEST_OPTS",
            (
                "--verbose --tb=long --capture=no"
                if is_local
                else "--verbose --tb=short --strict-markers"
            ),
        ),
        max_browser_instances=get_int(
            "MAX_BROWSER_INSTANCES", 1 if is_local else 2
        ),
        browser_window_size=env_vars.get(
            "BROWSER_WINDOW_SIZE", "1280,720" if is_local else "1920,1080"
        ),
        selenium_implicit_wait=get_int(
            "SELENIUM_IMPLICIT_WAIT", 5 if is_local else 10
        ),
        selenium_page_load_timeout=get_int(
            "SELENIUM_PAGE_LOAD_TIMEOUT", 15 if is_local else 30
        ),
        # Collections
        secrets=secrets,
        feature_flags=feature_flags,
        # Resources (environment-specific defaults)
        memory_limit=env_vars.get("MEMORY_LIMIT", "2g" if is_local else "4g"),
        cpu_limit=env_vars.get("CPU_LIMIT", "1" if is_local else "2"),
    )

    return config
