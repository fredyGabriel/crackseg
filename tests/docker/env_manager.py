"""Environment variable management system for CrackSeg Docker testing
infrastructure.

This module provides comprehensive environment variable handling for different
environments (local, staging, production) and test configurations with
validation, default value management, and security features.

Designed for Subtask 13.6 - Configure Environment Variable Management.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environment types."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class EnvironmentConfig:
    """Configuration container for environment-specific settings.

    This class encapsulates all environment variables with type safety,
    validation, and default values appropriate for crack segmentation testing.
    """

    # Environment Identification
    node_env: str = "local"
    crackseg_env: str = "local"
    project_name: str = "crackseg"

    # Application Configuration
    streamlit_server_headless: bool = False
    streamlit_server_port: int = 8501
    streamlit_server_address: str = "localhost"
    streamlit_browser_stats: bool = False

    # Development Features
    debug: bool = True
    log_level: str = "INFO"
    development_mode: bool = True
    hot_reload_enabled: bool = True

    # Testing Configuration
    test_browser: str = "chrome"
    test_parallel_workers: str | int = "auto"
    test_timeout: int = 300
    test_headless: bool = True
    test_debug: bool = False
    coverage_enabled: bool = True
    html_report_enabled: bool = True

    # Service Endpoints
    selenium_hub_host: str = "localhost"
    selenium_hub_port: int = 4444
    streamlit_host: str = "localhost"
    streamlit_port: int = 8501

    # Paths Configuration
    project_root: str = "/app"
    test_results_path: str = "./test-results"
    test_data_path: str = "./test-data"
    test_artifacts_path: str = "./test-artifacts"
    selenium_videos_path: str = "./selenium-videos"

    # ML/Training Configuration
    pytorch_cuda_alloc_conf: str = "max_split_size_mb:512"
    cuda_visible_devices: str = "0"
    model_cache_dir: str = "./cache/models"
    dataset_cache_dir: str = "./cache/datasets"

    # Performance Tuning
    pytest_opts: str = "--verbose --tb=short --strict-markers"
    max_browser_instances: int = 2
    browser_window_size: str = "1920,1080"
    selenium_implicit_wait: int = 10
    selenium_page_load_timeout: int = 30

    # Security (sensitive values handled separately)
    secrets: dict[str, str] = field(default_factory=dict)

    # Feature Flags
    feature_flags: dict[str, bool] = field(
        default_factory=lambda: {
            "FEATURE_ADVANCED_METRICS": True,
            "FEATURE_TENSORBOARD": True,
            "FEATURE_MODEL_COMPARISON": True,
            "FEATURE_EXPERIMENT_TRACKING": True,
        }
    )

    # Resource Constraints
    memory_limit: str = "4g"
    cpu_limit: str = "2"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration values and ensure consistency."""
        # Validate ports
        if not (1024 <= self.streamlit_server_port <= 65535):
            raise ValueError(
                f"Invalid Streamlit port: {self.streamlit_server_port}"
            )
        if not (1024 <= self.selenium_hub_port <= 65535):
            raise ValueError(
                f"Invalid Selenium hub port: {self.selenium_hub_port}"
            )

        # Validate timeout values
        if self.test_timeout <= 0:
            raise ValueError(
                f"Test timeout must be positive: {self.test_timeout}"
            )
        if self.selenium_implicit_wait < 0:
            raise ValueError(
                f"Selenium wait must be non-negative: "
                f"{self.selenium_implicit_wait}"
            )

        # Validate browser configuration
        valid_browsers = {"chrome", "firefox", "edge", "safari"}
        browsers = {b.strip().lower() for b in self.test_browser.split(",")}
        if not browsers.issubset(valid_browsers):
            invalid = browsers - valid_browsers
            raise ValueError(f"Invalid browsers: {invalid}")

        # Validate log level (case insensitive)
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")

        # Validate paths (basic checks)
        if not self.project_root:
            raise ValueError("Project root cannot be empty")


class EnvironmentManager:
    """Manages environment variables for different deployment environments.

    Provides methods to load, validate, and apply environment configurations
    with proper error handling and logging.
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize environment manager.

        Args:
            base_path: Base path for configuration files. Defaults to current
            directory.
        """
        self.base_path = base_path or Path.cwd()
        self.docker_path = self.base_path / "tests" / "docker"
        self._current_config: EnvironmentConfig | None = None

    def detect_environment(self) -> Environment:
        """Detect current environment from environment variables.

        Returns:
            Detected environment type.
        """
        env_var = os.getenv(
            "CRACKSEG_ENV", os.getenv("NODE_ENV", "local")
        ).lower()

        try:
            return Environment(env_var)
        except ValueError:
            logger.warning(
                f"Unknown environment '{env_var}', defaulting to 'local'"
            )
            return Environment.LOCAL

    def load_env_file(self, env_file: Path) -> dict[str, str]:
        """Load environment variables from .env file.

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

    def create_config_from_env(
        self, environment: Environment
    ) -> EnvironmentConfig:
        """Create configuration from environment variables and files.

        Args:
            environment: Target environment type.

        Returns:
            Validated environment configuration.
        """
        # Start with current environment variables
        env_vars = dict(os.environ)

        # Load environment-specific file if it exists
        env_file_name = f".env.{environment.value}"
        env_file_path = self.docker_path / env_file_name

        if env_file_path.exists():
            file_vars = self.load_env_file(env_file_path)
            env_vars.update(file_vars)
        else:
            # Try template file
            template_path = (
                self.docker_path / f"env.{environment.value}.template"
            )
            if template_path.exists():
                logger.info(f"Using template file: {template_path}")
                file_vars = self.load_env_file(template_path)
                env_vars.update(file_vars)
            else:
                logger.warning(
                    f"No environment file found for {environment.value}"
                )

        # Convert environment variables to configuration
        config = self._env_vars_to_config(env_vars, environment)
        return config

    def _env_vars_to_config(
        self, env_vars: dict[str, str], environment: Environment
    ) -> EnvironmentConfig:
        """Convert environment variables to EnvironmentConfig object.

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
                    f"Invalid integer value for {key}, using default: "
                    f"{default}"
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
        is_local = environment == Environment.LOCAL
        is_production = environment == Environment.PRODUCTION

        # Create configuration object with environment-specific defaults
        config = EnvironmentConfig(
            # Environment identification
            node_env=env_vars.get("NODE_ENV", environment.value),
            crackseg_env=env_vars.get("CRACKSEG_ENV", environment.value),
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
            log_level=env_vars.get(
                "LOG_LEVEL", "DEBUG" if is_local else "INFO"
            ),
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
            project_root=env_vars.get(
                "PROJECT_ROOT", "." if is_local else "/app"
            ),
            test_results_path=env_vars.get(
                "TEST_RESULTS_PATH", "./test-results"
            ),
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
            memory_limit=env_vars.get(
                "MEMORY_LIMIT", "2g" if is_local else "4g"
            ),
            cpu_limit=env_vars.get("CPU_LIMIT", "1" if is_local else "2"),
        )

        return config

    def apply_configuration(self, config: EnvironmentConfig) -> None:
        """Apply configuration to current environment.

        Args:
            config: Configuration to apply.
        """
        # Set environment variables
        os.environ["NODE_ENV"] = config.node_env
        os.environ["CRACKSEG_ENV"] = config.crackseg_env
        os.environ["PROJECT_NAME"] = config.project_name

        # Application settings
        os.environ["STREAMLIT_SERVER_HEADLESS"] = str(
            config.streamlit_server_headless
        ).lower()
        os.environ["STREAMLIT_SERVER_PORT"] = str(config.streamlit_server_port)
        os.environ["STREAMLIT_SERVER_ADDRESS"] = (
            config.streamlit_server_address
        )

        # Development settings
        os.environ["DEBUG"] = str(config.debug).lower()
        os.environ["LOG_LEVEL"] = config.log_level

        # Testing settings
        os.environ["TEST_BROWSER"] = config.test_browser
        os.environ["TEST_TIMEOUT"] = str(config.test_timeout)
        os.environ["TEST_HEADLESS"] = str(config.test_headless).lower()

        # Service endpoints
        os.environ["SELENIUM_HUB_HOST"] = config.selenium_hub_host
        os.environ["SELENIUM_HUB_PORT"] = str(config.selenium_hub_port)

        # Paths
        os.environ["PROJECT_ROOT"] = config.project_root
        os.environ["TEST_RESULTS_PATH"] = config.test_results_path
        os.environ["TEST_DATA_PATH"] = config.test_data_path

        # ML/Training
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.pytorch_cuda_alloc_conf
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

        # Apply feature flags
        for flag, value in config.feature_flags.items():
            os.environ[flag] = str(value).lower()

        # Note: Secrets should be handled separately for security
        logger.info(
            f"Applied configuration for environment: {config.crackseg_env}"
        )
        self._current_config = config

    def get_current_config(self) -> EnvironmentConfig | None:
        """Get current configuration."""
        return self._current_config

    def export_to_docker_compose(
        self, config: EnvironmentConfig
    ) -> dict[str, str]:
        """Export configuration for Docker Compose environment section.

        Args:
            config: Configuration to export.

        Returns:
            Dictionary suitable for Docker Compose environment section.
        """
        env_dict = {
            "NODE_ENV": config.node_env,
            "CRACKSEG_ENV": config.crackseg_env,
            "PROJECT_NAME": config.project_name,
            "STREAMLIT_SERVER_HEADLESS": str(
                config.streamlit_server_headless
            ).lower(),
            "STREAMLIT_SERVER_PORT": str(config.streamlit_server_port),
            "STREAMLIT_SERVER_ADDRESS": config.streamlit_server_address,
            "DEBUG": str(config.debug).lower(),
            "LOG_LEVEL": config.log_level,
            "TEST_BROWSER": config.test_browser,
            "TEST_TIMEOUT": str(config.test_timeout),
            "TEST_HEADLESS": str(config.test_headless).lower(),
            "SELENIUM_HUB_HOST": config.selenium_hub_host,
            "SELENIUM_HUB_PORT": str(config.selenium_hub_port),
            "PROJECT_ROOT": config.project_root,
            "TEST_RESULTS_PATH": config.test_results_path,
            "PYTORCH_CUDA_ALLOC_CONF": config.pytorch_cuda_alloc_conf,
            "CUDA_VISIBLE_DEVICES": config.cuda_visible_devices,
        }

        # Add feature flags
        env_dict.update(
            {k: str(v).lower() for k, v in config.feature_flags.items()}
        )

        return env_dict

    def save_config_to_file(
        self, config: EnvironmentConfig, file_path: Path
    ) -> None:
        """Save configuration to JSON file.

        Args:
            config: Configuration to save.
            file_path: Output file path.
        """
        config_dict = {
            "environment": config.crackseg_env,
            "node_env": config.node_env,
            "project_name": config.project_name,
            "streamlit": {
                "server_headless": config.streamlit_server_headless,
                "server_port": config.streamlit_server_port,
                "server_address": config.streamlit_server_address,
            },
            "testing": {
                "browser": config.test_browser,
                "timeout": config.test_timeout,
                "headless": config.test_headless,
                "debug": config.test_debug,
            },
            "services": {
                "selenium_hub_host": config.selenium_hub_host,
                "selenium_hub_port": config.selenium_hub_port,
            },
            "paths": {
                "project_root": config.project_root,
                "test_results": config.test_results_path,
                "test_data": config.test_data_path,
            },
            "feature_flags": config.feature_flags,
            "resources": {
                "memory_limit": config.memory_limit,
                "cpu_limit": config.cpu_limit,
            },
        }

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to: {file_path}")


def main() -> None:
    """Main function for environment manager CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CrackSeg Environment Manager"
    )
    parser.add_argument(
        "--env",
        choices=[e.value for e in Environment],
        default="local",
        help="Environment to configure",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate configuration only"
    )
    parser.add_argument("--export", help="Export to file path")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply configuration to current environment",
    )

    args = parser.parse_args()

    manager = EnvironmentManager()
    environment = Environment(args.env)

    try:
        config = manager.create_config_from_env(environment)

        if args.validate:
            print(f"✅ Configuration for {environment.value} is valid")

        if args.export:
            manager.save_config_to_file(config, Path(args.export))

        if args.apply:
            manager.apply_configuration(config)
            print(f"✅ Applied configuration for {environment.value}")

        if not any([args.validate, args.export, args.apply]):
            print(f"Configuration for {environment.value}:")
            print(f"  Environment: {config.crackseg_env}")
            print(f"  Debug: {config.debug}")
            print(f"  Test browser: {config.test_browser}")
            print(
                "  Selenium hub: "
                f"{config.selenium_hub_host}:{config.selenium_hub_port}"
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
