"""
Environment variable management system for CrackSeg Docker testing
infrastructure. This module provides comprehensive environment
variable handling for different environments (local, staging,
production) and test configurations with validation, default value
management, and security features. Designed for Subtask 13.6 -
Configure Environment Variable Management.
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path

from .env_config import EnvironmentConfig
from .env_utils import env_vars_to_config, load_env_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environment types."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class EnvironmentManager:
    """
    Manages environment variables for different deployment environments.
    Provides methods to load, validate, and apply environment
    configurations with proper error handling and logging.
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """
        Initialize environment manager. Args: base_path: Base path for
        configuration files. Defaults to current directory.
        """
        self.base_path = base_path or Path.cwd()
        self.docker_path = self.base_path / "tests" / "docker"
        self._current_config: EnvironmentConfig | None = None

    def detect_environment(self) -> Environment:
        """
        Detect current environment from environment variables. Returns:
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

    def create_config_from_env(
        self, environment: Environment
    ) -> EnvironmentConfig:
        """
        Create configuration from environment variables and files. Args:
        environment: Target environment type. Returns: Validated environment
        configuration.
        """
        # Start with current environment variables
        env_vars = dict(os.environ)

        # Load environment-specific file if it exists
        env_file_name = f".env.{environment.value}"
        env_file_path = self.docker_path / env_file_name

        if env_file_path.exists():
            file_vars = load_env_file(env_file_path)
            env_vars.update(file_vars)
        else:
            # Try template file
            template_path = (
                self.docker_path / f"env.{environment.value}.template"
            )
            if template_path.exists():
                logger.info(f"Using template file: {template_path}")
                file_vars = load_env_file(template_path)
                env_vars.update(file_vars)
            else:
                logger.warning(
                    f"No environment file found for {environment.value}"
                )

        # Convert environment variables to configuration
        config = env_vars_to_config(env_vars, environment.value)
        return config

    def apply_configuration(self, config: EnvironmentConfig) -> None:
        """
        Apply configuration to current environment. Args: config:
        Configuration to apply.
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
        """
        Save configuration to JSON file. Args: config: Configuration to save.
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
