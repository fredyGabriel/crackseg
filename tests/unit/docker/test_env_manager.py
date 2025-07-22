#!/usr/bin/env python3
"""
Unit tests for environment variable management system. Tests the
EnvironmentManager and EnvironmentConfig classes for proper
validation, configuration loading, and Docker integration. Designed
for Subtask 13.6 - Configure Environment Variable Management
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import directly from docker directory (now in extraPaths)
from env_manager import (
    Environment,
    EnvironmentConfig,
    EnvironmentManager,
)
from env_utils import load_env_file


class TestEnvironment:
    """Test Environment enum functionality."""

    def test_environment_values(self) -> None:
        """Test that Environment enum has expected values."""
        assert Environment.LOCAL.value == "local"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TEST.value == "test"

    def test_environment_enumeration(self) -> None:
        """Test that all expected environments are present."""
        expected_envs = {"local", "staging", "production", "test"}
        actual_envs = {env.value for env in Environment}
        assert actual_envs == expected_envs


class TestEnvironmentConfig:
    """Test EnvironmentConfig dataclass functionality."""

    def test_default_initialization(self) -> None:
        """Test EnvironmentConfig with default values."""
        config = EnvironmentConfig()

        assert config.node_env == "local"
        assert config.crackseg_env == "local"
        assert config.project_name == "crackseg"
        assert config.streamlit_server_port == 8501
        assert config.debug is True
        assert isinstance(config.secrets, dict)
        assert isinstance(config.feature_flags, dict)

    def test_custom_initialization(self) -> None:
        """Test EnvironmentConfig with custom values."""
        custom_secrets = {"API_KEY": "test_key"}
        custom_flags = {"FEATURE_TEST": True}

        config = EnvironmentConfig(
            node_env="test",
            crackseg_env="test",
            streamlit_server_port=9999,
            debug=False,
            secrets=custom_secrets,
            feature_flags=custom_flags,
        )

        assert config.node_env == "test"
        assert config.crackseg_env == "test"
        assert config.streamlit_server_port == 9999
        assert config.debug is False
        assert config.secrets == custom_secrets
        assert config.feature_flags == custom_flags

    def test_port_validation(self) -> None:
        """Test port number validation."""
        # Valid ports should work
        config = EnvironmentConfig(streamlit_server_port=8080)
        assert config.streamlit_server_port == 8080

        # Invalid ports should raise ValueError
        with pytest.raises(ValueError, match="Invalid Streamlit port"):
            EnvironmentConfig(streamlit_server_port=999)  # Too low

        with pytest.raises(ValueError, match="Invalid Streamlit port"):
            EnvironmentConfig(streamlit_server_port=70000)  # Too high

        with pytest.raises(ValueError, match="Invalid Selenium hub port"):
            EnvironmentConfig(selenium_hub_port=500)  # Too low

    def test_timeout_validation(self) -> None:
        """Test timeout value validation."""
        # Valid timeout should work
        config = EnvironmentConfig(test_timeout=120)
        assert config.test_timeout == 120

        # Invalid timeout should raise ValueError
        with pytest.raises(ValueError, match="Test timeout must be positive"):
            EnvironmentConfig(test_timeout=-10)

        with pytest.raises(ValueError, match="Test timeout must be positive"):
            EnvironmentConfig(test_timeout=0)

        # Negative selenium wait should fail
        with pytest.raises(
            ValueError, match="Selenium wait must be non-negative"
        ):
            EnvironmentConfig(selenium_implicit_wait=-5)

    def test_browser_validation(self) -> None:
        """Test browser configuration validation."""
        # Valid browsers should work
        config = EnvironmentConfig(test_browser="chrome")
        assert config.test_browser == "chrome"

        config = EnvironmentConfig(test_browser="chrome,firefox")
        assert config.test_browser == "chrome,firefox"

        # Invalid browser should raise ValueError
        with pytest.raises(ValueError, match="Invalid browsers"):
            EnvironmentConfig(test_browser="invalid_browser")

        with pytest.raises(ValueError, match="Invalid browsers"):
            EnvironmentConfig(test_browser="chrome,invalid_browser")

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        # Valid log levels should work
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = EnvironmentConfig(log_level=level)
            assert config.log_level == level

        # Case insensitive should work
        config = EnvironmentConfig(log_level="debug")
        assert (
            config.log_level == "debug"
        )  # Validation converts to uppercase for checking

        # Invalid log level should raise ValueError
        with pytest.raises(ValueError, match="Invalid log level"):
            EnvironmentConfig(log_level="INVALID")

    def test_project_root_validation(self) -> None:
        """Test project root validation."""
        # Valid project root should work
        config = EnvironmentConfig(project_root="/app")
        assert config.project_root == "/app"

        # Empty project root should raise ValueError
        with pytest.raises(ValueError, match="Project root cannot be empty"):
            EnvironmentConfig(project_root="")


class TestEnvironmentManager:
    """Test EnvironmentManager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.docker_dir = self.temp_dir / "docker"
        self.docker_dir.mkdir(parents=True, exist_ok=True)
        self.manager = EnvironmentManager(base_path=self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self) -> None:
        """Test EnvironmentManager initialization."""
        manager = EnvironmentManager()
        assert manager.base_path == Path.cwd()
        assert manager.docker_path == Path.cwd() / "docker"
        assert manager._current_config is None

        # Test with custom base path
        custom_manager = EnvironmentManager(self.temp_dir)
        assert custom_manager.base_path == self.temp_dir
        assert custom_manager.docker_path == self.temp_dir / "docker"

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_environment_default(self) -> None:
        """Test environment detection with default values."""
        environment = self.manager.detect_environment()
        assert environment == Environment.LOCAL

    @patch.dict(os.environ, {"CRACKSEG_ENV": "staging"}, clear=True)
    def test_detect_environment_crackseg_env(self) -> None:
        """Test environment detection with CRACKSEG_ENV."""
        environment = self.manager.detect_environment()
        assert environment == Environment.STAGING

    @patch.dict(os.environ, {"NODE_ENV": "production"}, clear=True)
    def test_detect_environment_node_env(self) -> None:
        """Test environment detection with NODE_ENV."""
        environment = self.manager.detect_environment()
        assert environment == Environment.PRODUCTION

    @patch.dict(
        os.environ, {"CRACKSEG_ENV": "test", "NODE_ENV": "local"}, clear=True
    )
    def test_detect_environment_precedence(self) -> None:
        """Test that CRACKSEG_ENV takes precedence over NODE_ENV."""
        environment = self.manager.detect_environment()
        assert environment == Environment.TEST

    @patch.dict(os.environ, {"CRACKSEG_ENV": "invalid"}, clear=True)
    def test_detect_environment_invalid(self) -> None:
        """Test environment detection with invalid value."""
        with patch("env_manager.logger.warning") as mock_warning:
            environment = self.manager.detect_environment()
            assert environment == Environment.LOCAL
            mock_warning.assert_called_once()

    def test_load_env_file_not_found(self) -> None:
        """Test loading non-existent environment file."""
        non_existent_file = self.docker_dir / "non_existent.env"

        with pytest.raises(FileNotFoundError):
            load_env_file(non_existent_file)

    def test_load_env_file_valid(self) -> None:
        """Test loading valid environment file."""
        env_file = self.docker_dir / "test.env"
        env_content = """
# Comment line
NODE_ENV=test
DEBUG=true
LOG_LEVEL=INFO

# Another comment
API_KEY="secret_key"
FEATURE_TEST='enabled'
EMPTY_VALUE=
"""
        env_file.write_text(env_content)

        env_vars = load_env_file(env_file)

        expected_vars = {
            "NODE_ENV": "test",
            "DEBUG": "true",
            "LOG_LEVEL": "INFO",
            "API_KEY": "secret_key",
            "FEATURE_TEST": "enabled",
            "EMPTY_VALUE": "",
        }

        assert env_vars == expected_vars

    def test_load_env_file_invalid_format(self) -> None:
        """Test loading environment file with invalid format."""
        env_file = self.docker_dir / "invalid.env"
        env_content = """
NODE_ENV=test
INVALID_LINE_WITHOUT_EQUALS
DEBUG=true
"""
        env_file.write_text(env_content)

        # Should still work, just skip invalid lines
        env_vars = load_env_file(env_file)

        expected_vars = {"NODE_ENV": "test", "DEBUG": "true"}

        assert env_vars == expected_vars

    @patch.dict(os.environ, {"TEST_VAR": "from_env"}, clear=True)
    def test_create_config_from_env_variables(self) -> None:
        """Test creating configuration from environment variables."""
        config = self.manager.create_config_from_env(Environment.LOCAL)

        assert config.node_env == "local"
        assert config.crackseg_env == "local"
        assert config.debug is True  # Default for local

    def test_create_config_from_template_file(self) -> None:
        """Test creating configuration from template file."""
        template_file = self.docker_dir / "env.test.template"
        template_content = """
NODE_ENV=test
CRACKSEG_ENV=test
DEBUG=false
TEST_BROWSER=firefox
"""
        template_file.write_text(template_content)

        config = self.manager.create_config_from_env(Environment.TEST)

        assert config.node_env == "test"
        assert config.crackseg_env == "test"
        assert config.debug is False
        assert config.test_browser == "firefox"

    def test_apply_configuration(self) -> None:
        """Test applying configuration to environment."""
        config = EnvironmentConfig(
            node_env="test",
            crackseg_env="test",
            debug=False,
            test_browser="chrome",
            streamlit_server_port=9000,
        )

        # Clear environment first
        test_vars = [
            "NODE_ENV",
            "CRACKSEG_ENV",
            "DEBUG",
            "TEST_BROWSER",
            "STREAMLIT_SERVER_PORT",
        ]
        for var in test_vars:
            os.environ.pop(var, None)

        self.manager.apply_configuration(config)

        assert os.environ["NODE_ENV"] == "test"
        assert os.environ["CRACKSEG_ENV"] == "test"
        assert os.environ["DEBUG"] == "false"
        assert os.environ["TEST_BROWSER"] == "chrome"
        assert os.environ["STREAMLIT_SERVER_PORT"] == "9000"
        assert self.manager.get_current_config() == config

    def test_export_to_docker_compose(self) -> None:
        """Test exporting configuration for Docker Compose."""
        config = EnvironmentConfig(
            node_env="test",
            crackseg_env="test",
            debug=False,
            streamlit_server_port=9000,
            feature_flags={"FEATURE_TEST": True},
        )

        docker_env = self.manager.export_to_docker_compose(config)

        expected_keys = {
            "NODE_ENV",
            "CRACKSEG_ENV",
            "PROJECT_NAME",
            "STREAMLIT_SERVER_HEADLESS",
            "STREAMLIT_SERVER_PORT",
            "DEBUG",
            "LOG_LEVEL",
            "TEST_BROWSER",
            "FEATURE_TEST",
        }

        assert set(docker_env.keys()).issuperset(expected_keys)
        assert docker_env["NODE_ENV"] == "test"
        assert docker_env["CRACKSEG_ENV"] == "test"
        assert docker_env["DEBUG"] == "false"
        assert docker_env["STREAMLIT_SERVER_PORT"] == "9000"
        assert docker_env["FEATURE_TEST"] == "true"

    def test_save_config_to_file(self) -> None:
        """Test saving configuration to JSON file."""
        config = EnvironmentConfig(
            node_env="test",
            crackseg_env="test",
            debug=False,
            feature_flags={"FEATURE_TEST": True},
        )

        output_file = self.temp_dir / "test_config.json"
        self.manager.save_config_to_file(config, output_file)

        assert output_file.exists()

        with open(output_file) as f:
            saved_config = json.load(f)

        assert saved_config["environment"] == "test"
        assert saved_config["node_env"] == "test"
        assert saved_config["testing"]["debug"] is False
        assert saved_config["feature_flags"]["FEATURE_TEST"] is True


class TestEnvironmentIntegration:
    """Integration tests for environment management system."""

    def setup_method(self) -> None:
        """Set up integration test fixtures."""
        # Clean up environment variables before each test to ensure isolation
        env_vars_to_clean = [
            "NODE_ENV",
            "CRACKSEG_ENV",
            "DEBUG",
            "LOG_LEVEL",
            "TEST_BROWSER",
            "STREAMLIT_SERVER_PORT",
            "PROJECT_NAME",
            "FEATURE_ADVANCED_METRICS",
            "FEATURE_TENSORBOARD",
            "STREAMLIT_SERVER_HEADLESS",
            "STREAMLIT_SERVER_ADDRESS",
            "SELENIUM_HUB_HOST",
            "PROJECT_ROOT",
            "TEST_TIMEOUT",
            "TEST_HEADLESS",
        ]
        for var in env_vars_to_clean:
            os.environ.pop(var, None)

        self.temp_dir = Path(tempfile.mkdtemp())
        self.docker_dir = self.temp_dir / "tests" / "docker"
        self.docker_dir.mkdir(parents=True, exist_ok=True)
        self.manager = EnvironmentManager(base_path=self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up integration test fixtures."""
        import shutil

        # Clean up environment variables that tests might have set
        env_vars_to_clean = [
            "NODE_ENV",
            "CRACKSEG_ENV",
            "DEBUG",
            "LOG_LEVEL",
            "TEST_BROWSER",
            "STREAMLIT_SERVER_PORT",
            "PROJECT_NAME",
            "FEATURE_ADVANCED_METRICS",
            "FEATURE_TENSORBOARD",
        ]
        for var in env_vars_to_clean:
            os.environ.pop(var, None)

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_environment_setup_workflow(self) -> None:
        """Test complete environment setup workflow."""
        # Create template file
        template_file = self.docker_dir / "env.test.template"
        template_content = """
NODE_ENV=test CRACKSEG_ENV=test DEBUG=false TEST_BROWSER=chrome
STREAMLIT_SERVER_PORT=8501 FEATURE_ADVANCED_METRICS=true
FEATURE_TENSORBOARD=false
"""
        template_file.write_text(template_content)

        # Load configuration
        config = self.manager.create_config_from_env(Environment.TEST)

        # Validate configuration
        assert config.node_env == "test"
        assert config.crackseg_env == "test"
        assert config.debug is False
        assert config.test_browser == "chrome"
        assert config.feature_flags["FEATURE_ADVANCED_METRICS"] is True
        assert config.feature_flags["FEATURE_TENSORBOARD"] is False

        # Apply configuration
        self.manager.apply_configuration(config)

        # Verify environment variables are set
        assert os.environ["NODE_ENV"] == "test"
        assert os.environ["CRACKSEG_ENV"] == "test"
        assert os.environ["DEBUG"] == "false"

        # Export for Docker Compose
        docker_env = self.manager.export_to_docker_compose(config)
        assert docker_env["NODE_ENV"] == "test"
        assert docker_env["FEATURE_ADVANCED_METRICS"] == "true"

        # Save to file
        output_file = self.temp_dir / "exported_config.json"
        self.manager.save_config_to_file(config, output_file)
        assert output_file.exists()

    def test_environment_specific_defaults(self) -> None:
        """Test that different environments have appropriate defaults."""
        # Clean up any existing templates to ensure pure environment-specific
        # defaults
        for template in self.docker_dir.glob("*.template"):
            template.unlink()

        # Local environment should have debug enabled
        local_config = self.manager.create_config_from_env(Environment.LOCAL)
        assert local_config.debug is True
        assert local_config.streamlit_server_headless is False
        assert local_config.test_headless is False

        # Production environment should have debug disabled
        prod_config = self.manager.create_config_from_env(
            Environment.PRODUCTION
        )
        assert prod_config.debug is False
        assert prod_config.streamlit_server_headless is True
        assert prod_config.test_headless is True

        # Test environment should have appropriate settings
        test_config = self.manager.create_config_from_env(Environment.TEST)
        assert test_config.debug is False
        assert test_config.streamlit_server_headless is True
        assert test_config.test_headless is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
