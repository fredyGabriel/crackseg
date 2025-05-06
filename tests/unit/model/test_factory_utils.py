"""
Unit tests for factory utilities module.

These tests verify the functionality of helper functions in factory_utils.py
for configuration validation, transformation, extraction of parameters, etc.
"""

import pytest
from unittest.mock import patch, MagicMock
import logging

from src.model.factory_utils import (
    ConfigurationError,
    validate_config,
    validate_component_types,
    check_parameter_types,
    hydra_to_dict,
    extract_runtime_params,
    merge_configs,
    filter_config,
    log_component_creation,
    log_configuration_error
)


class TestValidationFunctions:
    """Test configuration validation functions."""

    def test_validate_config(self):
        """Test validate_config with valid and invalid configurations."""
        # Valid config
        config = {"type": "TestType", "param1": 10, "param2": "value"}
        required_keys = ["type", "param1"]

        # Should not raise exception
        validate_config(config, required_keys, "test_component")

        # Invalid config - missing required key
        invalid_config = {"param2": "value"}

        with pytest.raises(ConfigurationError) as excinfo:
            validate_config(invalid_config, required_keys, "test_component")

        assert "Missing required configuration" in str(excinfo.value)
        assert "type, param1" in str(excinfo.value)

    def test_validate_component_types(self):
        """Test validate_component_types function."""
        # Valid component types
        config = {
            "encoder": "ResNet",
            "bottleneck": "Identity"
        }

        type_map = {
            "encoder": ["ResNet", "Swin"],
            "bottleneck": ["Identity", "LSTM"]
        }

        # Should not raise exception
        validate_component_types(config, type_map)

        # Invalid component type
        invalid_config = {
            "encoder": "Unknown",
            "bottleneck": "Identity"
        }

        with pytest.raises(ConfigurationError) as excinfo:
            validate_component_types(invalid_config, type_map)

        assert "Invalid encoder type" in str(excinfo.value)
        assert "ResNet, Swin" in str(excinfo.value)

    def test_check_parameter_types(self):
        """Test check_parameter_types function."""
        # Valid parameter types
        params = {
            "int_param": 10,
            "str_param": "text",
            "list_param": [1, 2, 3]
        }

        type_specs = {
            "int_param": int,
            "str_param": str,
            "list_param": list
        }

        # Should not raise exception
        check_parameter_types(params, type_specs)

        # Invalid parameter type
        invalid_params = {
            "int_param": "not an int",
            "str_param": "text"
        }

        with pytest.raises(ConfigurationError) as excinfo:
            check_parameter_types(invalid_params, type_specs)

        assert "Parameter 'int_param' has wrong type" in str(excinfo.value)
        assert "Expected int, got str" in str(excinfo.value)


class TestConfigTransformation:
    """Test configuration transformation functions."""

    def test_hydra_to_dict(self):
        """Test hydra_to_dict with various inputs."""
        # Test with regular dict
        regular_dict = {"param1": 10, "param2": "value"}
        result = hydra_to_dict(regular_dict)
        assert result == regular_dict
        assert result is not regular_dict  # Should be a copy

        # Test with DictConfig
        mock_dictconfig = MagicMock()
        mock_dictconfig.__class__.__name__ = "DictConfig"

        with patch('src.model.factory_utils.OmegaConf') as mock_omegaconf:
            mock_omegaconf.to_container.return_value = {"converted": True}
            result = hydra_to_dict(mock_dictconfig)

            # Verify OmegaConf.to_container was called
            mock_omegaconf.to_container.assert_called_once_with(
                mock_dictconfig, resolve=True
            )
            assert result == {"converted": True}

    def test_extract_runtime_params(self):
        """Test extract_runtime_params function."""
        # Create a component with attributes
        class TestComponent:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42
                self.hidden = "secret"

        component = TestComponent()

        # Map source attributes to target params
        param_mappings = {
            "attr1": "target1",
            "attr2": "target2",
            "missing": "target3"  # Attribute that doesn't exist
        }

        result = extract_runtime_params(component, param_mappings)

        # Check result
        assert "target1" in result
        assert result["target1"] == "value1"
        assert "target2" in result
        assert result["target2"] == 42
        assert "target3" not in result  # Should not be in result
        # Hidden attribute was not in mapping
        assert "hidden" not in result

    def test_merge_configs(self):
        """Test merge_configs function."""
        # Base config
        base_config = {
            "type": "TestType",
            "param1": 10,
            "param2": "original"
        }

        # Override config
        override_config = {
            "param2": "override",
            "param3": True
        }

        result = merge_configs(base_config, override_config)

        # Check result
        assert result["type"] == "TestType"  # From base
        assert result["param1"] == 10  # From base
        assert result["param2"] == "override"  # From override
        assert result["param3"] is True  # From override

        # Original configs should not be modified
        assert base_config["param2"] == "original"
        assert "param3" not in base_config

    def test_filter_config(self):
        """Test filter_config function."""
        # Test config
        config = {
            "type": "TestType",
            "param1": 10,
            "param2": "value",
            "param3": True,
            "param4": [1, 2, 3]
        }

        # Test with include_keys
        include_keys = {"type", "param1", "param3"}
        result = filter_config(config, include_keys=include_keys)

        assert "type" in result
        assert "param1" in result
        assert "param3" in result
        assert "param2" not in result
        assert "param4" not in result

        # Test with exclude_keys
        exclude_keys = {"param2", "param4"}
        result = filter_config(config, exclude_keys=exclude_keys)

        assert "type" in result
        assert "param1" in result
        assert "param3" in result
        assert "param2" not in result
        assert "param4" not in result

        # Test with both include_keys and exclude_keys
        result = filter_config(
            config, include_keys={"type", "param1", "param2"},
            exclude_keys={"param2"}
        )

        assert "type" in result
        assert "param1" in result
        assert "param2" not in result  # Excluded takes precedence
        assert "param3" not in result  # Not included
        assert "param4" not in result  # Not included


class TestLoggingHelpers:
    """Test logging helper functions."""

    def test_log_component_creation(self, caplog):
        """Test log_component_creation function."""
        # Capture log output
        with caplog.at_level(logging.INFO):
            log_component_creation("Encoder", "ResNet50")

        # Check log output
        assert "Instantiated Encoder: ResNet50" in caplog.text

        # Test with different log level
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            log_component_creation("Decoder", "UNet", level=logging.DEBUG)

        # Check log output
        assert "Instantiated Decoder: UNet" in caplog.text

    def test_log_configuration_error(self, caplog):
        """Test log_configuration_error function."""
        # Capture log output
        with caplog.at_level(logging.ERROR):
            log_configuration_error(
                "Validation", "Missing required parameter",
                {"type": "TestType", "param1": 10}
            )

        # Check log output
        assert "Configuration error (Validation)" in caplog.text
        assert "Missing required parameter" in caplog.text
        assert "TestType" in caplog.text

        # Test without config
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            log_configuration_error(
                "Type", "Unknown component type",
                level=logging.WARNING
            )

        # Check log output
        assert "Configuration error (Type)" in caplog.text
        assert "Unknown component type" in caplog.text

        # Test with large config (truncation)
        large_config = {f"param{i}": i for i in range(20)}
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            log_configuration_error(
                "Size", "Too many parameters", large_config
            )

        # Check log output
        assert "Configuration error (Size)" in caplog.text
        assert "Too many parameters" in caplog.text
        assert "..." in caplog.text  # Truncation indicator
