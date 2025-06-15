"""Unit tests for the YAMLValidator engine."""

import pytest

from scripts.gui.utils.config.exceptions import ValidationError
from scripts.gui.utils.config.validation.yaml_engine import YAMLValidator


@pytest.fixture
def validator() -> YAMLValidator:
    """Fixture to provide a YAMLValidator instance."""
    return YAMLValidator()


class TestYAMLValidator:
    """Test suite for the YAMLValidator class."""

    def test_validate_syntax_valid(self, validator: YAMLValidator):
        """Test syntax validation with valid YAML content."""
        valid_yaml = """
        key: value
        list:
          - item1
          - item2
        """
        is_valid, error = validator.validate_syntax(valid_yaml)
        assert is_valid is True
        assert error is None

    def test_validate_syntax_invalid(self, validator: YAMLValidator):
        """Test syntax validation with invalid YAML content."""
        invalid_yaml = "key: value: [invalid"
        is_valid, error = validator.validate_syntax(invalid_yaml)
        assert is_valid is False
        assert isinstance(error, ValidationError)
        assert "YAML syntax error" in str(error)
        assert error.line is not None

    def test_validate_structure_valid(self, validator: YAMLValidator):
        """Test structure validation with a valid configuration."""
        valid_config: dict[str, object] = {
            "model": {"architecture": "unet"},
            "training": {"epochs": 10},
            "data": {"path": "/data"},
        }
        is_valid, errors = validator.validate_structure(valid_config)
        assert is_valid is True
        assert not errors

    def test_validate_structure_missing_section(
        self, validator: YAMLValidator
    ):
        """Test structure validation with a missing required section."""
        invalid_config: dict[str, object] = {
            "model": {"architecture": "unet"},
            "training": {"epochs": 10},
        }
        is_valid, errors = validator.validate_structure(invalid_config)
        assert is_valid is False
        assert len(errors) == 1
        assert "Missing required section: 'data'" in str(errors[0])

    def test_validate_types_valid(self, validator: YAMLValidator):
        """Test type validation with correct data types."""
        valid_config: dict[str, object] = {
            "training": {"epochs": 100, "learning_rate": 0.001},
            "model": {"num_classes": 2},
            "data": {"augment": True},
        }
        is_valid, errors = validator.validate_types(valid_config)
        assert is_valid is True
        assert not errors

    def test_validate_types_invalid(self, validator: YAMLValidator):
        """Test type validation with incorrect data types."""
        invalid_config: dict[str, object] = {
            "training": {"epochs": "100"},  # Should be int
            "model": {"num_classes": 2.0},  # Should be int
        }
        is_valid, errors = validator.validate_types(invalid_config)
        assert is_valid is False
        assert len(errors) == 2
        assert "Invalid type for training.epochs" in str(errors[0])
        assert "Invalid type for model.num_classes" in str(errors[1])

    def test_validate_values_valid(self, validator: YAMLValidator):
        """Test value validation with valid configuration values."""
        valid_config: dict[str, object] = {
            "model": {"architecture": "unet", "encoder": {"type": "resnet50"}},
            "training": {"loss": {"type": "dice"}, "epochs": 50},
        }
        is_valid, errors = validator.validate_values(valid_config)
        assert is_valid is True
        assert not errors

    def test_validate_values_unknown_architecture(
        self, validator: YAMLValidator
    ):
        """Test value validation with an unknown model architecture."""
        invalid_config: dict[str, object] = {
            "model": {"architecture": "unknown_model"}
        }
        is_valid, errors = validator.validate_values(invalid_config)
        assert is_valid is False
        assert len(errors) == 1
        assert "Unknown model architecture" in str(errors[0])

    def test_validate_values_non_positive_epochs(
        self, validator: YAMLValidator
    ):
        """Test value validation with a non-positive epochs value."""
        invalid_config: dict[str, object] = {"training": {"epochs": 0}}
        is_valid, errors = validator.validate_values(invalid_config)
        assert is_valid is False
        assert len(errors) == 1
        assert "Invalid epochs value" in str(errors[0])

    def test_get_nested_value(self, validator: YAMLValidator):
        """Test the internal _get_nested_value helper."""
        config: dict[str, object] = {"a": {"b": {"c": 123}}}
        # Access validator's private method for testing purposes
        get_nested = validator._get_nested_value
        assert get_nested(config, "a.b.c") == 123
        assert get_nested(config, "a.b") == {"c": 123}
        assert get_nested(config, "a.x.y") is None
        assert get_nested(config, "z") is None
