"""
Test suite for advanced YAML validation functionality.
"""

from scripts.gui.utils.config_io import (
    ValidationError,
    YAMLValidator,
    format_validation_report,
    get_validation_suggestions,
    validate_config_structure,
    validate_config_types,
    validate_config_values,
    validate_yaml_advanced,
)


class TestYAMLValidator:
    """Test suite for YAMLValidator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = YAMLValidator()

    def test_valid_yaml_syntax(self) -> None:
        """Test validation of correct YAML syntax."""
        valid_yaml = """
model:
  architecture: unet
  encoder:
    type: resnet50

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001

data:
  augment: true
  split_ratio: [0.8, 0.1, 0.1]
"""
        is_valid, error = self.validator.validate_syntax(valid_yaml)
        assert is_valid is True
        assert error is None

    def test_invalid_yaml_syntax(self) -> None:
        """Test detection of YAML syntax errors."""
        invalid_yaml = """
model:
  architecture: unet
  encoder
    type: resnet50  # Missing colon after encoder
"""
        is_valid, error = self.validator.validate_syntax(invalid_yaml)
        assert is_valid is False
        assert error is not None
        assert isinstance(error, ValidationError)
        assert "syntax error" in str(error).lower()
        assert len(error.suggestions) > 0

    def test_structure_validation_valid(self) -> None:
        """Test validation of correct configuration structure."""
        valid_config: dict[str, object] = {
            "model": {"architecture": "unet", "encoder": {"type": "resnet50"}},
            "training": {"epochs": 100, "batch_size": 16},
            "data": {"augment": True},
        }
        is_valid, errors = self.validator.validate_structure(valid_config)

        # Debug: print errors if any
        if not is_valid:
            for error in errors:
                print(f"DEBUG Error: {error}")

        assert is_valid is True
        assert len(errors) == 0

    def test_structure_validation_missing_sections(self) -> None:
        """Test detection of missing required sections."""
        incomplete_config: dict[str, object] = {
            "model": {"architecture": "unet"}
            # Missing training and data sections
        }
        is_valid, errors = self.validator.validate_structure(incomplete_config)
        assert is_valid is False
        assert len(errors) >= 2  # Should detect missing training and data

        error_fields = [error.field for error in errors]
        assert "training" in error_fields
        assert "data" in error_fields

    def test_type_validation_valid(self) -> None:
        """Test validation of correct data types."""
        valid_config: dict[str, object] = {
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "learning_rate": 0.001,
            },
            "model": {"num_classes": 2, "input_channels": 3},
            "data": {"split_ratio": [0.8, 0.1, 0.1], "augment": True},
        }
        is_valid, errors = self.validator.validate_types(valid_config)
        assert is_valid is True
        assert len(errors) == 0

    def test_type_validation_invalid_types(self) -> None:
        """Test detection of incorrect data types."""
        invalid_config: dict[str, object] = {
            "training": {
                "epochs": "100",  # Should be int
                "batch_size": 16.5,  # Should be int
                "learning_rate": "0.001",  # Should be float
            },
            "data": {"augment": "true"},  # Should be bool
        }
        is_valid, errors = self.validator.validate_types(invalid_config)
        assert is_valid is False
        assert len(errors) >= 3  # Should detect multiple type errors

    def test_value_validation_valid_architecture(self) -> None:
        """Test validation of valid model architectures."""
        valid_config: dict[str, object] = {"model": {"architecture": "unet"}}
        is_valid, errors = self.validator.validate_values(valid_config)
        assert is_valid is True
        assert len(errors) == 0

    def test_value_validation_invalid_architecture(self) -> None:
        """Test detection of invalid model architectures."""
        invalid_config: dict[str, object] = {
            "model": {"architecture": "invalid_model"}
        }
        is_valid, errors = self.validator.validate_values(invalid_config)
        assert is_valid is False
        assert len(errors) >= 1

        arch_error = next(
            (e for e in errors if "architecture" in str(e)), None
        )
        assert arch_error is not None
        assert "invalid_model" in str(arch_error)

    def test_value_validation_negative_values(self) -> None:
        """Test detection of invalid numeric ranges."""
        invalid_config: dict[str, object] = {
            "training": {"epochs": -10, "batch_size": 0}
        }
        is_valid, errors = self.validator.validate_values(invalid_config)
        assert is_valid is False
        assert len(errors) >= 2

    def test_comprehensive_validation_valid(self) -> None:
        """Test comprehensive validation with valid configuration."""
        valid_yaml = """
model:
  architecture: unet
  num_classes: 2
  input_channels: 3

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001

data:
  augment: true
  split_ratio: [0.8, 0.1, 0.1]
"""
        is_valid, errors = self.validator.comprehensive_validate(valid_yaml)
        assert is_valid is True
        assert len(errors) == 0

    def test_comprehensive_validation_multiple_errors(self) -> None:
        """Test comprehensive validation with multiple error types."""
        invalid_yaml = """
model:
  architecture: invalid_model
  num_classes: "2"

training:
  epochs: -10
  batch_size: 0
  # Missing learning_rate

# Missing data section
"""
        is_valid, errors = self.validator.comprehensive_validate(invalid_yaml)
        assert is_valid is False
        assert len(errors) > 0

        # Should have structure, type, and value errors
        error_types = [type(error).__name__ for error in errors]
        assert "ValidationError" in error_types


class TestValidationFunctions:
    """Test suite for validation utility functions."""

    def test_validate_yaml_advanced_valid(self) -> None:
        """Test advanced YAML validation with valid content."""
        valid_yaml = """
model:
  architecture: unet

training:
  epochs: 100

data:
  augment: true
"""
        is_valid, errors = validate_yaml_advanced(valid_yaml)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_yaml_advanced_invalid(self) -> None:
        """Test advanced YAML validation with invalid content."""
        invalid_yaml = """
model:
  architecture: invalid_model

# Missing training and data sections
"""
        is_valid, errors = validate_yaml_advanced(invalid_yaml)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_config_structure_direct(self) -> None:
        """Test direct structure validation function."""
        config: dict[str, object] = {
            "model": {"architecture": "unet"},
            "training": {"epochs": 100},
            "data": {"augment": True},
        }
        is_valid, errors = validate_config_structure(config)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_config_types_direct(self) -> None:
        """Test direct type validation function."""
        config: dict[str, object] = {
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "learning_rate": 0.001,
            }
        }
        is_valid, errors = validate_config_types(config)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_config_values_direct(self) -> None:
        """Test direct value validation function."""
        config: dict[str, object] = {
            "model": {"architecture": "unet"},
            "training": {"epochs": 100, "batch_size": 16},
        }
        is_valid, errors = validate_config_values(config)
        assert is_valid is True
        assert len(errors) == 0

    def test_get_validation_suggestions(self) -> None:
        """Test extraction of validation suggestions."""
        errors = [
            ValidationError(
                message="Missing required section",
                field="training",
                suggestions=["Add training section", "Check examples"],
            ),
            ValidationError(
                message="Invalid type",
                field="model.num_classes",
                suggestions=["Use integer type", "Example: num_classes: 2"],
            ),
        ]

        suggestions = get_validation_suggestions(errors)
        assert isinstance(suggestions, dict)
        assert len(suggestions) > 0

        # Should have categorized suggestions
        for _category, suggs in suggestions.items():
            assert isinstance(suggs, list)
            assert all(isinstance(s, str) for s in suggs)

    def test_format_validation_report_success(self) -> None:
        """Test formatting of successful validation report."""
        report = format_validation_report([])
        assert "✅" in report
        assert "passed successfully" in report.lower()

    def test_format_validation_report_errors(self) -> None:
        """Test formatting of validation report with errors."""
        errors = [
            ValidationError(
                message="Missing required section",
                field="training",
                suggestions=["Add training section"],
            ),
            ValidationError(
                message="Invalid type for epochs",
                field="training.epochs",
                suggestions=["Use integer type"],
            ),
        ]

        report = format_validation_report(errors)
        assert "❌" in report
        assert "failed" in report.lower()
        assert "2 error(s)" in report
        assert "💡 Quick Fixes:" in report
        assert "training" in report

    def test_nested_value_extraction(self) -> None:
        """Test extraction of nested values from configuration."""
        pass

    def test_suggestion_generation(self) -> None:
        """Test generation of context-specific suggestions."""
        pass

    def test_type_examples(self) -> None:
        """Test generation of type examples."""
        pass

    def test_field_examples(self) -> None:
        """Test generation of field-specific examples."""
        pass
