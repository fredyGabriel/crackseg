"""Unit tests for validation report formatters."""

import pytest

from gui.utils.config.exceptions import ValidationError
from gui.utils.config.formatters import (
    format_validation_report,
    get_validation_suggestions,
)


@pytest.fixture
def sample_errors() -> list[ValidationError]:
    """Fixture to provide a sample list of validation errors."""
    return [
        ValidationError(
            "YAML syntax error", line=1, suggestions=["Fix indentation"]
        ),
        ValidationError(
            "Missing required section: 'model'",
            field="model",
            suggestions=["Add 'model' section"],
        ),
        ValidationError(
            "Invalid type for training.epochs",
            field="training.epochs",
            suggestions=["Use an integer for epochs"],
        ),
        ValidationError(
            "Unknown model architecture",
            field="model.architecture",
            suggestions=["Use a supported architecture"],
        ),
        ValidationError(
            "A general error", suggestions=["Check documentation"]
        ),
    ]


class TestFormatters:
    """Test suite for formatting functions."""

    def test_get_validation_suggestions(
        self, sample_errors: list[ValidationError]
    ):
        """Test extraction and categorization of suggestions."""
        suggestions = get_validation_suggestions(sample_errors)

        assert "syntax" in suggestions
        assert "Fix indentation" in suggestions["syntax"]
        assert "structure" in suggestions
        assert "Add 'model' section" in suggestions["structure"]
        assert "types" in suggestions
        assert "Use an integer for epochs" in suggestions["types"]
        assert "values" in suggestions
        assert "Use a supported architecture" in suggestions["values"]
        assert "general" in suggestions
        assert "Check documentation" in suggestions["general"]

    def test_get_validation_suggestions_empty(self):
        """Test suggestion extraction with no errors."""
        suggestions = get_validation_suggestions([])
        assert not suggestions

    def test_get_validation_suggestions_removes_duplicates(self):
        """Test that duplicate suggestions are removed."""
        errors = [
            ValidationError("Error 1", suggestions=["Fix A", "Fix B"]),
            ValidationError("Error 2", suggestions=["Fix A", "Fix C"]),
        ]
        suggestions = get_validation_suggestions(errors)
        assert (
            suggestions["general"].sort() == ["Fix A", "Fix B", "Fix C"].sort()
        )

    def test_format_validation_report_success(self):
        """Test report formatting with no errors."""
        report = format_validation_report([])
        assert "validation passed successfully" in report

    def test_format_validation_report_with_errors(
        self, sample_errors: list[ValidationError]
    ):
        """Test report formatting with a list of errors."""
        report = format_validation_report(sample_errors)

        assert "validation failed with 5 error(s)" in report
        assert "Syntax Errors:" in report
        assert "Structure Errors:" in report
        assert "Type Errors:" in report
        assert "Value Errors:" in report
        assert "Quick Fixes:" in report
        assert "Use a supported architecture" in report
