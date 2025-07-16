"""Unit tests for the error categorization engine."""

import pytest

from gui.utils.config.exceptions import ValidationError
from gui.utils.config.validation.error_categorizer import (
    CategorizedError,
    ErrorCategorizer,
    ErrorCategory,
    ErrorSeverity,
)


@pytest.fixture
def categorizer() -> ErrorCategorizer:
    """Fixture to provide an ErrorCategorizer instance."""
    return ErrorCategorizer()


class TestCategorizedError:
    """Test suite for the CategorizedError class."""

    def test_initialization(self):
        """Test successful initialization and attribute access."""
        original = ValidationError("Original error")
        cat_error = CategorizedError(
            original_error=original,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYNTAX,
            user_message="User-friendly message",
            suggestions=["Fix it"],
        )
        assert cat_error.original_error == original
        assert cat_error.severity == ErrorSeverity.CRITICAL
        assert cat_error.emoji == "🔴"
        assert cat_error.color == "error"
        assert cat_error.user_message == "User-friendly message"


class TestErrorCategorizer:
    """Test suite for the ErrorCategorizer class."""

    def test_categorize_syntax_error(self, categorizer: ErrorCategorizer):
        """Test categorization of a syntax error."""
        error = ValidationError("could not find expected ':'")
        cat_error = categorizer.categorize_error(error)

        assert cat_error.category == ErrorCategory.SYNTAX
        assert cat_error.severity == ErrorSeverity.CRITICAL
        assert "Missing colon" in cat_error.user_message

    def test_categorize_structure_error(self, categorizer: ErrorCategorizer):
        """Test categorization of a structure error."""
        error = ValidationError("missing required section 'model'")
        cat_error = categorizer.categorize_error(error)

        assert cat_error.category == ErrorCategory.STRUCTURE
        assert cat_error.severity == ErrorSeverity.CRITICAL
        assert "Missing required section" in cat_error.user_message

    def test_categorize_type_error(self, categorizer: ErrorCategorizer):
        """Test categorization of a type error."""
        error = ValidationError("expected int, got str")
        cat_error = categorizer.categorize_error(error)

        assert cat_error.category == ErrorCategory.TYPE
        assert cat_error.severity == ErrorSeverity.WARNING
        assert "Data type does not match" in cat_error.user_message

    def test_categorize_value_error(self, categorizer: ErrorCategorizer):
        """Test categorization of a value error."""
        error = ValidationError("unknown model architecture")
        cat_error = categorizer.categorize_error(error)

        assert cat_error.category == ErrorCategory.VALUE
        assert cat_error.severity == ErrorSeverity.WARNING
        assert "Unknown model architecture" in cat_error.user_message

    def test_categorize_default_case(self, categorizer: ErrorCategorizer):
        """Test categorization of an unknown error."""
        error = ValidationError("A completely unknown error")
        cat_error = categorizer.categorize_error(error)

        assert cat_error.category == ErrorCategory.VALUE
        assert cat_error.severity == ErrorSeverity.WARNING
        assert "A completely unknown error" in str(cat_error.original_error)
