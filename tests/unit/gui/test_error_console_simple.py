"""
Simplified unit tests for error console and categorization functionality.

Tests core functionality of ErrorCategorizer and ErrorConsole components.
"""

from scripts.gui.utils.config.exceptions import ValidationError
from scripts.gui.utils.config.validation.error_categorizer import (
    CategorizedError,
    ErrorCategorizer,
    ErrorCategory,
    ErrorSeverity,
)


class TestErrorCategorizerCore:
    """Test core functionality of ErrorCategorizer class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.categorizer = ErrorCategorizer()

    def test_categorize_syntax_error(self) -> None:
        """Test categorization of YAML syntax errors."""
        error = ValidationError(
            "YAML syntax error: could not find expected ':'",
            line=5,
            column=10,
        )

        categorized = self.categorizer.categorize_error(
            error, "model:\n  architecture unet"
        )

        assert categorized.severity == ErrorSeverity.CRITICAL
        assert categorized.category == ErrorCategory.SYNTAX
        assert "missing colon" in categorized.user_message.lower()
        assert len(categorized.suggestions) > 0

    def test_categorize_structure_error(self) -> None:
        """Test categorization of structure validation errors."""
        error = ValidationError(
            "Missing required section: 'model'",
            field="model",
        )

        categorized = self.categorizer.categorize_error(error)

        assert categorized.severity == ErrorSeverity.CRITICAL
        assert categorized.category == ErrorCategory.STRUCTURE
        assert "missing required section" in categorized.user_message.lower()

    def test_categorize_type_error(self) -> None:
        """Test categorization of type validation errors."""
        error = ValidationError(
            "Invalid type for training.epochs: expected int, got str",
            field="training.epochs",
        )

        categorized = self.categorizer.categorize_error(error)

        assert categorized.severity == ErrorSeverity.WARNING
        assert categorized.category == ErrorCategory.TYPE

    def test_categorize_value_error(self) -> None:
        """Test categorization of value validation errors."""
        error = ValidationError(
            "Unknown model architecture: 'invalid_arch'",
            field="model.architecture",
        )

        categorized = self.categorizer.categorize_error(error)

        assert categorized.severity == ErrorSeverity.WARNING
        assert categorized.category == ErrorCategory.VALUE
        assert any("unet" in s.lower() for s in categorized.suggestions)

    def test_multiple_errors_sorting(self) -> None:
        """Test that multiple errors are sorted by severity."""
        errors = [
            ValidationError("Unknown encoder: test", field="encoder"),
            ValidationError("YAML syntax error: invalid", line=1),
            ValidationError("Deprecated field: old_param", field="old_param"),
        ]

        categorized = self.categorizer.categorize_errors(errors)

        # Should be sorted with critical errors first
        assert len(categorized) == 3
        # Find the critical error (syntax error)
        critical_errors = [
            e for e in categorized if e.severity == ErrorSeverity.CRITICAL
        ]
        assert len(critical_errors) == 1
        assert "syntax" in critical_errors[0].original_error.args[0].lower()

    def test_context_building_with_line_info(self) -> None:
        """Test context building when line information is available."""
        content = """model:
  architecture: unet
  encoder:
    type: resnet50
training:
  epochs: invalid_value"""

        error = ValidationError(
            "Type error in epochs",
            line=6,
            field="training.epochs",
        )

        categorized = self.categorizer.categorize_error(error, content)

        assert "line_content" in categorized.context
        assert "context_lines" in categorized.context
        assert "epochs: invalid_value" in categorized.context["line_content"]

    def test_quick_fixes_generation(self) -> None:
        """Test generation of quick fix suggestions."""
        test_cases = [
            ("model.architecture", "architecture"),
            ("model.encoder", "encoder"),
            ("training.epochs", "positive"),
            ("training.batch_size", "power"),
        ]

        for field, expected_keyword in test_cases:
            error = ValidationError("Test error", field=field)
            categorized = self.categorizer.categorize_error(error)

            assert len(categorized.quick_fixes) > 0
            # At least one quick fix should contain the expected keyword
            assert any(
                expected_keyword.lower() in fix.lower()
                for fix in categorized.quick_fixes
            )

    def test_humanize_error_message(self) -> None:
        """Test humanization of technical error messages."""
        test_cases = [
            (
                "YAML syntax error: mapping values are not allowed",
                "mapping format",
            ),
            ("could not find expected ':'", "missing colon"),
            ("found unexpected end of stream", "incomplete file"),
        ]

        for technical_msg, friendly_keyword in test_cases:
            result = self.categorizer._humanize_error_message(technical_msg)  # type: ignore[reportPrivateUsage]
            assert friendly_keyword.lower() in result.lower()
            assert not result.startswith("YAML syntax error:")

    def test_line_context_analysis(self) -> None:
        """Test analysis of line context for common issues."""
        content_with_tabs = (
            "model:\n\tarchitecture: unet"  # Tab instead of spaces
        )
        content_with_trailing = (
            "model:  \n  architecture: unet"  # Trailing spaces
        )
        # Caso de indentación faltante
        content_with_indent_issue = "model:\narchitecture: unet"

        # Test tab detection
        suggestions = self.categorizer._analyze_line_context(  # type: ignore[reportPrivateUsage]
            2, content_with_tabs
        )
        assert any(
            "spaces" in s.lower() and "tabs" in s.lower() for s in suggestions
        )

        # Test trailing spaces detection
        suggestions = self.categorizer._analyze_line_context(  # type: ignore[reportPrivateUsage]
            1, content_with_trailing
        )
        assert any("trailing spaces" in s.lower() for s in suggestions)

        # Test missing indentation detection
        suggestions = self.categorizer._analyze_line_context(  # type: ignore[reportPrivateUsage]
            1, content_with_indent_issue
        )
        assert any("indent" in s.lower() for s in suggestions)


class TestCategorizedError:
    """Test CategorizedError functionality."""

    def test_categorized_error_properties(self) -> None:
        """Test CategorizedError property accessors."""
        original_error = ValidationError(
            "Test", line=10, column=5, field="test.field"
        )

        categorized = CategorizedError(
            original_error=original_error,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.TYPE,
            user_message="User message",
            suggestions=["suggestion"],
        )

        # Test property delegation
        assert categorized.line == 10
        assert categorized.column == 5
        assert categorized.field == "test.field"

        # Test emoji and color mapping
        assert categorized.emoji == "🟡"  # Warning emoji
        assert categorized.color == "warning"

    def test_severity_emoji_mapping(self) -> None:
        """Test that all severity levels have proper emoji mapping."""
        severities = [
            (ErrorSeverity.CRITICAL, "🔴"),
            (ErrorSeverity.WARNING, "🟡"),
            (ErrorSeverity.INFO, "🔵"),
            (ErrorSeverity.SUGGESTION, "💡"),
        ]

        for severity, expected_emoji in severities:
            error = CategorizedError(
                ValidationError("test"),
                severity,
                ErrorCategory.SYNTAX,
                "test message",
                [],
            )
            assert error.emoji == expected_emoji


class TestErrorCategorizerPatterns:
    """Test specific pattern matching in ErrorCategorizer."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.categorizer = ErrorCategorizer()

    def test_field_specific_suggestions(self) -> None:
        """Test field-specific suggestion generation."""
        field_tests = [
            ("model.architecture", ["unet", "deeplabv3plus"]),
            ("model.encoder", ["resnet50", "efficientnet"]),
            ("training.epochs", ["50-200", "early stopping"]),
            ("training.batch_size", ["power of 2", "RTX 3070"]),
            ("training.learning_rate", ["1e-4", "scheduler"]),
        ]

        for field, expected_keywords in field_tests:
            suggestions = self.categorizer._analyze_field_context(field)  # type: ignore[reportPrivateUsage]

            # Check that at least one expected keyword appears in suggestions
            found_keywords = []
            for keyword in expected_keywords:
                if any(keyword.lower() in s.lower() for s in suggestions):
                    found_keywords.append(keyword)

            assert (
                len(found_keywords) > 0
            ), f"No expected keywords found for field {field}"

    def test_syntax_pattern_detection(self) -> None:
        """Test detection of common syntax patterns."""
        syntax_errors = [
            "could not find expected ':'",
            "found unexpected end of stream",
            "mapping values are not allowed",
            "found character that cannot start",
        ]

        for error_msg in syntax_errors:
            error = ValidationError(error_msg)
            categorized = self.categorizer.categorize_error(error)

            assert categorized.category == ErrorCategory.SYNTAX
            assert categorized.severity == ErrorSeverity.CRITICAL
            assert len(categorized.suggestions) > 0
