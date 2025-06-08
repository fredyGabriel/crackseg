"""
Unit tests for error console and categorization functionality.

Tests the ErrorCategorizer and ErrorConsole components for proper
error categorization, user-friendly messaging, and UI rendering.
"""

from unittest.mock import Mock, patch

from scripts.gui.components.error_console import ErrorConsole
from scripts.gui.utils.config.exceptions import ValidationError
from scripts.gui.utils.config.validation.error_categorizer import (
    CategorizedError,
    ErrorCategorizer,
    ErrorCategory,
    ErrorSeverity,
)


class TestErrorCategorizer:
    """Test cases for ErrorCategorizer class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.categorizer = ErrorCategorizer()

    def test_categorize_syntax_error(self) -> None:
        """Test categorization of YAML syntax errors."""
        error = ValidationError(
            message="YAML syntax error: could not find expected ':'",
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
        assert ":" in str(categorized.suggestions)

    def test_categorize_structure_error(self) -> None:
        """Test categorization of structure validation errors."""
        error = ValidationError(
            message="Missing required section: 'model'",
            field="model",
        )

        categorized = self.categorizer.categorize_error(error)

        assert categorized.severity == ErrorSeverity.CRITICAL
        assert categorized.category == ErrorCategory.STRUCTURE
        assert "missing required section" in categorized.user_message.lower()
        assert any("template" in s.lower() for s in categorized.suggestions)

    def test_categorize_type_error(self) -> None:
        """Test categorization of type validation errors."""
        error = ValidationError(
            message="Invalid type for training.epochs: expected int, got str",
            field="training.epochs",
        )

        categorized = self.categorizer.categorize_error(error)

        assert categorized.severity == ErrorSeverity.WARNING
        assert categorized.category == ErrorCategory.TYPE
        assert "type" in categorized.user_message.lower()

    def test_categorize_value_error(self) -> None:
        """Test categorization of value validation errors."""
        error = ValidationError(
            message="Unknown model architecture: 'invalid_arch'",
            field="model.architecture",
        )

        categorized = self.categorizer.categorize_error(error)

        assert categorized.severity == ErrorSeverity.WARNING
        assert categorized.category == ErrorCategory.VALUE
        assert "architecture" in categorized.user_message.lower()
        assert any("unet" in s.lower() for s in categorized.suggestions)

    def test_categorize_multiple_errors_sorting(self) -> None:
        """Test that multiple errors are sorted by severity."""
        errors = [
            ValidationError(message="Unknown encoder: test", field="encoder"),
            ValidationError(message="YAML syntax error: invalid", line=1),
            ValidationError(
                message="Deprecated field: old_param", field="old_param"
            ),
        ]

        categorized = self.categorizer.categorize_errors(errors)

        # Should be sorted with critical errors first
        assert len(categorized) == 3
        assert (
            categorized[0].severity == ErrorSeverity.CRITICAL
        )  # Syntax error
        assert (
            categorized[1].severity == ErrorSeverity.WARNING
        )  # Unknown encoder
        assert (
            categorized[2].severity == ErrorSeverity.INFO
        )  # Deprecated field

    def test_context_building_with_line_info(self) -> None:
        """Test context building when line information is available."""
        content = """model:
  architecture: unet
  encoder:
    type: resnet50
training:
  epochs: invalid_value"""

        error = ValidationError(
            message="Type error in epochs",
            line=6,
            field="training.epochs",
        )

        categorized = self.categorizer.categorize_error(error, content)

        assert "line_content" in categorized.context
        assert "context_lines" in categorized.context
        assert "epochs: invalid_value" in categorized.context["line_content"]
        assert (
            categorized.context["context_start_line"] == 4
        )  # 3 lines before line 6

    def test_quick_fixes_generation(self) -> None:
        """Test generation of quick fix suggestions."""
        test_cases = [
            ("model.architecture", "architecture"),
            ("model.encoder", "encoder"),
            ("training.epochs", "positive"),
            ("training.batch_size", "power"),
        ]

        for field, expected_keyword in test_cases:
            error = ValidationError(message="Test error", field=field)
            categorized = self.categorizer.categorize_error(error)

            assert len(categorized.quick_fixes) > 0
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
        content_with_indent_issue = (
            "model:\narchitecture: unet"  # Missing indentation
        )

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

        # Test indentation issues
        suggestions = self.categorizer._analyze_line_context(  # type: ignore[reportPrivateUsage]
            1, content_with_indent_issue
        )
        assert any("indent" in s.lower() for s in suggestions)


class TestErrorConsole:
    """Test cases for ErrorConsole class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.console = ErrorConsole()

    @patch("streamlit.success")
    def test_render_no_errors(self, mock_success: Mock) -> None:
        """Test rendering when no errors are present."""
        self.console.render_error_summary([], key="test")

        mock_success.assert_called_once()
        assert "no errors found" in str(mock_success.call_args).lower()

    @patch("streamlit.error")
    @patch("streamlit.metric")
    def test_render_critical_errors(
        self, mock_metric: Mock, mock_error: Mock
    ) -> None:
        """Test rendering of critical errors."""
        errors = [
            ValidationError(message="YAML syntax error", line=1),
            ValidationError(message="Missing required section", field="model"),
        ]

        with patch.object(self.console, "_render_category_section"):
            self.console.render_error_summary(errors, key="test")

            # Should call error display for critical issues
            mock_error.assert_called()
            assert "critical" in str(mock_error.call_args).lower()

            # Should render metrics
            assert (
                mock_metric.call_count >= 4
            )  # Total, Critical, Warnings, Suggestions

    @patch("streamlit.warning")
    def test_render_warning_errors(self, mock_warning: Mock) -> None:
        """Test rendering of warning-level errors."""
        errors = [
            ValidationError(message="Unknown encoder: test", field="encoder"),
        ]

        with patch.object(self.console, "_render_category_section"):
            self.console.render_error_summary(errors, key="test")

            # Should show warning for non-critical issues
            mock_warning.assert_called()
            assert "issues found" in str(mock_warning.call_args).lower()

    def test_group_errors_by_category(self) -> None:
        """Test grouping of errors by category."""
        categorized_errors = [
            CategorizedError(
                ValidationError("syntax"),
                ErrorSeverity.CRITICAL,
                ErrorCategory.SYNTAX,
                "msg",
                [],
            ),
            CategorizedError(
                ValidationError("structure"),
                ErrorSeverity.CRITICAL,
                ErrorCategory.STRUCTURE,
                "msg",
                [],
            ),
            CategorizedError(
                ValidationError("another syntax"),
                ErrorSeverity.WARNING,
                ErrorCategory.SYNTAX,
                "msg",
                [],
            ),
        ]

        groups = self.console._group_errors_by_category(categorized_errors)  # type: ignore[reportPrivateUsage]

        assert len(groups) == 2  # SYNTAX and STRUCTURE
        assert len(groups[ErrorCategory.SYNTAX]) == 2
        assert len(groups[ErrorCategory.STRUCTURE]) == 1

    @patch("streamlit.expander")
    def test_render_error_details_expandable(
        self, mock_expander: Mock
    ) -> None:
        """Test rendering of expandable error details."""
        error = CategorizedError(
            original_error=ValidationError(
                "Test error", line=5, field="test.field"
            ),
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYNTAX,
            user_message="User friendly message",
            suggestions=["suggestion 1", "suggestion 2"],
            quick_fixes=["fix 1"],
            context={"line_content": "test: value"},
        )

        mock_expander_context = Mock()
        mock_expander.return_value.__enter__ = Mock(
            return_value=mock_expander_context
        )
        mock_expander.return_value.__exit__ = Mock(return_value=None)

        # Create proper context manager mocks for columns
        mock_col1 = Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)

        mock_col2 = Mock()
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.markdown"),
        ):
            mock_columns.return_value = [mock_col1, mock_col2]

            self.console.render_error_details_expandable(error, "test_key")

            # Should create expander with emoji and message
            mock_expander.assert_called_once()
            title = str(mock_expander.call_args[0][0])
            assert "🔴" in title  # Critical emoji
            assert "User friendly message" in title
            assert "Line 5" in title

    @patch("streamlit.tabs")
    def test_render_fix_suggestions_interactive(self, mock_tabs: Mock) -> None:
        """Test rendering of interactive fix suggestions."""
        errors = [
            CategorizedError(
                ValidationError("critical"),
                ErrorSeverity.CRITICAL,
                ErrorCategory.SYNTAX,
                "Critical error",
                [],
                None,
            ),
            CategorizedError(
                ValidationError("warning"),
                ErrorSeverity.WARNING,
                ErrorCategory.TYPE,
                "Warning error",
                [],
                None,
            ),
        ]

        # Create proper context manager mocks for tabs
        mock_tab1 = Mock()
        mock_tab1.__enter__ = Mock(return_value=mock_tab1)
        mock_tab1.__exit__ = Mock(return_value=None)

        mock_tab2 = Mock()
        mock_tab2.__enter__ = Mock(return_value=mock_tab2)
        mock_tab2.__exit__ = Mock(return_value=None)

        mock_tabs.return_value = [mock_tab1, mock_tab2]

        with patch.object(
            self.console, "_render_solution_card"
        ) as mock_render_card:
            self.console.render_fix_suggestions_interactive(errors, "test_key")

            # Should create tabs for different severities
            mock_tabs.assert_called_once()
            tab_names = mock_tabs.call_args[0][0]
            assert any("Critical" in name for name in tab_names)
            assert any("Warning" in name for name in tab_names)

            # Should render solution cards
            assert mock_render_card.call_count == 2

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


class TestErrorCategorizerPatterns:
    """Test specific pattern matching in ErrorCategorizer."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.categorizer = ErrorCategorizer()

    def test_syntax_patterns_matching(self) -> None:
        """Test that syntax error patterns are correctly matched."""
        test_cases = [
            ("could not find expected ':'", "colon"),
            ("found unexpected end of stream", "incomplete"),
            ("mapping values are not allowed", "mapping"),
            ("found character that cannot start", "character"),
        ]

        for error_message, expected_keyword in test_cases:
            error = ValidationError(error_message)
            categorized = self.categorizer.categorize_error(error)

            # Check both user message and suggestions for the keyword
            assert any(
                expected_keyword.lower() in text.lower()
                for text in [categorized.user_message]
                + categorized.suggestions
            )

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

            assert len(found_keywords) > 0, (
                f"No keywords for {field}: expected {expected_keywords}, "
                f"got {suggestions}"
            )
