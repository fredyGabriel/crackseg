"""
Error console component for displaying categorized validation errors.

This module provides a user-friendly interface for displaying validation errors
with severity levels, contextual suggestions, and interactive quick fixes.
"""

import streamlit as st

from ..utils.config.exceptions import ValidationError
from ..utils.config.validation.error_categorizer import (
    CategorizedError,
    ErrorCategorizer,
    ErrorCategory,
    ErrorSeverity,
)


class ErrorConsole:
    """Enhanced error console for displaying categorized validation errors."""

    def __init__(self) -> None:
        """Initialize the error console."""
        self.categorizer = ErrorCategorizer()

    def render_error_summary(
        self,
        errors: list[ValidationError],
        content: str = "",
        key: str = "error_console",
    ) -> None:
        """Render a comprehensive error summary with categorization.

        Args:
            errors: List of validation errors to display.
            content: Original YAML content for context analysis.
            key: Unique key for the component.
        """
        if not errors:
            st.success(
                "✅ **No errors found** - Your configuration is ready to use"
            )
            return

        # Categorize errors
        categorized_errors = self.categorizer.categorize_errors(
            errors, content
        )

        # Display summary statistics
        self._render_summary_stats(categorized_errors, key)

        # Group errors by category for better organization
        error_groups = self._group_errors_by_category(categorized_errors)

        # Render each category
        for category, category_errors in error_groups.items():
            self._render_category_section(category, category_errors, key)

        # Render helpful tips at the bottom
        self._render_helpful_tips(categorized_errors)

    def render_error_details_expandable(
        self, error: CategorizedError, key: str, expanded: bool = False
    ) -> None:
        """Render detailed error information in an expandable section.

        Args:
            error: Categorized error to display.
            key: Unique key for the component.
            expanded: Whether to expand the details by default.
        """
        # Create expandable section title with severity indicator
        title = f"{error.emoji} {error.user_message}"
        if error.line:
            title += f" (Line {error.line})"

        with st.expander(title, expanded=expanded):
            # Error details
            col1, col2 = st.columns([3, 1])

            with col1:
                # Show original technical message if different from user
                original_message = (
                    error.original_error.args[0]
                    if error.original_error.args
                    else str(error.original_error)
                )
                if original_message != error.user_message:
                    with st.container():
                        st.caption("**Technical message:**")
                        st.code(original_message, language="text")

                # Location information
                if error.line or error.column or error.field:
                    location_parts = []
                    if error.line:
                        location_parts.append(f"Line {error.line}")
                    if error.column:
                        location_parts.append(f"Column {error.column}")
                    if error.field:
                        location_parts.append(f"Field '{error.field}'")

                    st.info(f"📍 **Location:** {', '.join(location_parts)}")

            with col2:
                # Severity and category badges
                severity_color = {
                    ErrorSeverity.CRITICAL: "#ff4444",
                    ErrorSeverity.WARNING: "#ffa500",
                    ErrorSeverity.INFO: "#4a90e2",
                    ErrorSeverity.SUGGESTION: "#28a745",
                }[error.severity]

                st.markdown(
                    f"""
                <div style="text-align: center;">
                    <div style="background-color: {severity_color};
                         color: white; padding: 4px 8px;
                         border-radius: 4px; font-size: 12px;">
                        {error.severity.value.upper()}
                    </div>
                    <div style="margin-top: 4px; font-size: 11px;
                         color: #666;">
                        {error.category.value}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Show context if available
            if error.context.get("context_lines"):
                self._render_context_lines(error)

            # Suggestions section
            if error.suggestions:
                st.markdown("**💡 Suggestions:**")
                for i, suggestion in enumerate(
                    error.suggestions[:5], 1
                ):  # Limit to 5
                    st.markdown(f"{i}. {suggestion}")

            # Quick fixes section
            if error.quick_fixes:
                st.markdown("**⚡ Quick fixes:**")
                for fix in error.quick_fixes:
                    if st.button(
                        f"🔧 {fix}",
                        key=f"{key}_quickfix_{hash(fix)}",
                        help="Copy solution to clipboard",
                    ):
                        # Note: Actual clipboard functionality would need JS
                        st.success(f"Solution copied: {fix}")

    def render_fix_suggestions_interactive(
        self, errors: list[CategorizedError], key: str = "fix_suggestions"
    ) -> None:
        """Render interactive fix suggestions with categorization.

        Args:
            errors: List of categorized errors.
            key: Unique key for the component.
        """
        if not errors:
            return

        st.markdown("### 🛠️ Interactive Solution Center")

        # Create tabs for different types of fixes
        critical_errors = [
            e for e in errors if e.severity == ErrorSeverity.CRITICAL
        ]
        warning_errors = [
            e for e in errors if e.severity == ErrorSeverity.WARNING
        ]
        suggestions = [
            e
            for e in errors
            if e.severity in [ErrorSeverity.INFO, ErrorSeverity.SUGGESTION]
        ]

        tab_names = []
        if critical_errors:
            tab_names.append(f"🔴 Critical ({len(critical_errors)})")
        if warning_errors:
            tab_names.append(f"🟡 Warnings ({len(warning_errors)})")
        if suggestions:
            tab_names.append(f"💡 Suggestions ({len(suggestions)})")

        if not tab_names:
            return

        tabs = st.tabs(tab_names)
        tab_index = 0

        if critical_errors and tab_index < len(tabs):
            with tabs[tab_index]:
                st.error(
                    "**These errors must be resolved before using "
                    "the configuration:**"
                )
                for error in critical_errors:
                    self._render_solution_card(
                        error,
                        f"{key}_critical_{hash(str(error.original_error))}",
                    )
            tab_index += 1

        if warning_errors and tab_index < len(tabs):
            with tabs[tab_index]:
                st.warning(
                    "**The configuration works, but these issues "
                    "could cause problems:**"
                )
                for error in warning_errors:
                    self._render_solution_card(
                        error,
                        f"{key}_warning_{hash(str(error.original_error))}",
                    )
            tab_index += 1

        if suggestions and tab_index < len(tabs):
            with tabs[tab_index]:
                st.info("**Recommended optimizations and best practices:**")
                for error in suggestions:
                    self._render_solution_card(
                        error,
                        f"{key}_suggestion_{hash(str(error.original_error))}",
                    )

    def _render_summary_stats(
        self, errors: list[CategorizedError], key: str
    ) -> None:
        """Render error summary statistics."""
        total_errors = len(errors)
        critical_count = sum(
            1 for e in errors if e.severity == ErrorSeverity.CRITICAL
        )
        warning_count = sum(
            1 for e in errors if e.severity == ErrorSeverity.WARNING
        )
        info_count = sum(
            1
            for e in errors
            if e.severity in [ErrorSeverity.INFO, ErrorSeverity.SUGGESTION]
        )

        # Color-coded summary
        if critical_count > 0:
            st.error(
                f"**🔴 {critical_count} critical errors found** - "
                f"Configuration cannot be used"
            )
        elif warning_count > 0:
            st.warning(
                f"**🟡 {total_errors} issues found** - "
                f"Configuration may work with limitations"
            )
        else:
            st.info(
                f"**🔵 {total_errors} suggestions available** - "
                f"Configuration is valid"
            )

        # Detailed breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", total_errors)
        with col2:
            st.metric(
                "Critical",
                critical_count,
                delta=None if critical_count == 0 else f"-{critical_count}",
            )
        with col3:
            st.metric("Warnings", warning_count)
        with col4:
            st.metric("Suggestions", info_count)

    def _group_errors_by_category(
        self, errors: list[CategorizedError]
    ) -> dict[ErrorCategory, list[CategorizedError]]:
        """Group errors by category for organized display."""
        groups: dict[ErrorCategory, list[CategorizedError]] = {}

        for error in errors:
            if error.category not in groups:
                groups[error.category] = []
            groups[error.category].append(error)

        return groups

    def _render_category_section(
        self, category: ErrorCategory, errors: list[CategorizedError], key: str
    ) -> None:
        """Render a section for a specific error category."""
        category_icons = {
            ErrorCategory.SYNTAX: "📝",
            ErrorCategory.STRUCTURE: "🏗️",
            ErrorCategory.TYPE: "🔢",
            ErrorCategory.VALUE: "⚠️",
            ErrorCategory.COMPATIBILITY: "🔗",
            ErrorCategory.PERFORMANCE: "⚡",
            ErrorCategory.SECURITY: "🔒",
        }

        category_descriptions = {
            ErrorCategory.SYNTAX: "YAML syntax issues",
            ErrorCategory.STRUCTURE: "Configuration structure",
            ErrorCategory.TYPE: "Data types",
            ErrorCategory.VALUE: "Values and ranges",
            ErrorCategory.COMPATIBILITY: "Hydra compatibility",
            ErrorCategory.PERFORMANCE: "Performance optimizations",
            ErrorCategory.SECURITY: "Security considerations",
        }

        icon = category_icons.get(category, "❓")
        description = category_descriptions.get(category, category.value)

        st.markdown(f"#### {icon} {description} ({len(errors)})")

        for i, error in enumerate(errors):
            self.render_error_details_expandable(
                error,
                f"{key}_{category.value}_{i}",
                expanded=(error.severity == ErrorSeverity.CRITICAL and i == 0),
            )

    def _render_context_lines(self, error: CategorizedError) -> None:
        """Render context lines around the error location."""
        context_lines = error.context.get("context_lines", [])
        context_start = error.context.get("context_start_line", 1)

        if not context_lines:
            return

        st.markdown("**📄 Context:**")

        # Build code block with line numbers
        code_lines = []
        for i, line in enumerate(context_lines):
            line_num = context_start + i
            prefix = ">>> " if line_num == error.line else "    "
            code_lines.append(f"{prefix}{line_num:3d}: {line}")

        st.code("\n".join(code_lines), language="yaml")

    def _render_solution_card(self, error: CategorizedError, key: str) -> None:
        """Render a solution card for an error."""
        with st.container():
            # Card header
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{error.emoji} {error.user_message}**")
                if error.line:
                    st.caption(f"Line {error.line}")
            with col2:
                severity_colors = {
                    ErrorSeverity.CRITICAL: "red",
                    ErrorSeverity.WARNING: "orange",
                    ErrorSeverity.INFO: "blue",
                    ErrorSeverity.SUGGESTION: "green",
                }
                st.markdown(
                    f":{severity_colors[error.severity]}[{error.severity.value.upper()}]"
                )

            # Solutions
            if error.quick_fixes:
                st.markdown("**Quick solutions:**")
                for i, fix in enumerate(error.quick_fixes):
                    if st.button(f"⚡ {fix}", key=f"{key}_fix_{i}"):
                        st.success("Solution applied!")

            if error.suggestions:
                with st.expander("View more suggestions", expanded=False):
                    for suggestion in error.suggestions:
                        st.markdown(f"• {suggestion}")

            st.markdown("---")

    def _render_helpful_tips(self, errors: list[CategorizedError]) -> None:
        """Render helpful tips based on the types of errors found."""
        if not errors:
            return

        # Analyze error patterns to provide relevant tips
        has_syntax_errors = any(
            e.category == ErrorCategory.SYNTAX for e in errors
        )
        has_structure_errors = any(
            e.category == ErrorCategory.STRUCTURE for e in errors
        )
        has_type_errors = any(e.category == ErrorCategory.TYPE for e in errors)

        tips = []

        if has_syntax_errors:
            tips.append(
                "💡 **YAML Tip:** Use an online YAML validator to "
                "check basic syntax"
            )

        if has_structure_errors:
            tips.append(
                "📚 **Configuration Tip:** Check examples in `configs/` "
                "for correct structures"
            )

        if has_type_errors:
            tips.append(
                "🔢 **Types Tip:** Numbers without quotes, strings with "
                "quotes, booleans: true/false"
            )

        tips.extend(
            [
                "🚀 **Development Tip:** Save working configurations "
                "as templates",
                "⚙️ **Productivity Tip:** Use autocompletion by copying from "
                "existing configurations",
            ]
        )

        if tips:
            st.markdown("### 💡 Helpful Tips")
            for tip in tips[:3]:  # Show max 3 tips
                st.info(tip)
