"""
Comprehensive Error Reporting System for Configuration Validation.

This module provides advanced error reporting capabilities with detailed
context, suggestions, and actionable recommendations for fixing configuration
validation errors in crack segmentation projects.

Key Features:
- Categorized error reporting with severity levels
- Context-aware error messages with line/column information
- Intelligent suggestions based on error patterns
- HTML and plain text report generation
- Error aggregation and deduplication
- User-friendly explanations for technical issues
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for better organization."""

    SYNTAX = "syntax"
    SCHEMA = "schema"
    TYPE = "type"
    VALUE = "value"
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class ErrorReport:
    """Comprehensive error report container."""

    title: str
    description: str
    severity: ErrorSeverity
    category: ErrorCategory
    field_name: str | None = None
    line: int | None = None
    column: int | None = None
    suggestions: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    related_docs: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


class ConfigErrorReporter:
    """
    Advanced error reporter for configuration validation issues.

    Provides comprehensive error analysis, categorization, and reporting
    with intelligent suggestions and context-aware recommendations.
    """

    def __init__(self) -> None:
        """Initialize the error reporter."""
        self.error_patterns = self._initialize_error_patterns()
        self.suggestion_templates = self._initialize_suggestion_templates()
        self.documentation_links = self._initialize_documentation_links()

    def generate_comprehensive_report(
        self,
        errors: list[ValidationError],
        warnings: list[str] | None = None,
        config_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive error report from validation errors.

        Args:
            errors: List of validation errors
            warnings: List of warning messages
            config_path: Path to the configuration file

        Returns:
            Comprehensive report dictionary
        """
        warnings = warnings or []

        # Convert errors to detailed reports
        error_reports = [self._create_error_report(error) for error in errors]
        warning_reports = [
            self._create_warning_report(warning) for warning in warnings
        ]

        # Categorize and prioritize
        categorized_errors = self._categorize_reports(error_reports)
        categorized_warnings = self._categorize_reports(warning_reports)

        # Generate summary statistics
        summary = self._generate_summary(error_reports, warning_reports)

        # Create actionable recommendations
        recommendations = self._generate_recommendations(
            error_reports, warning_reports
        )

        return {
            "summary": summary,
            "errors": {
                "total": len(error_reports),
                "by_category": categorized_errors,
                "details": error_reports,
            },
            "warnings": {
                "total": len(warning_reports),
                "by_category": categorized_warnings,
                "details": warning_reports,
            },
            "recommendations": recommendations,
            "config_path": config_path,
            "timestamp": self._get_timestamp(),
        }

    def format_error_report(
        self, report: dict[str, Any], format_type: str = "text"
    ) -> str:
        """
        Format error report for display.

        Args:
            report: Error report dictionary
            format_type: Output format ("text", "html", "markdown")

        Returns:
            Formatted report string
        """
        if format_type == "html":
            return self._format_html_report(report)
        elif format_type == "markdown":
            return self._format_markdown_report(report)
        else:
            return self._format_text_report(report)

    def get_quick_fixes(self, error: ValidationError) -> list[str]:
        """
        Get quick fix suggestions for a specific error.

        Args:
            error: Validation error to analyze

        Returns:
            List of quick fix suggestions
        """
        fixes = []

        # Pattern-based fixes
        for pattern, fix_template in self.suggestion_templates.items():
            if pattern in error.message.lower():
                fixes.extend(fix_template)

        # Field-specific fixes
        if error.field:
            field_fixes = self._get_field_specific_fixes(
                error.field, error.message
            )
            fixes.extend(field_fixes)

        # Add existing suggestions from error
        if hasattr(error, "suggestions") and error.suggestions:
            fixes.extend(error.suggestions)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(fixes))

    def _create_error_report(self, error: ValidationError) -> ErrorReport:
        """Create detailed error report from validation error."""
        # Determine category and severity
        category = self._determine_category(error)
        severity = self._determine_severity(error, category)

        # Generate enhanced suggestions
        suggestions = self.get_quick_fixes(error)

        # Generate examples
        examples = self._generate_examples(error)

        # Get related documentation
        docs = self._get_related_documentation(error)

        return ErrorReport(
            title=self._generate_error_title(error),
            description=error.message,
            severity=severity,
            category=category,
            field_name=error.field,
            line=getattr(error, "line", None),
            column=getattr(error, "column", None),
            suggestions=suggestions,
            examples=examples,
            related_docs=docs,
            context={"original_error": error},
        )

    def _create_warning_report(self, warning: str) -> ErrorReport:
        """Create warning report from warning message."""
        return ErrorReport(
            title="Configuration Warning",
            description=warning,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.PERFORMANCE,
            suggestions=self._extract_warning_suggestions(warning),
        )

    def _determine_category(self, error: ValidationError) -> ErrorCategory:
        """Determine error category based on error characteristics."""
        message = error.message.lower()

        if any(
            keyword in message for keyword in ["yaml", "syntax", "parsing"]
        ):
            return ErrorCategory.SYNTAX
        elif any(
            keyword in message for keyword in ["missing", "required", "schema"]
        ):
            return ErrorCategory.SCHEMA
        elif any(
            keyword in message
            for keyword in ["type", "integer", "float", "string"]
        ):
            return ErrorCategory.TYPE
        elif any(
            keyword in message
            for keyword in ["unknown", "invalid", "unsupported"]
        ):
            return ErrorCategory.VALUE
        elif any(
            keyword in message for keyword in ["incompatible", "combination"]
        ):
            return ErrorCategory.COMPATIBILITY
        elif any(
            keyword in message for keyword in ["vram", "memory", "batch"]
        ):
            return ErrorCategory.PERFORMANCE
        else:
            return ErrorCategory.SCHEMA

    def _determine_severity(
        self, error: ValidationError, category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity based on category and content."""
        message = error.message.lower()

        # Critical errors that prevent execution
        if any(
            keyword in message
            for keyword in ["critical", "fatal", "cannot load"]
        ):
            return ErrorSeverity.CRITICAL

        # Category-based severity
        if category in [ErrorCategory.SYNTAX, ErrorCategory.SCHEMA]:
            return ErrorSeverity.ERROR
        elif category == ErrorCategory.COMPATIBILITY:
            return ErrorSeverity.ERROR
        elif category in [ErrorCategory.TYPE, ErrorCategory.VALUE]:
            return ErrorSeverity.ERROR
        elif category == ErrorCategory.PERFORMANCE:
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.ERROR

    def _categorize_reports(
        self, reports: list[ErrorReport]
    ) -> dict[str, list[ErrorReport]]:
        """Categorize error reports by category."""
        categorized = {}
        for report in reports:
            category_name = report.category.value
            if category_name not in categorized:
                categorized[category_name] = []
            categorized[category_name].append(report)
        return categorized

    def _generate_summary(
        self,
        error_reports: list[ErrorReport],
        warning_reports: list[ErrorReport],
    ) -> dict[str, Any]:
        """Generate summary statistics."""
        total_errors = len(error_reports)
        total_warnings = len(warning_reports)

        # Count by severity
        severity_counts = {}
        for report in error_reports + warning_reports:
            severity = report.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Count by category
        category_counts = {}
        for report in error_reports + warning_reports:
            category = report.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

        # Determine overall status
        if total_errors == 0:
            status = "valid" if total_warnings == 0 else "valid_with_warnings"
        else:
            critical_errors = sum(
                1
                for r in error_reports
                if r.severity == ErrorSeverity.CRITICAL
            )
            status = "critical" if critical_errors > 0 else "invalid"

        return {
            "status": status,
            "total_issues": total_errors + total_warnings,
            "errors": total_errors,
            "warnings": total_warnings,
            "by_severity": severity_counts,
            "by_category": category_counts,
            "is_valid": total_errors == 0,
        }

    def _generate_recommendations(
        self,
        error_reports: list[ErrorReport],
        warning_reports: list[ErrorReport],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Priority-based recommendations
        critical_errors = [
            r for r in error_reports if r.severity == ErrorSeverity.CRITICAL
        ]
        if critical_errors:
            recommendations.append(
                "🚨 CRITICAL: Fix critical errors before proceeding with "
                "training"
            )

        # Category-specific recommendations
        syntax_errors = [
            r for r in error_reports if r.category == ErrorCategory.SYNTAX
        ]
        if syntax_errors:
            recommendations.append(
                "📝 Fix YAML syntax errors using a YAML validator or linter"
            )

        schema_errors = [
            r for r in error_reports if r.category == ErrorCategory.SCHEMA
        ]
        if schema_errors:
            recommendations.append(
                "🔧 Add missing required configuration sections and fields"
            )

        type_errors = [
            r for r in error_reports if r.category == ErrorCategory.TYPE
        ]
        if type_errors:
            recommendations.append(
                "🔢 Verify data types match expected formats (int, float, str)"
            )

        # Performance warnings
        perf_warnings = [
            r
            for r in warning_reports
            if r.category == ErrorCategory.PERFORMANCE
        ]
        if perf_warnings:
            recommendations.append(
                "⚡ Consider performance optimizations to prevent VRAM issues"
            )

        # General recommendations
        if error_reports or warning_reports:
            recommendations.extend(
                [
                    "📚 Check example configurations in configs/ directory",
                    "🔍 Use the config validation panel for real-time help",
                    "💾 Save working configs as templates for later use",
                ]
            )

        return recommendations

    def _generate_error_title(self, error: ValidationError) -> str:
        """Generate descriptive title for error."""
        if error.field:
            return f"Configuration Error in '{error.field}'"
        else:
            return "Configuration Validation Error"

    def _generate_examples(self, error: ValidationError) -> list[str]:
        """Generate example fixes for error."""
        examples = []

        if error.field:
            field_parts = error.field.split(".")
            if len(field_parts) >= 2:
                field_name = field_parts[-1]

                # Generate examples based on field type
                if field_name in ["epochs", "batch_size", "num_workers"]:
                    examples.append(f"{error.field}: 100")
                elif field_name in ["learning_rate", "weight_decay"]:
                    examples.append(f"{error.field}: 0.001")
                elif field_name in ["architecture", "encoder", "decoder"]:
                    examples.append(f"{error.field}: unet")
                elif field_name == "image_size":
                    examples.append(f"{error.field}: [512, 512]")

        return examples

    def _get_related_documentation(self, error: ValidationError) -> list[str]:
        """Get related documentation links."""
        docs = []

        if error.field:
            field_parts = error.field.split(".")
            if field_parts:
                section = field_parts[0]
                docs.extend(self.documentation_links.get(section, []))

        return docs

    def _get_field_specific_fixes(self, field: str, message: str) -> list[str]:
        """Get field-specific fix suggestions."""
        fixes = []

        if "missing" in message.lower():
            fixes.append(f"Add '{field}:' to your configuration")

            # Suggest specific values based on field name
            if "architecture" in field:
                fixes.append("Try: architecture: unet")
            elif "encoder" in field:
                fixes.append("Try: encoder: resnet50")
            elif "decoder" in field:
                fixes.append("Try: decoder: unet")
            elif "epochs" in field:
                fixes.append("Try: epochs: 100")
            elif "learning_rate" in field:
                fixes.append("Try: learning_rate: 0.001")

        return fixes

    def _extract_warning_suggestions(self, warning: str) -> list[str]:
        """Extract suggestions from warning messages."""
        suggestions = []

        warning_lower = warning.lower()

        if "vram" in warning_lower or "memory" in warning_lower:
            suggestions.extend(
                [
                    "Reduce batch_size to fit in available VRAM",
                    "Consider using gradient accumulation",
                    "Enable mixed precision training (use_amp: true)",
                ]
            )

        if "batch" in warning_lower:
            suggestions.append("Adjust batch_size for optimal performance")

        if "learning rate" in warning_lower:
            suggestions.extend(
                [
                    "Try learning rates between 1e-5 and 1e-2",
                    "Use learning rate scheduling for better convergence",
                ]
            )

        return suggestions

    def _format_text_report(self, report: dict[str, Any]) -> str:
        """Format report as plain text."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("Configuration Validation Report")
        lines.append("=" * 60)

        # Summary
        summary = report["summary"]
        lines.append(f"\nStatus: {summary['status'].upper()}")
        lines.append(f"Total Issues: {summary['total_issues']}")
        lines.append(
            f"Errors: {summary['errors']}, Warnings: {summary['warnings']}"
        )

        # Errors
        if report["errors"]["total"] > 0:
            lines.append("\n" + "=" * 30 + " ERRORS " + "=" * 30)
            for error_report in report["errors"]["details"]:
                lines.append(f"\n❌ {error_report.title}")
                lines.append(f"   {error_report.description}")
                if error_report.suggestions:
                    lines.append("   Suggestions:")
                    for suggestion in error_report.suggestions[
                        :3
                    ]:  # Limit to 3
                        lines.append(f"   • {suggestion}")

        # Warnings
        if report["warnings"]["total"] > 0:
            lines.append("\n" + "=" * 30 + " WARNINGS " + "=" * 30)
            for warning_report in report["warnings"]["details"]:
                lines.append(f"\n⚠️  {warning_report.title}")
                lines.append(f"   {warning_report.description}")

        # Recommendations
        if report["recommendations"]:
            lines.append("\n" + "=" * 30 + " RECOMMENDATIONS " + "=" * 30)
            for rec in report["recommendations"]:
                lines.append(f"• {rec}")

        return "\n".join(lines)

    def _format_html_report(self, report: dict[str, Any]) -> str:
        """Format report as HTML."""
        # Basic HTML formatting - can be enhanced as needed
        status = report["summary"]["status"]
        total = report["summary"]["total_issues"]
        html = f"""
        <div class="config-validation-report">
            <h2>Configuration Validation Report</h2>
            <div class="summary">
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Total:</strong> {total}</p>
            </div>
        """

        if report["errors"]["total"] > 0:
            html += "<div class='errors'><h3>Errors</h3><ul>"
            for error in report["errors"]["details"]:
                html += (
                    f"<li><strong>{error.title}</strong>"
                    f"<br>{error.description}</li>"
                )
            html += "</ul></div>"

        if report["warnings"]["total"] > 0:
            html += "<div class='warnings'><h3>Warnings</h3><ul>"
            for warning in report["warnings"]["details"]:
                html += (
                    f"<li><strong>{warning.title}</strong>"
                    f"<br>{warning.description}</li>"
                )
            html += "</ul></div>"

        html += "</div>"
        return html

    def _format_markdown_report(self, report: dict[str, Any]) -> str:
        """Format report as Markdown."""
        lines = []

        lines.append("# Configuration Validation Report")
        lines.append(f"\n**Status:** {report['summary']['status']}")
        lines.append(f"**Total Issues:** {report['summary']['total_issues']}")

        if report["errors"]["total"] > 0:
            lines.append("\n## Errors")
            for error in report["errors"]["details"]:
                lines.append(f"\n### ❌ {error.title}")
                lines.append(f"{error.description}")
                if error.suggestions:
                    lines.append("\n**Suggestions:**")
                    for suggestion in error.suggestions:
                        lines.append(f"- {suggestion}")

        return "\n".join(lines)

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _initialize_error_patterns(self) -> dict[str, str]:
        """Initialize common error patterns."""
        return {
            "missing required": "Add the missing field to your configuration",
            "unknown": "Check valid options in documentation",
            "invalid type": "Verify the data type matches requirements",
            "yaml error": "Fix YAML syntax using a validator",
        }

    def _initialize_suggestion_templates(self) -> dict[str, list[str]]:
        """Initialize suggestion templates for common issues."""
        return {
            "missing": [
                "Add the missing field to your configuration",
                "Check example configurations in configs/ directory",
            ],
            "unknown": [
                "Verify the value is supported",
                "Check documentation for valid options",
            ],
            "type": [
                "Check data type requirements",
                "Use appropriate format (int, float, string, list)",
            ],
            "vram": [
                "Reduce batch_size to fit in available VRAM",
                "Enable mixed precision training",
            ],
        }

    def _initialize_documentation_links(self) -> dict[str, list[str]]:
        """Initialize documentation links for different sections."""
        return {
            "model": [
                "configs/model/README.md",
                "docs/guides/model_configuration.md",
            ],
            "training": [
                "configs/training/README.md",
                "docs/guides/training_configuration.md",
            ],
            "data": [
                "configs/data/README.md",
                "docs/guides/data_configuration.md",
            ],
        }


# Global reporter instance
_error_reporter = ConfigErrorReporter()


def generate_error_report(
    errors: list[ValidationError],
    warnings: list[str] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function for generating comprehensive error reports.

    Args:
        errors: List of validation errors
        warnings: List of warning messages
        config_path: Path to configuration file

    Returns:
        Comprehensive error report dictionary
    """
    return _error_reporter.generate_comprehensive_report(
        errors, warnings, config_path
    )
