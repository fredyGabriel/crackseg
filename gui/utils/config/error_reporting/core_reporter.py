"""
Core error reporter for configuration validation issues.

This module contains the main error reporting logic with intelligent
suggestions and context-aware recommendations.
"""

import logging
from typing import Any

from ..exceptions import ValidationError
from .report_models import ErrorCategory, ErrorReport, ErrorSeverity

logger = logging.getLogger(__name__)


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

        # Categorize reports
        categorized_errors = self._categorize_reports(error_reports)
        categorized_warnings = self._categorize_reports(warning_reports)

        # Generate summary and recommendations
        summary = self._generate_summary(error_reports, warning_reports)
        recommendations = self._generate_recommendations(
            error_reports, warning_reports
        )

        return {
            "summary": summary,
            "errors": {
                "total": len(error_reports),
                "categorized": categorized_errors,
                "reports": error_reports,
            },
            "warnings": {
                "total": len(warning_reports),
                "categorized": categorized_warnings,
                "reports": warning_reports,
            },
            "recommendations": recommendations,
            "config_path": config_path,
            "timestamp": self._get_timestamp(),
        }

    def get_quick_fixes(self, error: ValidationError) -> list[str]:
        """
        Get quick fix suggestions for a specific error.

        Args:
            error: Validation error to analyze

        Returns:
            List of quick fix suggestions
        """
        suggestions = []

        # Add field-specific fixes
        if error.field:
            field_fixes = self._get_field_specific_fixes(
                error.field, error.message
            )
            suggestions.extend(field_fixes)

        # Add pattern-based suggestions
        for pattern, suggestion in self.error_patterns.items():
            if pattern.lower() in error.message.lower():
                suggestions.append(suggestion)

        # Add generic suggestions based on error type
        if "must be" in error.message.lower():
            suggestions.append(
                "Check the data type and ensure it matches the expected format"
            )
        elif "missing" in error.message.lower():
            suggestions.append("Add the missing field to your configuration")
        elif "invalid" in error.message.lower():
            suggestions.append(
                "Verify the value is within the allowed range or options"
            )

        return list(set(suggestions))  # Remove duplicates

    def _create_error_report(self, error: ValidationError) -> ErrorReport:
        """Create a detailed error report from a validation error."""
        category = self._determine_category(error)
        severity = self._determine_severity(error, category)

        return ErrorReport(
            title=self._generate_error_title(error),
            description=error.message,
            severity=severity,
            category=category,
            field_name=error.field,
            suggestions=self.get_quick_fixes(error),
            examples=self._generate_examples(error),
            related_docs=self._get_related_documentation(error),
            context={"value": error.value},
        )

    def _create_warning_report(self, warning: str) -> ErrorReport:
        """Create a warning report from a warning message."""
        return ErrorReport(
            title="Configuration Warning",
            description=warning,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.PERFORMANCE,
            suggestions=self._extract_warning_suggestions(warning),
        )

    def _determine_category(self, error: ValidationError) -> ErrorCategory:
        """Determine the category of an error based on its characteristics."""
        message = error.message.lower()

        if "type" in message or "must be" in message:
            return ErrorCategory.TYPE
        elif "missing" in message or "required" in message:
            return ErrorCategory.SCHEMA
        elif "invalid" in message or "not allowed" in message:
            return ErrorCategory.VALUE
        elif "compatibility" in message or "requires" in message:
            return ErrorCategory.COMPATIBILITY
        elif "performance" in message or "memory" in message:
            return ErrorCategory.PERFORMANCE
        elif "security" in message or "permission" in message:
            return ErrorCategory.SECURITY
        elif "syntax" in message or "format" in message:
            return ErrorCategory.SYNTAX
        else:
            return ErrorCategory.SCHEMA

    def _determine_severity(
        self, error: ValidationError, category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine the severity of an error."""
        message = error.message.lower()

        # Critical errors
        if any(
            keyword in message for keyword in ["critical", "fatal", "security"]
        ):
            return ErrorSeverity.CRITICAL

        # Performance warnings
        if category == ErrorCategory.PERFORMANCE:
            return ErrorSeverity.WARNING

        # Type and schema errors are typically errors
        if category in {ErrorCategory.TYPE, ErrorCategory.SCHEMA}:
            return ErrorSeverity.ERROR

        # Value errors can be warnings or errors
        if category == ErrorCategory.VALUE:
            if "recommended" in message or "consider" in message:
                return ErrorSeverity.WARNING
            return ErrorSeverity.ERROR

        return ErrorSeverity.ERROR

    def _categorize_reports(
        self, reports: list[ErrorReport]
    ) -> dict[str, list[ErrorReport]]:
        """Categorize reports by severity and category."""
        categorized = {}
        for report in reports:
            key = f"{report.severity.value}_{report.category.value}"
            if key not in categorized:
                categorized[key] = []
            categorized[key].append(report)
        return categorized

    def _generate_summary(
        self,
        error_reports: list[ErrorReport],
        warning_reports: list[ErrorReport],
    ) -> dict[str, Any]:
        """Generate a summary of the error reports."""
        total_errors = len(error_reports)
        total_warnings = len(warning_reports)

        severity_counts = {}
        category_counts = {}

        for report in error_reports + warning_reports:
            severity_counts[report.severity.value] = (
                severity_counts.get(report.severity.value, 0) + 1
            )
            category_counts[report.category.value] = (
                category_counts.get(report.category.value, 0) + 1
            )

        return {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "has_critical_errors": any(
                r.severity == ErrorSeverity.CRITICAL for r in error_reports
            ),
            "overall_status": (
                "critical"
                if any(
                    r.severity == ErrorSeverity.CRITICAL for r in error_reports
                )
                else (
                    "error"
                    if total_errors > 0
                    else "warning" if total_warnings > 0 else "success"
                )
            ),
        }

    def _generate_recommendations(
        self,
        error_reports: list[ErrorReport],
        warning_reports: list[ErrorReport],
    ) -> list[str]:
        """Generate actionable recommendations based on errors and warnings."""
        recommendations = []

        # Count error types
        error_types = {}
        for report in error_reports:
            error_type = report.category.value
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Generate recommendations based on error patterns
        if error_types.get("schema", 0) > 0:
            recommendations.append(
                "Review your configuration structure and ensure all required "
                "fields are present"
            )

        if error_types.get("type", 0) > 0:
            recommendations.append(
                "Check data types in your configuration - ensure strings, "
                "numbers, and booleans are used correctly"
            )

        if error_types.get("value", 0) > 0:
            recommendations.append(
                "Verify parameter values are within acceptable ranges and "
                "match expected formats"
            )

        if error_types.get("compatibility", 0) > 0:
            recommendations.append(
                "Check component compatibility - ensure encoder/decoder "
                "combinations are valid"
            )

        # Add performance recommendations
        performance_warnings = [
            r
            for r in warning_reports
            if r.category == ErrorCategory.PERFORMANCE
        ]
        if performance_warnings:
            recommendations.append(
                "Consider performance optimizations - batch sizes, image "
                "sizes, and model complexity"
            )

        # Add general recommendations
        if len(error_reports) > 5:
            recommendations.append(
                "Consider using the configuration wizard to generate a "
                "valid base configuration"
            )

        if not recommendations:
            recommendations.append(
                "Configuration appears to be valid with minor warnings"
            )

        return recommendations

    def _generate_error_title(self, error: ValidationError) -> str:
        """Generate a concise title for an error."""
        if error.field:
            return f"Error in {error.field}"
        return "Configuration Error"

    def _generate_examples(self, error: ValidationError) -> list[str]:
        """Generate example fixes for an error."""
        examples = []

        field = error.field or ""
        field_lower = field.lower()

        if "architecture" in field_lower:
            examples.extend(
                [
                    "architecture: unet",
                    "architecture: swin_unet",
                    "architecture: deeplabv3plus",
                ]
            )
        elif "encoder" in field_lower:
            examples.extend(
                [
                    "encoder: resnet50",
                    "encoder: efficientnet_b0",
                    "encoder: swin_transformer",
                ]
            )
        elif "decoder" in field_lower:
            examples.extend(
                [
                    "decoder: unet",
                    "decoder: fpn",
                    "decoder: deeplabv3plus",
                ]
            )
        elif "batch_size" in field_lower:
            examples.extend(
                [
                    "batch_size: 8",
                    "batch_size: 16",
                    "batch_size: 32",
                ]
            )
        elif "learning_rate" in field_lower:
            examples.extend(
                [
                    "learning_rate: 0.001",
                    "learning_rate: 0.0001",
                    "learning_rate: 0.00001",
                ]
            )

        return examples

    def _get_related_documentation(self, error: ValidationError) -> list[str]:
        """Get related documentation links for an error."""
        field = error.field or ""
        message = error.message.lower()

        docs = []

        if "architecture" in field or "model" in message:
            docs.extend(
                [
                    "docs/guides/model_architectures.md",
                    "docs/guides/encoder_decoder_combinations.md",
                ]
            )

        if "training" in field or "optimizer" in message:
            docs.extend(
                [
                    "docs/guides/training_configuration.md",
                    "docs/guides/optimizer_selection.md",
                ]
            )

        if "data" in field or "dataset" in message:
            docs.extend(
                [
                    "docs/guides/data_preparation.md",
                    "docs/guides/augmentation.md",
                ]
            )

        return docs

    def _get_field_specific_fixes(self, field: str, message: str) -> list[str]:
        """Get field-specific fix suggestions."""
        fixes = []

        field_lower = field.lower()

        if "architecture" in field_lower:
            fixes.extend(
                [
                    "Choose from supported architectures: unet, swin_unet, "
                    "deeplabv3plus",
                    "Ensure architecture matches your encoder/decoder "
                    "combination",
                ]
            )

        if "batch_size" in field_lower:
            fixes.extend(
                [
                    "Reduce batch size for large images (512x512: max 16, "
                    "1024x1024: max 4)",
                    "Consider your GPU memory limits (RTX 3070 Ti: 8GB VRAM)",
                ]
            )

        if "learning_rate" in field_lower:
            fixes.extend(
                [
                    "Use learning rate between 1e-6 and 1e-3",
                    "Start with 1e-4 and adjust based on training progress",
                ]
            )

        if "image_size" in field_lower:
            fixes.extend(
                [
                    "Use square images (e.g., 512x512, 1024x1024)",
                    "Ensure size is power of 2 for optimal performance",
                ]
            )

        return fixes

    def _extract_warning_suggestions(self, warning: str) -> list[str]:
        """Extract suggestions from warning messages."""
        suggestions = []

        if "batch size" in warning.lower():
            suggestions.append(
                "Consider reducing batch size to avoid memory issues"
            )
        elif "learning rate" in warning.lower():
            suggestions.append(
                "Consider reducing learning rate for better convergence"
            )
        elif "epochs" in warning.lower():
            suggestions.append(
                "Consider using early stopping to prevent overfitting"
            )
        elif "image size" in warning.lower():
            suggestions.append(
                "Consider using standard image sizes (256, 512, 1024)"
            )

        return suggestions

    def _get_timestamp(self) -> str:
        """Get current timestamp for reports."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _initialize_error_patterns(self) -> dict[str, str]:
        """Initialize common error patterns and their suggestions."""
        return {
            "must be a string": "Ensure the value is enclosed in quotes",
            "must be an integer": "Remove quotes and ensure it's a whole "
            "number",
            "must be a number": "Remove quotes and ensure it's a valid number",
            "missing required": "Add the missing field to your configuration",
            "invalid": "Check the field value against valid options",
        }

    def _initialize_suggestion_templates(self) -> dict[str, list[str]]:
        """Initialize suggestion templates for different error types."""
        return {
            "type_error": [
                "Check the data type of the field",
                "Ensure the value matches the expected format",
                "Remove quotes for numbers, add quotes for strings",
            ],
            "missing_field": [
                "Add the missing field to your configuration",
                "Check the documentation for required fields",
                "Use a configuration template as a starting point",
            ],
            "invalid_value": [
                "Check the allowed values for this field",
                "Ensure the value is within the valid range",
                "Refer to the documentation for valid options",
            ],
        }

    def _initialize_documentation_links(self) -> dict[str, list[str]]:
        """Initialize documentation links for different topics."""
        return {
            "architecture": "docs/guides/model_architectures.md",
            "encoder": "docs/guides/encoder_selection.md",
            "decoder": "docs/guides/decoder_selection.md",
            "training": "docs/guides/training_configuration.md",
            "data": "docs/guides/data_preparation.md",
        }
