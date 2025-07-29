"""
Utility functions for error reporting.

This module provides convenience functions and utilities for error reporting
operations.
"""

from typing import Any

from ..exceptions import ValidationError
from .core_reporter import ConfigErrorReporter


def generate_error_report(
    errors: list[ValidationError],
    warnings: list[str] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function to generate a comprehensive error report.

    Args:
        errors: List of validation errors
        warnings: List of warning messages
        config_path: Path to the configuration file

    Returns:
        Comprehensive report dictionary
    """
    reporter = ConfigErrorReporter()
    return reporter.generate_comprehensive_report(
        errors, warnings, config_path
    )


def format_error_report(
    report: dict[str, Any], format_type: str = "text"
) -> str:
    """
    Format an error report in the specified format.

    Args:
        report: Error report dictionary
        format_type: Format type ("text", "html", "markdown")

    Returns:
        Formatted report string
    """
    from .formatters import ErrorReportFormatter

    formatter = ErrorReportFormatter()
    return formatter.format_report(report, format_type)


def get_quick_fixes(error: ValidationError) -> list[str]:
    """
    Get quick fix suggestions for a specific error.

    Args:
        error: Validation error to analyze

    Returns:
        List of quick fix suggestions
    """
    reporter = ConfigErrorReporter()
    return reporter.get_quick_fixes(error)


def categorize_errors(
    errors: list[ValidationError],
) -> dict[str, list[ValidationError]]:
    """
    Categorize errors by type for better organization.

    Args:
        errors: List of validation errors

    Returns:
        Dictionary mapping error categories to lists of errors
    """
    categories = {
        "type_errors": [],
        "schema_errors": [],
        "value_errors": [],
        "compatibility_errors": [],
        "performance_errors": [],
        "other_errors": [],
    }

    for error in errors:
        message = error.message.lower()

        if "type" in message or "must be" in message:
            categories["type_errors"].append(error)
        elif "missing" in message or "required" in message:
            categories["schema_errors"].append(error)
        elif "invalid" in message or "not allowed" in message:
            categories["value_errors"].append(error)
        elif "compatibility" in message or "requires" in message:
            categories["compatibility_errors"].append(error)
        elif "performance" in message or "memory" in message:
            categories["performance_errors"].append(error)
        else:
            categories["other_errors"].append(error)

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def get_error_summary(errors: list[ValidationError]) -> dict[str, Any]:
    """
    Generate a summary of validation errors.

    Args:
        errors: List of validation errors

    Returns:
        Summary dictionary
    """
    if not errors:
        return {
            "total_errors": 0,
            "error_types": {},
            "critical_errors": 0,
            "status": "success",
        }

    # Count error types
    error_types = {}
    critical_errors = 0

    for error in errors:
        message = error.message.lower()

        # Determine error type
        if "type" in message or "must be" in message:
            error_type = "type"
        elif "missing" in message or "required" in message:
            error_type = "schema"
        elif "invalid" in message or "not allowed" in message:
            error_type = "value"
        elif "compatibility" in message or "requires" in message:
            error_type = "compatibility"
        elif "performance" in message or "memory" in message:
            error_type = "performance"
        else:
            error_type = "other"

        error_types[error_type] = error_types.get(error_type, 0) + 1

        # Count critical errors
        if any(
            keyword in message for keyword in ["critical", "fatal", "security"]
        ):
            critical_errors += 1

    return {
        "total_errors": len(errors),
        "error_types": error_types,
        "critical_errors": critical_errors,
        "status": (
            "critical"
            if critical_errors > 0
            else "error" if errors else "success"
        ),
    }


def get_field_specific_suggestions(field: str, message: str) -> list[str]:
    """
    Get field-specific suggestions for configuration errors.

    Args:
        field: Field name that caused the error
        message: Error message

    Returns:
        List of field-specific suggestions
    """
    suggestions = []

    field_lower = field.lower()
    message_lower = message.lower()

    # Architecture-specific suggestions
    if "architecture" in field_lower:
        suggestions.extend(
            [
                "Choose from supported architectures: unet, swin_unet, "
                "deeplabv3plus",
                "Ensure architecture matches your encoder/decoder combination",
                "Check documentation for architecture requirements",
            ]
        )

    # Encoder-specific suggestions
    if "encoder" in field_lower:
        suggestions.extend(
            [
                "Choose from supported encoders: resnet50, efficientnet_b0, "
                "swin_transformer",
                "Ensure encoder is compatible with your chosen architecture",
                "Consider model complexity and memory requirements",
            ]
        )

    # Decoder-specific suggestions
    if "decoder" in field_lower:
        suggestions.extend(
            [
                "Choose from supported decoders: unet, fpn, deeplabv3plus",
                "Ensure decoder is compatible with your encoder",
                "Consider the segmentation task requirements",
            ]
        )

    # Training-specific suggestions
    if "batch_size" in field_lower:
        suggestions.extend(
            [
                "Reduce batch size for large images (512x512: max 16, "
                "1024x1024: max 4)",
                "Consider your GPU memory limits (RTX 3070 Ti: 8GB VRAM)",
                "Start with smaller batch sizes and increase gradually",
            ]
        )

    if "learning_rate" in field_lower:
        suggestions.extend(
            [
                "Use learning rate between 1e-6 and 1e-3",
                "Start with 1e-4 and adjust based on training progress",
                "Consider using learning rate scheduling",
            ]
        )

    if "epochs" in field_lower:
        suggestions.extend(
            [
                "Use between 50-500 epochs for most tasks",
                "Consider using early stopping to prevent overfitting",
                "Monitor validation metrics to determine optimal epoch count",
            ]
        )

    # Data-specific suggestions
    if "image_size" in field_lower:
        suggestions.extend(
            [
                "Use square images (e.g., 512x512, 1024x1024)",
                "Ensure size is power of 2 for optimal performance",
                "Consider memory constraints when choosing image size",
            ]
        )

    if "data_path" in field_lower or "dataset" in field_lower:
        suggestions.extend(
            [
                "Ensure the data path exists and is accessible",
                "Check that the dataset follows the expected structure",
                "Verify that image and mask files are properly paired",
            ]
        )

    # Generic suggestions based on error type
    if "must be" in message_lower:
        suggestions.append(
            "Check the data type and ensure it matches the expected format"
        )
    elif "missing" in message_lower:
        suggestions.append("Add the missing field to your configuration")
    elif "invalid" in message_lower:
        suggestions.append(
            "Verify the value is within the allowed range or options"
        )

    return list(set(suggestions))  # Remove duplicates


def create_error_context(error: ValidationError) -> dict[str, Any]:
    """
    Create context information for an error.

    Args:
        error: Validation error

    Returns:
        Context dictionary
    """
    context = {
        "field": error.field,
        "value": error.value,
        "message": error.message,
        "suggestions": get_field_specific_suggestions(
            error.field or "", error.message
        ),
    }

    # Add field-specific context
    if error.field:
        field_lower = error.field.lower()

        if "architecture" in field_lower:
            context["valid_options"] = [
                "unet",
                "swin_unet",
                "deeplabv3plus",
                "fcn",
                "segnet",
            ]
        elif "encoder" in field_lower:
            context["valid_options"] = [
                "resnet50",
                "efficientnet_b0",
                "swin_transformer",
                "cnn",
            ]
        elif "decoder" in field_lower:
            context["valid_options"] = ["unet", "fpn", "deeplabv3plus", "pan"]
        elif "batch_size" in field_lower:
            context["recommended_range"] = "1-32"
            context["memory_consideration"] = "Consider GPU memory limits"
        elif "learning_rate" in field_lower:
            context["recommended_range"] = "1e-6 to 1e-3"
            context["typical_start"] = "1e-4"

    return context
