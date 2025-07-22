"""
Validation reporting and formatting utilities. This module provides
functions for formatting validation errors into readable reports and
extracting actionable suggestions from validation results.
"""

from .exceptions import ValidationError


def get_validation_suggestions(
    errors: list[ValidationError],
) -> dict[str, list[str]]:
    """
    Extract and organize validation suggestions by category. Args: errors:
    List of validation errors. Returns: Dictionary mapping error
    categories to suggestions.
    """
    suggestions_by_category: dict[str, list[str]] = {
        "syntax": [],
        "structure": [],
        "types": [],
        "values": [],
        "general": [],
    }

    for error in errors:
        category = "general"
        if "syntax" in str(error).lower():
            category = "syntax"
        elif error.field and (
            "model" in error.field
            or "training" in error.field
            or "data" in error.field
        ):
            if "missing" in str(error).lower():
                category = "structure"
            elif "type" in str(error).lower():
                category = "types"
            else:
                category = "values"

        suggestions_by_category[category].extend(error.suggestions)

    # Remove empty categories and duplicates
    return {
        cat: list(set(suggs))
        for cat, suggs in suggestions_by_category.items()
        if suggs
    }


def format_validation_report(errors: list[ValidationError]) -> str:
    """
    Format validation errors into a human-readable report. Args: errors:
    List of validation errors. Returns: Formatted validation report as
    string.
    """
    if not errors:
        return "✅ Configuration validation passed successfully!"

    report_lines = [
        f"❌ Configuration validation failed with {len(errors)} error(s):",
        "",
    ]

    # Group errors by type
    syntax_errors = [e for e in errors if "syntax" in str(e).lower()]
    structure_errors = [
        e for e in errors if e.field and "missing" in str(e).lower()
    ]
    type_errors = [e for e in errors if "type" in str(e).lower()]
    value_errors = [
        e
        for e in errors
        if e not in syntax_errors + structure_errors + type_errors
    ]

    # Report syntax errors first
    if syntax_errors:
        report_lines.extend(
            [
                "🔍 Syntax Errors:",
                *[f"  • {error}" for error in syntax_errors],
                "",
            ]
        )

    # Report structure errors
    if structure_errors:
        report_lines.extend(
            [
                "🏗️ Structure Errors:",
                *[f"  • {error}" for error in structure_errors],
                "",
            ]
        )

    # Report type errors
    if type_errors:
        report_lines.extend(
            ["🔢 Type Errors:", *[f"  • {error}" for error in type_errors], ""]
        )

    # Report value errors
    if value_errors:
        report_lines.extend(
            [
                "⚠️ Value Errors:",
                *[f"  • {error}" for error in value_errors],
                "",
            ]
        )

    # Add summary suggestions
    all_suggestions = get_validation_suggestions(errors)
    if all_suggestions:
        report_lines.append("💡 Quick Fixes:")
        for category, suggestions in all_suggestions.items():
            if suggestions:
                report_lines.append(f"  {category.title()}:")
                for suggestion in suggestions[
                    :3
                ]:  # Limit to top 3 suggestions
                    report_lines.append(f"    - {suggestion}")
        report_lines.append("")

    return "\n".join(report_lines)
